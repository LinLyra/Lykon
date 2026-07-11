#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "esp_mac.h"

#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "esp_now.h"

#include "lwip/sockets.h"
#include "lwip/inet.h"

#include "esp32_s3_szp.h"


// ====== 每套主节点烧录前改这里 ======
// 第一套设备主节点：GROUP_ID = 1，NODE_ID = 1
// 第二套设备主节点：GROUP_ID = 2，NODE_ID = 1
#define GROUP_ID 1
#define NODE_ID  1

#if NODE_ID != 1
#error "Master node must use NODE_ID 1"
#endif

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define ESPNOW_CHANNEL 1

#define START_INTERVAL_MS       1000   // 每 1 秒发送一次 START/心跳
#define IMU_SAMPLE_INTERVAL_MS  20     // 主节点自己采样间隔，20ms = 50Hz

#define MSG_START 1
#define MSG_IMU   2

// 主节点 Wi-Fi 热点
#define WIFI_AP_SSID "IMU_MASTER_G" STR(GROUP_ID)
#define WIFI_AP_PASS "12345678"
#define WIFI_AP_MAX_CONN 4

// UDP 发电脑
#define UDP_PORT 3333
#define UDP_TARGET_IP "192.168.4.255"   // SoftAP 默认网段广播地址

#define IMU_QUEUE_LEN 128

static const char *TAG = "IMU_MASTER";

static const uint8_t BROADCAST_MAC[6] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

typedef struct __attribute__((packed))
{
    uint8_t group_id;       // 第几套设备
    uint8_t msg_type;       // MSG_START 或 MSG_IMU
    uint8_t node_id;        // 本组内节点编号
    uint32_t seq;           // 数据序号
    int64_t timestamp_us;   // 节点本地采样时间戳，单位 us

    int16_t acc_x;
    int16_t acc_y;
    int16_t acc_z;
    int16_t gyr_x;
    int16_t gyr_y;
    int16_t gyr_z;
} imu_msg_t;

static uint32_t g_start_seq = 0;
static uint32_t g_local_seq = 0;

static bool g_session_started = false;
static int64_t g_next_local_sample_us = 0;

static t_sQMI8658 QMI8658;

static QueueHandle_t s_imu_queue = NULL;

static int s_udp_sock = -1;
static struct sockaddr_in s_udp_dest_addr;


// ================= LCD 显示部分 =================

static void lcd_draw_7seg_digit(int x, int y, int w, int h, int t, uint8_t digit, uint16_t color)
{
    bool A = false;
    bool B = false;
    bool C = false;
    bool D = false;
    bool E = false;
    bool F = false;
    bool G = false;

    switch (digit) {
        case 0:
            A = B = C = D = E = F = true;
            break;
        case 1:
            B = C = true;
            break;
        case 2:
            A = B = D = E = G = true;
            break;
        case 3:
            A = B = C = D = G = true;
            break;
        case 4:
            B = C = F = G = true;
            break;
        case 5:
            A = C = D = F = G = true;
            break;
        case 6:
            A = C = D = E = F = G = true;
            break;
        case 7:
            A = B = C = true;
            break;
        case 8:
            A = B = C = D = E = F = G = true;
            break;
        case 9:
            A = B = C = D = F = G = true;
            break;
        default:
            G = true;
            break;
    }

    if (A) lcd_fill_rect(x + t,     y,                 x + w - t, y + t,             color);
    if (B) lcd_fill_rect(x + w - t, y + t,             x + w,     y + h / 2,         color);
    if (C) lcd_fill_rect(x + w - t, y + h / 2,         x + w,     y + h - t,         color);
    if (D) lcd_fill_rect(x + t,     y + h - t,         x + w - t, y + h,             color);
    if (E) lcd_fill_rect(x,         y + h / 2,         x + t,     y + h - t,         color);
    if (F) lcd_fill_rect(x,         y + t,             x + t,     y + h / 2,         color);
    if (G) lcd_fill_rect(x + t,     y + h / 2 - t / 2, x + w - t, y + h / 2 + t / 2, color);
}


static void lcd_show_group_node_id(uint8_t group_id, uint8_t node_id)
{
    lcd_fill_rect(0, 0, 320, 240, 0x0000);

    uint16_t white = 0xFFFF;

    lcd_draw_7seg_digit(45, 35, 90, 170, 16, group_id, white);
    lcd_draw_7seg_digit(185, 35, 90, 170, 16, node_id, white);
}


// ================= UDP 部分 =================

static esp_err_t udp_broadcast_init(void)
{
    s_udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (s_udp_sock < 0) {
        ESP_LOGE(TAG, "UDP socket create failed, errno=%d", errno);
        return ESP_FAIL;
    }

    int broadcast_enable = 1;
    int ret = setsockopt(
        s_udp_sock,
        SOL_SOCKET,
        SO_BROADCAST,
        &broadcast_enable,
        sizeof(broadcast_enable)
    );

    if (ret < 0) {
        ESP_LOGE(TAG, "UDP setsockopt SO_BROADCAST failed, errno=%d", errno);
        close(s_udp_sock);
        s_udp_sock = -1;
        return ESP_FAIL;
    }

    memset(&s_udp_dest_addr, 0, sizeof(s_udp_dest_addr));
    s_udp_dest_addr.sin_family = AF_INET;
    s_udp_dest_addr.sin_port = htons(UDP_PORT);
    s_udp_dest_addr.sin_addr.s_addr = inet_addr(UDP_TARGET_IP);

    ESP_LOGI(TAG, "UDP target: %s:%d", UDP_TARGET_IP, UDP_PORT);

    return ESP_OK;
}


static void udp_send_imu(const imu_msg_t *msg, int64_t master_timestamp_us)
{
    if (s_udp_sock < 0) {
        return;
    }

    char line[192];

    int len = snprintf(
        line,
        sizeof(line),
        "%d,%d,%lu,%lld,%lld,%d,%d,%d,%d,%d,%d\n",
        msg->group_id,
        msg->node_id,
        (unsigned long)msg->seq,
        (long long)msg->timestamp_us,
        (long long)master_timestamp_us,
        msg->acc_x,
        msg->acc_y,
        msg->acc_z,
        msg->gyr_x,
        msg->gyr_y,
        msg->gyr_z
    );

    if (len <= 0 || len >= sizeof(line)) {
        return;
    }

    int err = sendto(
        s_udp_sock,
        line,
        len,
        0,
        (struct sockaddr *)&s_udp_dest_addr,
        sizeof(s_udp_dest_addr)
    );

    if (err < 0) {
        static uint32_t fail_count = 0;

        if ((fail_count++ % 100) == 0) {
            ESP_LOGW(TAG, "UDP send failed, errno=%d", errno);
        }
    }
}


// ================= IMU 数据处理 =================

static void handle_imu_sample(const imu_msg_t *msg)
{
    int64_t master_timestamp_us = esp_timer_get_time();

    // 1. 发给电脑
    udp_send_imu(msg, master_timestamp_us);

    // 2. 串口少量打印
    if (msg->seq % 50 == 0) {
        ESP_LOGI(TAG,
                 "IMU group=%d node=%d seq=%lu node_ts=%lld master_ts=%lld acc=[%d,%d,%d] gyr=[%d,%d,%d]",
                 msg->group_id,
                 msg->node_id,
                 (unsigned long)msg->seq,
                 (long long)msg->timestamp_us,
                 (long long)master_timestamp_us,
                 msg->acc_x,
                 msg->acc_y,
                 msg->acc_z,
                 msg->gyr_x,
                 msg->gyr_y,
                 msg->gyr_z);
    }
}


static void collect_local_imu_data(void)
{
    if (!qmi8658_Read_AccAndGry(&QMI8658)) {
        static uint32_t fail_count = 0;

        if ((fail_count++ % 50) == 0) {
            ESP_LOGW(TAG, "Skip invalid local IMU sample");
        }

        return;
    }

    imu_msg_t msg = {0};

    msg.group_id = GROUP_ID;
    msg.msg_type = MSG_IMU;
    msg.node_id = NODE_ID;
    msg.seq = g_local_seq;
    msg.timestamp_us = esp_timer_get_time();

    msg.acc_x = QMI8658.acc_x;
    msg.acc_y = QMI8658.acc_y;
    msg.acc_z = QMI8658.acc_z;
    msg.gyr_x = QMI8658.gyr_x;
    msg.gyr_y = QMI8658.gyr_y;
    msg.gyr_z = QMI8658.gyr_z;

    g_local_seq++;

    handle_imu_sample(&msg);
}


static void process_received_imu_queue(void)
{
    imu_msg_t msg;

    while (xQueueReceive(s_imu_queue, &msg, 0) == pdTRUE) {
        handle_imu_sample(&msg);
    }
}


// ================= ESP-NOW 部分 =================

static void espnow_send_cb(const esp_now_send_info_t *tx_info, esp_now_send_status_t status)
{
    (void)tx_info;
    (void)status;
}


static void espnow_recv_cb(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len)
{
    (void)recv_info;

    if (data == NULL || len != sizeof(imu_msg_t)) {
        return;
    }

    const imu_msg_t *msg = (const imu_msg_t *)data;

    // 只处理本组设备
    if (msg->group_id != GROUP_ID) {
        return;
    }

    // 忽略自己发出去的广播包
    if (msg->node_id == NODE_ID) {
        return;
    }

    // 主节点只接收本组从节点 IMU 数据
    if (msg->msg_type == MSG_IMU) {
        if (s_imu_queue != NULL) {
            if (xQueueSend(s_imu_queue, msg, 0) != pdTRUE) {
                static uint32_t drop_count = 0;

                if ((drop_count++ % 100) == 0) {
                    ESP_LOGW(TAG, "IMU queue full, drop packet");
                }
            }
        }
    }
}


static void wifi_espnow_init(void)
{
    esp_err_t ret = nvs_flash_init();

    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    } else {
        ESP_ERROR_CHECK(ret);
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_sta();
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

    // APSTA：AP 给电脑连接，STA 用于 ESP-NOW
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_APSTA));

    wifi_config_t ap_config = {
        .ap = {
            .ssid = WIFI_AP_SSID,
            .ssid_len = 0,
            .channel = ESPNOW_CHANNEL,
            .password = WIFI_AP_PASS,
            .max_connection = WIFI_AP_MAX_CONN,
            .authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = {
                .required = false,
            },
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));

    ESP_ERROR_CHECK(esp_wifi_start());

    // 固定信道，必须和从节点 ESPNOW_CHANNEL 一致
    ESP_ERROR_CHECK(esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE));

    // 关闭省电
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    ESP_LOGI(TAG,
             "Wi-Fi AP started. SSID=%s PASS=%s CHANNEL=%d",
             WIFI_AP_SSID,
             WIFI_AP_PASS,
             ESPNOW_CHANNEL);

    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_send_cb(espnow_send_cb));
    ESP_ERROR_CHECK(esp_now_register_recv_cb(espnow_recv_cb));

    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, BROADCAST_MAC, 6);
    peer.channel = ESPNOW_CHANNEL;
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;

    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    uint8_t sta_mac[6];
    uint8_t ap_mac[6];

    ESP_ERROR_CHECK(esp_read_mac(sta_mac, ESP_MAC_WIFI_STA));
    ESP_ERROR_CHECK(esp_read_mac(ap_mac, ESP_MAC_WIFI_SOFTAP));

    ESP_LOGI(TAG,
             "MASTER GROUP_ID=%d NODE_ID=%d STA MAC=" MACSTR " AP MAC=" MACSTR,
             GROUP_ID,
             NODE_ID,
             MAC2STR(sta_mac),
             MAC2STR(ap_mac));
}


static void send_start_cmd(void)
{
    bool first_start = !g_session_started;

    int64_t now_us = esp_timer_get_time();

    imu_msg_t msg = {0};

    msg.group_id = GROUP_ID;
    msg.msg_type = MSG_START;
    msg.node_id = NODE_ID;
    msg.seq = g_start_seq++;
    msg.timestamp_us = now_us;

    esp_err_t ret = esp_now_send(BROADCAST_MAC, (uint8_t *)&msg, sizeof(msg));

    if (ret == ESP_OK) {
        if (first_start) {
            g_session_started = true;
            g_local_seq = 0;
            g_next_local_sample_us = now_us;

            ESP_LOGI(TAG,
                     "TX FIRST START group=%d master=%d start_seq=%lu, local seq reset",
                     msg.group_id,
                     msg.node_id,
                     (unsigned long)msg.seq);
        } else {
            ESP_LOGI(TAG,
                     "TX START heartbeat group=%d master=%d seq=%lu",
                     msg.group_id,
                     msg.node_id,
                     (unsigned long)msg.seq);
        }
    } else {
        ESP_LOGE(TAG, "TX START failed: %s", esp_err_to_name(ret));
    }
}


// ================= 主程序 =================

void app_main(void)
{
    ESP_LOGI(TAG, "Boot master group=%d node=%d", GROUP_ID, NODE_ID);

    s_imu_queue = xQueueCreate(IMU_QUEUE_LEN, sizeof(imu_msg_t));
    if (s_imu_queue == NULL) {
        ESP_LOGE(TAG, "Create IMU queue failed");
        return;
    }

    // LCD 和 IMU 都用 I2C，所以 I2C 只初始化一次
    bsp_i2c_init();

    // 初始化屏幕
    pca9557_init();
    bsp_lcd_init();
    bsp_display_brightness_set(30);

    // 显示：左边组号，右边节点号
    lcd_show_group_node_id(GROUP_ID, NODE_ID);

    // 初始化 IMU
    qmi8658_init();

    // 初始化 Wi-Fi AP + ESP-NOW
    wifi_espnow_init();

    // 初始化 UDP
    udp_broadcast_init();

    ESP_LOGI(TAG, "Master ready. First START will begin session.");

    int64_t last_start_us = esp_timer_get_time() - START_INTERVAL_MS * 1000LL;

    while (1) {
        int64_t now_us = esp_timer_get_time();

        // 处理从节点发来的 IMU 数据
        process_received_imu_queue();

        // 第一次 START 会开启采集；后续 START 作为心跳
        if (now_us - last_start_us >= START_INTERVAL_MS * 1000LL) {
            last_start_us = now_us;
            send_start_cmd();
        }

        // 主节点只有在第一次 START 成功后，才开始采集自己的 IMU
        if (g_session_started && now_us >= g_next_local_sample_us) {
            collect_local_imu_data();

            // 按固定周期推进下一次采样时间，减少 vTaskDelay 引起的漂移
            g_next_local_sample_us += IMU_SAMPLE_INTERVAL_MS * 1000LL;

            // 如果落后太多，不疯狂补发，追到当前时间附近
            if (now_us - g_next_local_sample_us > IMU_SAMPLE_INTERVAL_MS * 1000LL) {
                g_next_local_sample_us = now_us + IMU_SAMPLE_INTERVAL_MS * 1000LL;
            }
        }

        vTaskDelay(pdMS_TO_TICKS(1));
    }
}
