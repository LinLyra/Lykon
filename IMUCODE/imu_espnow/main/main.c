#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "esp_mac.h"

#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "esp_now.h"

#include "esp32_s3_szp.h"


// ====== 每块从动节点烧录前改这里 ======
// 第一套设备：GROUP_ID = 1，NODE_ID = 2/3/4
// 第二套设备：GROUP_ID = 2，NODE_ID = 2/3/4
#define GROUP_ID 1
#define NODE_ID  4

#if NODE_ID == 1
#error "Slave project must not use NODE_ID 1"
#endif

#define ESPNOW_CHANNEL 1

#define IMU_SEND_INTERVAL_MS 20     // 20ms = 50Hz，先稳定测试；10ms = 100Hz

#define MSG_START 1
#define MSG_IMU   2

static const char *TAG = "IMU_SLAVE";

static const uint8_t BROADCAST_MAC[6] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

typedef struct __attribute__((packed))
{
    uint8_t group_id;       // 第几套设备
    uint8_t msg_type;       // MSG_START 或 MSG_IMU
    uint8_t node_id;        // 本组内节点编号
    uint32_t seq;           // 数据序号
    int64_t timestamp_us;   // 本节点本地时间戳，单位 us

    int16_t acc_x;
    int16_t acc_y;
    int16_t acc_z;
    int16_t gyr_x;
    int16_t gyr_y;
    int16_t gyr_z;
} imu_msg_t;

static volatile bool g_collecting = false;
static volatile bool g_espnow_send_busy = false;

static uint32_t g_seq = 0;
static int64_t g_start_time_us = 0;
static int64_t g_next_sample_us = 0;

static t_sQMI8658 QMI8658;


// ================= LCD 显示 =================

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


// ================= ESP-NOW =================

static void espnow_send_cb(const esp_now_send_info_t *tx_info, esp_now_send_status_t status)
{
    (void)tx_info;
    (void)status;

    g_espnow_send_busy = false;
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

    // 从节点只响应主节点 START
    if (msg->msg_type == MSG_START && msg->node_id == 1) {

        // 第一次收到 START：真正开始采集，并清零 seq
        if (!g_collecting) {
            g_collecting = true;
            g_seq = 0;
            g_start_time_us = esp_timer_get_time();
            g_next_sample_us = g_start_time_us;

            ESP_LOGI(TAG,
                     "RX FIRST START group=%d master=%d master_seq=%lu, local seq reset",
                     msg->group_id,
                     msg->node_id,
                     (unsigned long)msg->seq);
        } else {
            // 后续 START 只当作主节点还在线，不再重置 seq
            ESP_LOGI(TAG,
                     "RX START heartbeat group=%d master=%d master_seq=%lu",
                     msg->group_id,
                     msg->node_id,
                     (unsigned long)msg->seq);
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

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE));

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_send_cb(espnow_send_cb));
    ESP_ERROR_CHECK(esp_now_register_recv_cb(espnow_recv_cb));

    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, BROADCAST_MAC, 6);
    peer.channel = ESPNOW_CHANNEL;
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;

    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    uint8_t mac[6];
    ESP_ERROR_CHECK(esp_read_mac(mac, ESP_MAC_WIFI_STA));

    ESP_LOGI(TAG,
             "SLAVE GROUP_ID=%d NODE_ID=%d WiFi STA MAC=" MACSTR,
             GROUP_ID,
             NODE_ID,
             MAC2STR(mac));
}


// ================= IMU 发送 =================

static void send_imu_data(void)
{
    if (g_espnow_send_busy) {
        return;
    }

    if (!qmi8658_Read_AccAndGry(&QMI8658)) {
        static uint32_t fail_count = 0;

        if ((fail_count++ % 50) == 0) {
            ESP_LOGW(TAG, "Skip invalid IMU sample");
        }

        return;
    }

    imu_msg_t msg = {0};

    msg.group_id = GROUP_ID;
    msg.msg_type = MSG_IMU;
    msg.node_id = NODE_ID;
    msg.seq = g_seq;
    msg.timestamp_us = esp_timer_get_time();

    msg.acc_x = QMI8658.acc_x;
    msg.acc_y = QMI8658.acc_y;
    msg.acc_z = QMI8658.acc_z;
    msg.gyr_x = QMI8658.gyr_x;
    msg.gyr_y = QMI8658.gyr_y;
    msg.gyr_z = QMI8658.gyr_z;

    g_espnow_send_busy = true;

    esp_err_t ret = esp_now_send(BROADCAST_MAC, (uint8_t *)&msg, sizeof(msg));

    if (ret == ESP_OK) {
        g_seq++;

        if (msg.seq % 50 == 0) {
            ESP_LOGI(TAG,
                     "TX IMU group=%d node=%d seq=%lu ts=%lld acc=[%d,%d,%d] gyr=[%d,%d,%d]",
                     msg.group_id,
                     msg.node_id,
                     (unsigned long)msg.seq,
                     (long long)msg.timestamp_us,
                     msg.acc_x,
                     msg.acc_y,
                     msg.acc_z,
                     msg.gyr_x,
                     msg.gyr_y,
                     msg.gyr_z);
        }
    } else {
        g_espnow_send_busy = false;

        static uint32_t fail_count = 0;
        if ((fail_count++ % 50) == 0) {
            ESP_LOGE(TAG, "TX IMU failed: %s", esp_err_to_name(ret));
        }
    }
}


// ================= 主程序 =================

void app_main(void)
{
    ESP_LOGI(TAG, "Boot slave group=%d node=%d", GROUP_ID, NODE_ID);

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

    // 初始化 ESP-NOW
    wifi_espnow_init();

    ESP_LOGI(TAG, "Slave waiting for START command...");

    while (1) {
        if (g_collecting) {
            int64_t now_us = esp_timer_get_time();

            if (now_us >= g_next_sample_us) {
                send_imu_data();

                // 按固定周期推进下一次采样时间，减少 vTaskDelay 造成的漂移
                g_next_sample_us += IMU_SEND_INTERVAL_MS * 1000LL;

                // 如果因为某次阻塞落后太多，直接追到当前时间附近，避免疯狂补发
                if (now_us - g_next_sample_us > IMU_SEND_INTERVAL_MS * 1000LL) {
                    g_next_sample_us = now_us + IMU_SEND_INTERVAL_MS * 1000LL;
                }
            }

            vTaskDelay(pdMS_TO_TICKS(1));
        } else {
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }
}
