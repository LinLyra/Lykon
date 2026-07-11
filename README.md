# IMUcode

基于 **ESP32-S3 + QMI8658 六轴 IMU + ESP-NOW + Wi-Fi UDP** 的多节点同步采集项目。

本项目支持多组设备。每组包含 1 个主节点和 3 个从节点：

- 主节点通过 ESP-NOW 发送 START
- 从节点采集 IMU 并通过 ESP-NOW 回传
- 主节点采集自身 IMU
- 主节点通过 Wi-Fi UDP 将整组数据发送到电脑
- 电脑端 Python 程序接收并保存为 CSV

## 目录结构

```text
IMUcode/
├─ imu_espnow/        # 从节点 ESP-IDF 工程
├─ imu_master/        # 主节点 ESP-IDF 工程
├─ udp_receiver.py    # 电脑端 UDP 接收和 CSV 保存程序
├─ .gitignore
└─ README.md
```

## 硬件与软件

- MCU：ESP32-S3
- IMU：QMI8658
- LCD：ST7789，320 × 240
- 节点间通信：ESP-NOW
- 主节点到电脑：Wi-Fi UDP
- ESP-IDF：v6.0.2
- Target：esp32s3

## 节点与分组

程序通过 `GROUP_ID` 和 `NODE_ID` 区分设备。

第一组：

| 设备 | GROUP_ID | NODE_ID |
|---|---:|---:|
| 主节点 | 1 | 1 |
| 从节点 2 | 1 | 2 |
| 从节点 3 | 1 | 3 |
| 从节点 4 | 1 | 4 |

第二组仍使用节点编号 1、2、3、4，只需把 `GROUP_ID` 改为 2。

例如：

```c
#define GROUP_ID 2
#define NODE_ID  3
```

表示第二组的 3 号从节点。

## LCD 显示

LCD 使用黑色背景和白色七段数字显示：

```text
左侧：GROUP_ID
右侧：NODE_ID
```

例如 `1  4` 表示第一组的 4 号节点。

## QMI8658 六轴数据

| 字段 | 含义 |
|---|---|
| `acc_x` | X 轴加速度 |
| `acc_y` | Y 轴加速度 |
| `acc_z` | Z 轴加速度 |
| `gyr_x` | X 轴角速度 |
| `gyr_y` | Y 轴角速度 |
| `gyr_z` | Z 轴角速度 |

当前程序保存的是 16 位有符号原始值。

当前配置：

```c
qmi8658_register_write_byte(QMI8658_CTRL2, 0x95); // ACC ±4g
qmi8658_register_write_byte(QMI8658_CTRL3, 0xd5); // GYR ±512 dps
```

换算：

```text
acc_g   = acc_raw / 8192.0
gyr_dps = gyr_raw / 64.0
acc_mps2 = acc_raw / 8192.0 × 9.80665
```

## ESP-NOW 数据包

主节点和从节点必须使用完全相同的结构：

```c
typedef struct __attribute__((packed))
{
    uint8_t group_id;
    uint8_t msg_type;
    uint8_t node_id;
    uint32_t seq;
    int64_t timestamp_us;

    int16_t acc_x;
    int16_t acc_y;
    int16_t acc_z;
    int16_t gyr_x;
    int16_t gyr_y;
    int16_t gyr_z;
} imu_msg_t;
```

消息类型：

```c
#define MSG_START 1
#define MSG_IMU   2
```

## 同步采集流程

推荐启动顺序：

1. 先启动所有从节点。
2. 确认从节点进入 `Slave waiting for START command...`。
3. 最后启动或复位主节点。
4. 主节点发送第一次 START。
5. 主节点和从节点将 `seq` 清零。
6. 所有节点按固定周期采集。

第一次 START 用于开始新一轮采集，后续 START 作为心跳，不再清零。

注意：每块 ESP32 的 `esp_timer_get_time()` 都是本机开机时间，不能直接作为跨节点统一时钟。跨节点对齐优先使用：

```text
group_id + node_id + seq
```

## 从节点工程

目录：

```text
imu_espnow/
```

烧录前修改：

```c
#define GROUP_ID 1
#define NODE_ID  2
```

不同从节点分别设置为 2、3、4。

采样发送周期：

```c
#define IMU_SEND_INTERVAL_MS 20
```

对应关系：

```text
10 ms  = 100 Hz
20 ms  = 50 Hz
100 ms = 10 Hz
```

从节点使用发送 busy 标志，等待上一包发送完成后再发送下一包，以减少：

```text
ESP_ERR_ESPNOW_NO_MEM
```

## 主节点工程

目录：

```text
imu_master/
```

第一组主节点：

```c
#define GROUP_ID 1
#define NODE_ID  1
```

主要参数：

```c
#define START_INTERVAL_MS       1000
#define IMU_SAMPLE_INTERVAL_MS  20
#define UDP_PORT                3333
```

默认热点：

```text
SSID: IMU_MASTER_G1
Password: 12345678
```

SSID 会随 `GROUP_ID` 变化：

```text
GROUP_ID=1 → IMU_MASTER_G1
GROUP_ID=2 → IMU_MASTER_G2
```

电脑连接后提示“无 Internet”属于正常现象。

## ESP-NOW 信道

主节点和所有从节点必须一致：

```c
#define ESPNOW_CHANNEL 1
```

信道不同会导致从节点收不到 START 或主节点收不到 IMU 数据。

## 电脑端 UDP 接收

脚本：

```text
udp_receiver.py
```

默认配置：

```python
SAVE_DIR = r"D:\esp\imu_data"
UDP_PORT = 3333
```

运行：

```powershell
python D:\esp\udp_receiver.py
```

程序会创建：

```text
D:\esp\imu_data\imu_data_YYYYMMDD_HHMMSS.csv
```

## CSV 格式

```text
group,node,seq,node_timestamp_us,master_timestamp_us,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z
```

字段：

| 字段 | 含义 |
|---|---|
| `group` | 设备组编号 |
| `node` | 节点编号 |
| `seq` | 采样序号 |
| `node_timestamp_us` | 节点本地采样时间 |
| `master_timestamp_us` | 主节点处理该数据的时间 |
| `acc_x/y/z` | 三轴加速度原始值 |
| `gyr_x/y/z` | 三轴角速度原始值 |

## 编译与烧录

进入工程：

```powershell
cd D:\esp\imu_espnow
```

或：

```powershell
cd D:\esp\imu_master
```

设置目标：

```powershell
idf.py set-target esp32s3
```

编译：

```powershell
idf.py build
```

查看串口：

```powershell
python -m serial.tools.list_ports
```

烧录并监视：

```powershell
idf.py -p COM5 flash monitor
```

退出 monitor：

```text
Ctrl + ]
```

## 删除编译缓存并重新烧录

修改节点编号后仍显示旧编号时：

```powershell
if (Test-Path .\build) {
    Remove-Item -Recurse -Force .\build
}
```

可选：擦除 Flash：

```powershell
idf.py -p COM5 erase-flash
```

重新烧录：

```powershell
idf.py -p COM5 flash monitor
```

烧录 4 号节点后应看到：

```text
SLAVE GROUP_ID=1 NODE_ID=4
TX IMU group=1 node=4 seq=0
```

## Windows 防火墙

电脑已连接热点但收不到 UDP 时，可在管理员 PowerShell 中执行：

```powershell
New-NetFirewallRule `
  -DisplayName "IMU UDP 3333" `
  -Direction Inbound `
  -Protocol UDP `
  -LocalPort 3333 `
  -Action Allow
```

## 常见问题

### ESP_ERR_ESPNOW_NO_MEM

表示 ESP-NOW 发送队列暂时没有可用空间。

建议：

- 等待发送完成回调后再发下一包
- 使用发送 busy 标志
- 将采样周期先调整为 20 ms
- 减少串口打印频率

### 主节点收不到某个节点

检查：

- `GROUP_ID` 是否一致
- `NODE_ID` 是否烧录正确
- `ESPNOW_CHANNEL` 是否一致
- 从节点是否收到 `RX FIRST START`
- 从节点日志中的节点编号是否正确
- 主从节点 `imu_msg_t` 是否完全一致

### LCD 花屏

使用：

```c
lcd_fill_rect(0, 0, 320, 240, 0x0000);
```

清屏，避免使用依赖大块 PSRAM 缓冲区的整屏清屏函数。

### I2C legacy driver 警告

ESP-IDF v6 会提示 `driver/i2c.h` 已进入 EOL。当前代码仍可运行，但升级到 ESP-IDF v7 前需要迁移到新版 I2C Master API。

## 多组设备

独立热点方式：

```text
第一组：IMU_MASTER_G1
第二组：IMU_MASTER_G2
```

电脑通常一次只能连接一个热点。

若需要多组同时向同一台电脑发送，建议让多个主节点接入同一个路由器或同一个统一热点，再向电脑固定 IP 发送 UDP，并通过 `group` 字段区分数据。

## 正式采集建议

1. 确认各节点编号正确。
2. 先启动全部从节点。
3. 最后启动主节点。
4. 先以 50 Hz 测试稳定性。
5. 检查各节点 `seq` 是否连续。
6. 确认没有大量 `ESP_ERR_ESPNOW_NO_MEM`。
7. 再逐步提高到 100 Hz。

## License

请根据项目实际用途补充许可证信息。
