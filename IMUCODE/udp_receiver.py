import socket
import os
from datetime import datetime

# ====== 固定保存文件夹 ======
SAVE_DIR = r"D:\esp\imu_data"

# ====== UDP 端口，要和 ESP32 主节点一致 ======
UDP_PORT = 3333

# 自动创建文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

# 每次运行生成一个新的 CSV 文件
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(SAVE_DIR, f"imu_data_{time_str}.csv")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 允许端口复用，避免脚本重启时报端口占用
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

sock.bind(("", UDP_PORT))

print(f"Waiting for IMU UDP data on port {UDP_PORT}...")
print(f"Saving to: {csv_path}")

count = 0

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    # 新版表头：增加 master_timestamp_us
    f.write("group,node,seq,node_timestamp_us,master_timestamp_us,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z\n")
    f.flush()

    while True:
        data, addr = sock.recvfrom(1024)

        line = data.decode(errors="ignore").strip()

        if not line:
            continue

        parts = line.split(",")

        # 新版主节点应该发 11 列
        if len(parts) != 11:
            print(f"Bad line from {addr}: {line}")
            continue

        f.write(line + "\n")
        count += 1

        # 每 100 行刷新一次，减少频繁写硬盘造成的卡顿
        if count % 100 == 0:
            f.flush()
            print(f"Received {count} lines, latest: {line}")
