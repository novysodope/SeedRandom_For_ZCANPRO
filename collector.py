# -*- coding: utf-8 -*-
import time
import zcanpro
import os
from collections import defaultdict

stopTask = False
CAN_ID = 0

# ==================== 配置区 ====================
REQUEST_ID = 0x111          # 诊断请求CAN ID
RESPONSE_ID = 0x222         # 诊断响应CAN ID
SEED_LENGTH = 4             # 种子字节数
TARGET_COUNT = 5066120      # 目标采集次数
OUTPUT_FILE = r"seeds.bin"  # 存储路径
# =====================================================

last_seen = defaultdict(lambda: {'data': None, 'time': 0})

def z_notify(type, obj):
    zcanpro.write_log("Notify " + str(type) + " " + str(obj))
    if type == "stop":
        zcanpro.write_log("Stop...")
        global stopTask
        stopTask = True

def clear_rx_buffer(busID):
    """清空接收缓冲区，丢弃所有当前等待的帧"""
    while True:
        ok, frms = zcanpro.receive(busID)
        if not ok or not frms:
            break

def enter_extended_session(busID):
    """发送10 03进入扩展会话，等待50 03肯定响应"""
    clear_rx_buffer(busID)
    req_data = [0x02, 0x10, 0x03] + [0x00] * 5
    req = {"can_id": REQUEST_ID, "data": req_data, "is_canfd": 0, "canfd_brs": 0, "timestamp_us": 0}
    zcanpro.write_log("Sending 10 03...")
    if not zcanpro.transmit(busID, [req]):
        return False

    start = time.time()
    timeout = 2.0
    while time.time() - start < timeout:
        ok, frms = zcanpro.receive(busID)
        if ok and frms:
            for frm in frms:
                if frm["can_id"] == RESPONSE_ID and len(frm["data"]) >= 3:
                    if frm["data"][1] == 0x50 and frm["data"][2] == 0x03:
                        zcanpro.write_log("Enter extended session OK")
                        return True
        time.sleep(0.01)
    zcanpro.write_log("Enter extended session timeout")
    return False

def z_main():
    global stopTask
    buses = zcanpro.get_buses()
    zcanpro.write_log("Get buses: " + str(buses))
    if len(buses) < 1:
        return
    busID = buses[0]["busID"]

    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    try:
        f_out = open(OUTPUT_FILE, "ab")
        file_size = os.path.getsize(OUTPUT_FILE)
        seeds_collected = file_size // SEED_LENGTH
        zcanpro.write_log(f"Output file opened in append mode. Existing seeds: {seeds_collected}")
        zcanpro.write_log(f"Output file opened: {OUTPUT_FILE}")
    except Exception as e:
        zcanpro.write_log(f"Failed to open output file: {e}")
        return

    if not enter_extended_session(busID):
        zcanpro.write_log("Cannot enter extended session, exit.")
        f_out.close()
        return

    request_data = [0x02, 0x27, 0x01] + [0x00] * 5
    count = seeds_collected
    zcanpro.write_log(f"Start collecting seeds. Target: {TARGET_COUNT}, current: {count}")
    

    while not stopTask and count < TARGET_COUNT:
        # 可选：发送前清空缓冲区（注释掉以提升速度，若发现旧数据干扰可取消注释）
        # clear_rx_buffer(busID)

        # 发送27 01，并打印发送内容
        msg = {"can_id": REQUEST_ID, "data": request_data, "is_canfd": 0, "canfd_brs": 0, "timestamp_us": 0}
        # zcanpro.write_log(f"TX: ID={hex(REQUEST_ID)} Data={[hex(b) for b in request_data]}")
        if not zcanpro.transmit(busID, [msg]):
            zcanpro.write_log("Transmit error, retry...")
            continue

        # 等待响应（最多0.1秒）
        start_time = time.time()
        received = False
        while time.time() - start_time < 0.1:
            if stopTask:
                break
            ok, frms = zcanpro.receive(busID)
            if ok and frms:
                for frm in frms:
                    if frm["can_id"] != RESPONSE_ID:
                        continue
                    data = frm["data"]
                    # 打印原始响应
                    # zcanpro.write_log(f"RX: ID={hex(RESPONSE_ID)} Data={[hex(b) for b in data]}")

                    # 检查肯定响应67 01
                    if len(data) >= 3 and data[1] == 0x67 and data[2] == 0x01:
                        seed_start = 3
                        if len(data) >= seed_start + SEED_LENGTH:
                            seed = data[seed_start:seed_start + SEED_LENGTH]
                            f_out.write(bytes(seed))
                            f_out.flush()
                            count += 1
                            zcanpro.write_log(f"Seed collected: {[hex(b) for b in seed]} (Total: {count})")
                            if count % 1000 == 0:
                                zcanpro.write_log(f"Progress: {count}/{TARGET_COUNT}")
                            received = True
                            break
                        else:
                            zcanpro.write_log(f"Warning: response length {len(data)} < {seed_start + SEED_LENGTH}, cannot extract seed")
                    else:
                        # 非期望响应，可能是否定响应或其他，可以打印出来
                        zcanpro.write_log(f"Unexpected response: {[hex(b) for b in data]}")
            if received:
                break
            # 极短休眠，避免CPU空转
            time.sleep(0.0001)

        if not received:
            zcanpro.write_log("Warning: No response within 100ms")

    f_out.close()
    zcanpro.write_log(f"Collection finished. Total seeds collected: {count}")
