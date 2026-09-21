"""Read the 100 m module using repeated single measurements (9600 8N1)."""
import argparse
import csv
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import re
import time

import serial
from serial.tools import list_ports

ERRORS = {10: "电量低", 14: "计算错误", 15: "超出量程",
          16: "信号弱或测量时间过长", 18: "环境光太强", 26: "超出显示范围"}


def parse_frame(buffer, address=0x80):
    """Consume one valid reply, retaining incomplete bytes for the next read."""
    header = bytes((address, 6, 0x82))
    while len(buffer) >= 3:
        if buffer[:3] != header:
            del buffer[0]
            continue
        for size in (11, 12):
            if len(buffer) < size:
                continue
            frame = bytes(buffer[:size])
            payload = frame[3:-1]
            valid = (re.fullmatch(rb"\d{3}\.\d{3,4}", payload)
                     or re.fullmatch(rb"ERR-{1,3}\d{2}", payload))
            if valid and sum(frame) % 256 == 0:
                del buffer[:size]
                value = payload.decode("ascii")
                if value.startswith("ERR"):
                    return "device_error", "", value + " " + ERRORS.get(int(value[-2:]), "未知错误码")
                return "ok", str(Decimal(value) * 1000), ""
        if len(buffer) < 12:
            return None
        del buffer[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="100米模块实时读取，Ctrl+C停止")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--address", type=lambda s: int(s, 0), default=0x80)
    parser.add_argument("--timeout", type=float, default=4.0, help="每次响应超时秒数")
    parser.add_argument("--interval", type=float, default=0.1, help="每次完成后的间隔秒数")
    parser.add_argument("--duration", type=float, default=0, help="运行秒数，0表示持续运行")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output" / "sensor_checks")
    args = parser.parse_args()
    if args.list_ports:
        for port in list_ports.comports():
            print(port.device, port.description, port.hwid)
        return 0
    if not 0 <= args.address <= 255 or args.timeout <= 0 or args.interval < 0 or args.duration < 0:
        parser.error("地址需在0..255；timeout>0，interval和duration>=0")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    path = args.log_dir / ("laser100m_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".csv")
    request = bytes((args.address, 6, 2))
    request += bytes(((-sum(request)) & 255,))
    counts = {"ok": 0, "device_error": 0, "timeout": 0, "invalid_response": 0}
    started = time.monotonic()
    stop_at = started + args.duration if args.duration else float("inf")
    print(f"{args.port} | 9600 8N1 | 重复单次测量 | Ctrl+C停止\n记录：{path}", flush=True)
    exit_code = 0
    try:
        with path.open("w", newline="", encoding="utf-8-sig") as log:
            writer = csv.writer(log)
            writer.writerow(("timestamp", "elapsed_s", "status", "distance_mm", "message", "response_wait_ms", "raw_hex"))
            with serial.Serial(port=None, baudrate=9600, timeout=0.05, write_timeout=1) as device:
                device.port = args.port
                device.dtr = False
                device.rts = False
                device.open()
                while time.monotonic() < stop_at:
                    device.reset_input_buffer()
                    sent_at = time.monotonic()
                    device.write(request)
                    device.flush()
                    deadline = min(sent_at + args.timeout, stop_at)
                    buffer, raw = bytearray(), bytearray()
                    result = None
                    while time.monotonic() < deadline:
                        chunk = device.read(min(device.in_waiting or 1, 4096))
                        buffer.extend(chunk)
                        raw.extend(chunk)
                        result = parse_frame(buffer, args.address)
                        if result is not None:
                            break
                    if result is None:
                        status = "invalid_response" if raw else "timeout"
                        message = "收到字节但无完整有效测距帧" if raw else "未收到响应"
                        if time.monotonic() >= stop_at:
                            message += "（达到运行时限）"
                        result = status, "", message
                    status, distance, message = result
                    counts[status] += 1
                    now = datetime.now().isoformat(timespec="milliseconds")
                    wait_ms = (time.monotonic() - sent_at) * 1000
                    writer.writerow((now, f"{time.monotonic()-started:.3f}", status, distance, message, f"{wait_ms:.1f}", raw.hex(" ")))
                    log.flush()
                    display = f"距离 {distance} mm" if status == "ok" else f"[{status}] {message}"
                    print(f"{now}  {display}  等待 {wait_ms:.0f} ms", flush=True)
                    time.sleep(max(0, min(args.interval, stop_at - time.monotonic())))
    except KeyboardInterrupt:
        print("\n已停止。", flush=True)
    except (serial.SerialException, OSError) as exc:
        print(f"读取失败：{exc}。检查连接及串口是否被其他软件占用。", flush=True)
        exit_code = 1
    print(f"统计：{counts}；串口已释放。记录：{path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
