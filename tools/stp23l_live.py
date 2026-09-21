"""STP-23L free-motion live plot. Uses packet medians, no temporal filtering."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import math
from pathlib import Path
import time

from PyQt5 import QtWidgets

try:
    from .laser100m_live import LiveWindow
    from .stp23l_range_test import Stp23lParser, distance_candidates, serial, available_ports
except ImportError:
    from laser100m_live import LiveWindow
    from stp23l_range_test import Stp23lParser, distance_candidates, serial, available_ports


def capture(args, events, stop):
    path = None
    try:
        folder = Path(__file__).resolve().parents[1] / 'rangefinder_test_logs'
        folder.mkdir(exist_ok=True)
        path = folder / ('stp23l_live_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
        with path.open('w', newline='', encoding='utf-8-sig') as log:
            fields = ['wall_time','elapsed_s','module_timestamp','status','raw_mm','corrected_mm',
                      'offset_mm','display_limit_mm','valid_points','first_mm','bad_checksum','bad_length']
            fields += [f'p{i}_{name}' for i in range(1,13)
                       for name in ('distance_mm','noise','peak','confidence','integration','ref_tof')]
            writer = csv.DictWriter(log, fieldnames=fields)
            writer.writeheader()
            with serial.Serial(port=None, baudrate=230400, timeout=.05) as device:
                device.port = args.port
                device.dtr = False
                device.rts = False
                device.open()
                device.reset_input_buffer()
                events.put(('status','已连接STP，等待上电自动输出；未发送配置命令。'))
                parser = Stp23lParser()
                started = last_flush = time.monotonic()
                while not stop.is_set():
                    chunk = device.read(min(device.in_waiting or 1,4096))
                    now = time.monotonic()
                    for frame in parser.feed(chunk):
                        candidates = distance_candidates(frame,0,None)
                        raw = candidates['median_mm']
                        corrected = raw + args.offset_mm if raw is not None else None
                        status = ('device_error' if raw is None else
                                  'ok' if 0 <= corrected <= args.max_mm else 'outside_display_range')
                        row = dict(wall_time=datetime.now().isoformat(timespec='milliseconds'),
                                   elapsed_s=now-started,module_timestamp=frame.module_timestamp,
                                   status=status,raw_mm=raw,corrected_mm=corrected,
                                   offset_mm=args.offset_mm,display_limit_mm=args.max_mm,
                                   valid_points=candidates['valid_points'],first_mm=candidates['first_mm'],
                                   bad_checksum=parser.bad_checksum,bad_length=parser.bad_length)
                        for i,point in enumerate(frame.points,1):
                            row.update({f'p{i}_{key}': value for key,value in asdict(point).items()})
                        writer.writerow(row)
                        events.put(('frame',now-started,now,raw,corrected,status,
                                    '无非负距离样本' if raw is None else None,parser.bad_checksum))
                    if now-last_flush>=1:
                        log.flush()
                        last_flush=now
    except Exception as exc:
        events.put(('status',f'采集失败：{exc}；可用串口：{available_ports()}'))
    finally:
        events.put(('finished',str(path) if path else '未创建记录'))


def make_window(args):
    window = LiveWindow(args,capture_fn=capture,
                        config_text=f'STP-23L | {args.port} | 230400 baud | 包中位数 | 补偿 {args.offset_mm:+g} mm')
    window.setWindowTitle('STP-23L · 实时动态测距')
    window.note.setText(f'每包12点取非负距离的中位数，每收到一包更新一次；频率显示为包频率，不是标称120点/秒。\n'
                        f'默认补偿0，灰蓝线重合；无跨包平滑。最近{args.window_s:g}秒，显示0～{args.max_mm:g} mm。\n'
                        '异常/超范围及超过0.25秒的间断会断线；每包12点和质量字段完整保存在CSV。')
    return window


def main():
    parser=argparse.ArgumentParser(description='STP-23L自由移动实时曲线')
    parser.add_argument('--port',default='COM3')
    parser.add_argument('--offset-mm',type=float,default=0)
    parser.add_argument('--max-mm',type=float,default=500)
    parser.add_argument('--window-s',type=float,default=15)
    parser.add_argument('--list-ports',action='store_true')
    args=parser.parse_args()
    if args.list_ports:
        print(available_ports()); return 0
    if serial is None:
        parser.error('需要安装pyserial')
    if not all(math.isfinite(v) for v in (args.offset_mm,args.max_mm,args.window_s)) or min(args.max_mm,args.window_s)<=0:
        parser.error('参数必须为有限数字，显示上限和时长须大于0')
    app=QtWidgets.QApplication([])
    window=make_window(args)
    window.show()
    return app.exec_()


if __name__=='__main__':
    raise SystemExit(main())
