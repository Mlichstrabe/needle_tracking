"""Live distance display: python tools/laser100m_live.py --port COM3."""
import argparse
from collections import deque
import csv
import json
from datetime import datetime
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

try:
    from .laser100m_range_test import applied_offset_mm, corrected_distance, Laser100mParser, command, serial
except ImportError:
    from laser100m_range_test import applied_offset_mm, corrected_distance, Laser100mParser, command, serial


FREQUENCY_CODES = {5: 0x05, 10: 0x0A, 20: 0x14}
CONFIG_ACK_TIMEOUT_S = 0.35
FIRST_FRAME_TIMEOUT_S = 2.0
FIRST_FRAME_RETRY_AFTER_S = 0.8


def configure_output(device, stop, frequency_hz=10, timeout=CONFIG_ACK_TIMEOUT_S):
    """Set the documented output interval to zero, then set output frequency.

    This does not shorten the sensor's internal measurement/recovery time.
    """
    results = {}
    frequency_value = FREQUENCY_CODES[frequency_hz]
    for name, request, expected in (
        ('interval_zero', command(0xFA, 4, 5, 0), bytes.fromhex('FA 04 85 7D')),
        (f'frequency_{frequency_hz}hz', command(0xFA, 4, 0x0A, frequency_value),
         bytes.fromhex('FA 04 8A 78')),
    ):
        if stop.is_set():
            break
        device.reset_input_buffer()
        device.write(request)
        device.flush()
        received = bytearray()
        deadline = time.monotonic() + timeout
        confirmed = False
        while not stop.is_set() and time.monotonic() < deadline:
            received.extend(device.read(min(device.in_waiting or 1, 4096)))
            if expected in received:
                confirmed = True
                break
            if len(received) > 4096:
                del received[:-16]
        results[name] = {'confirmed': confirmed, 'request_hex': request.hex(' '),
                         'response_tail_hex': received[-256:].hex(' ')}
    return results


def capture(args, events, stop):
    """Serial and disk IO stay off the GUI thread. Every reply is logged."""
    path = None
    finish_reason = ""
    try:
        folder = Path(getattr(
            args,
            'log_dir',
            Path(__file__).resolve().parents[1] / 'rangefinder_test_logs',
        ))
        folder.mkdir(exist_ok=True)
        path = folder / ('laser100m_live_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
        with path.open('w', newline='', encoding='utf-8-sig') as log:
            writer = csv.writer(log)
            writer.writerow(['wall_time', 'elapsed_s', 'status', 'raw_mm', 'corrected_mm',
                             'offset_mm', 'display_limit_mm', 'error_code', 'raw_hex', 'reply_gap_s'])
            with serial.Serial(port=None, baudrate=9600, timeout=.05, write_timeout=1) as device:
                device.port = args.port
                device.dtr = False
                device.rts = False
                device.open()
                events.put(('connected', f'串口已打开：{args.port}，正在启动测距'))
                events.put(('status', f'串口已连接，设置输出间隔0和{args.frequency} Hz；每项最多等待{CONFIG_ACK_TIMEOUT_S:g}秒…'))
                # 配置应答不是开始采集的必要条件。设备没有回 ACK 时快速
                # 放弃等待，继续请求测量，避免连接阶段固定阻塞约 16 秒。
                settings = configure_output(device, stop, args.frequency,
                                             timeout=CONFIG_ACK_TIMEOUT_S)
                with path.with_suffix('.settings.json').open('w', encoding='utf-8') as settings_file:
                    json.dump({'port': args.port, 'baud': 9600,
                               'near_offset_max_mm': args.offset_mm,
                               'offset_full_until_mm': args.offset_full_until_mm,
                               'offset_zero_from_mm': args.offset_zero_from_mm,
                               'correction_formula': ('raw<=full_until: +max_offset; '
                                                      'full_until<raw<zero_from: linear taper; '
                                                      'raw>=zero_from: +0'),
                               'display_limit_mm': args.max_mm, 'frequency_hz': args.frequency,
                               'configuration': settings},
                              settings_file, ensure_ascii=False, indent=2)
                if stop.is_set():
                    return
                interval_text = '输出间隔0已确认' if settings['interval_zero']['confirmed'] else '输出间隔0未确认'
                frequency_key = f'frequency_{args.frequency}hz'
                frequency_text = (f'{args.frequency} Hz已确认' if settings[frequency_key]['confirmed']
                                  else f'{args.frequency} Hz未确认')
                events.put(('config_text',
                            f'{args.port} | {interval_text} | {frequency_text} | '
                            f'≤{args.offset_full_until_mm:g} mm补偿 {args.offset_mm:+g} mm，'
                            f'至{args.offset_zero_from_mm:g} mm归零'))
                continuous_command = command(0x80, 6, 3)
                device.reset_input_buffer()
                device.write(continuous_command)
                device.flush()
                events.put(('status',
                            f'已请求连续测量，等待首帧（最多 {FIRST_FRAME_TIMEOUT_S:g} 秒）…'))
                parser = Laser100mParser()
                started = time.monotonic()
                last_flush = started
                last_reply = None
                first_frame_received = False
                continuous_attempts = 1
                first_frame_deadline = started + FIRST_FRAME_TIMEOUT_S
                retry_at = started + FIRST_FRAME_RETRY_AFTER_S
                rx_bytes = 0
                while not stop.is_set():
                    chunk = device.read(min(device.in_waiting or 1, 4096))
                    rx_bytes += len(chunk)
                    now = time.monotonic()
                    frames = parser.feed(chunk)
                    if frames:
                        first_frame_received = True
                    for frame in frames:
                        raw = frame.distance_mm
                        offset = applied_offset_mm(raw, args.offset_mm,
                                                   args.offset_full_until_mm,
                                                   args.offset_zero_from_mm)
                        corrected = corrected_distance(raw, args.offset_mm,
                                                       args.offset_full_until_mm,
                                                       args.offset_zero_from_mm)
                        status = frame.status
                        if raw is not None and not 0 <= corrected <= args.max_mm:
                            status = 'outside_display_range'
                        writer.writerow([datetime.now().isoformat(timespec='milliseconds'),
                                         f'{now-started:.6f}', status, raw, corrected,
                                         offset, args.max_mm, frame.error_code, frame.raw_hex,
                                         now-last_reply if last_reply is not None else None])
                        last_reply = now
                        events.put(('frame', now-started, now, raw, corrected, status,
                                    frame.error_code, parser.bad_checksum))
                    if not first_frame_received:
                        if continuous_attempts == 1 and now >= retry_at:
                            device.reset_input_buffer()
                            device.write(continuous_command)
                            device.flush()
                            continuous_attempts = 2
                            events.put(('status', '首帧未到，已快速重发连续测量命令'))
                        if now >= first_frame_deadline:
                            finish_reason = (
                                f'首帧超时 {FIRST_FRAME_TIMEOUT_S:g}s；'
                                f'尝试 {continuous_attempts} 次，收到 {rx_bytes} 字节，'
                                f'校验失败 {parser.bad_checksum}，解析失败 {parser.bad_payload}'
                            )
                            events.put(('status', finish_reason))
                            break
                    if now-last_flush >= 1:
                        log.flush()
                        last_flush = now
    except Exception as exc:
        events.put(('status', f'采集失败：{exc}'))
        finish_reason = f'采集失败：{exc}'
    finally:
        events.put(('finished', str(path) if path else '未创建记录', finish_reason))


class LiveWindow(QtWidgets.QWidget):
    def __init__(self, args, capture_fn=capture, config_text=None):
        super().__init__()
        self.args = args
        self.capture_fn = capture_fn
        requested_frequency = getattr(args, 'frequency', 10)
        self.config_text = config_text or (
            f'{args.port} | 请求{requested_frequency} Hz | '
            f'≤{args.offset_full_until_mm:g} mm补偿 {args.offset_mm:+g} mm，'
            f'至{args.offset_zero_from_mm:g} mm归零'
        )
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.close_pending = False
        self.last_rx = None
        self.latest_raw = None
        self.latest_corrected = None
        self.latest_offset = None
        self.latest_elapsed = None
        self.latest_received = None
        self.comparison_path = None
        self.comparison_count = 0
        self.times = deque(maxlen=10000)
        self.raw_values = deque(maxlen=10000)
        self.corrected_values = deque(maxlen=10000)
        self.arrivals = deque()
        self.total = self.outside = self.errors = self.long_gaps = 0
        self.max_gap = 0.0
        self.setWindowTitle('100米模块 · 实时距离')
        self.resize(1000, 680)
        layout = QtWidgets.QVBoxLayout(self)
        self.config = QtWidgets.QLabel(self.config_text)
        layout.addWidget(self.config)
        self.value = QtWidgets.QLabel('— mm')
        self.value.setAlignment(QtCore.Qt.AlignCenter)
        self.value.setMinimumHeight(150)
        self.value.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.value, 2)
        self.raw_label = QtWidgets.QLabel('原始距离：—')
        layout.addWidget(self.raw_label)
        comparison_row = QtWidgets.QHBoxLayout()
        comparison_row.addWidget(QtWidgets.QLabel('真实距离：'))
        self.actual_distance = QtWidgets.QDoubleSpinBox()
        self.actual_distance.setRange(0.1, 2000.0)
        self.actual_distance.setDecimals(1)
        self.actual_distance.setSingleStep(1.0)
        self.actual_distance.setValue(100.0)
        self.actual_distance.setSuffix(' mm')
        self.actual_distance.setKeyboardTracking(False)
        comparison_row.addWidget(self.actual_distance)
        self.record_comparison_button = QtWidgets.QPushButton('记录当前对照点')
        self.record_comparison_button.setEnabled(False)
        self.record_comparison_button.clicked.connect(self.record_comparison)
        comparison_row.addWidget(self.record_comparison_button)
        comparison_row.addStretch(1)
        layout.addLayout(comparison_row)
        self.comparison_result = QtWidgets.QLabel('输入尺具测得的真实距离，收到有效数据后可记录误差。')
        self.comparison_result.setWordWrap(True)
        self.comparison_result.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.comparison_result)
        self.status = QtWidgets.QLabel('固定模块，将平整靶面前后移动；点击“开始采集”。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.metrics = QtWidgets.QLabel('接收频率：—')
        layout.addWidget(self.metrics)
        self.plot = pg.PlotWidget()
        self.plot.setBackground('w')
        self.plot.showGrid(x=True, y=True, alpha=.2)
        self.plot.setLabel('left', 'Distance', units='mm')
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setYRange(0, args.max_mm, padding=0)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.addLegend()
        self.raw_curve = self.plot.plot(pen=pg.mkPen('#929baa', width=1), name='Raw', connect='finite')
        self.corrected_curve = self.plot.plot(pen=pg.mkPen('#1673b8', width=2), name='Corrected', connect='finite')
        layout.addWidget(self.plot, 1)
        note = self.note = QtWidgets.QLabel(f'灰线=原始值，蓝线=补偿值；最近{args.window_s:g}秒。显示范围0～{args.max_mm:g} mm。\n'
                               '超出显示范围、错误回复及超过0.25秒的接收间断会断线；全部回复仍保存。无平滑滤波。')
        layout.addWidget(note)
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton('开始采集')
        self.stop_button = QtWidgets.QPushButton('停止并保存')
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)
        self.log_label = QtWidgets.QLabel('每次开始采集会创建新的CSV。')
        self.log_label.setWordWrap(True)
        self.log_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.log_label)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)
        QtCore.QTimer.singleShot(0, self._fit_value_font)

    def _set_value_text(self, text):
        self.value.setText(text)
        self._fit_value_font()

    def _fit_value_font(self):
        """Use the largest bold font that fits the current distance area."""
        area = self.value.contentsRect()
        if area.width() < 20 or area.height() < 20:
            return
        text = self.value.text() or '—'
        low, high = 20, min(320, max(20, area.height()))
        best = low
        while low <= high:
            size = (low + high) // 2
            font = QtGui.QFont(self.value.font())
            font.setBold(True)
            font.setPixelSize(size)
            bounds = QtGui.QFontMetrics(font).boundingRect(text)
            if bounds.width() <= area.width() * .94 and bounds.height() <= area.height() * .90:
                best = size
                low = size + 1
            else:
                high = size - 1
        font = QtGui.QFont(self.value.font())
        font.setBold(True)
        font.setPixelSize(best)
        self.value.setFont(font)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._fit_value_font)

    def record_comparison(self):
        if self.latest_corrected is None or self.latest_received is None:
            self.comparison_result.setText('没有可记录的有效测量值。')
            return
        age = time.monotonic() - self.latest_received
        if age > .5:
            self.comparison_result.setText(f'最新有效数据已有 {age:.2f} 秒，等待新数据后再记录。')
            self.record_comparison_button.setEnabled(False)
            return
        actual = self.actual_distance.value()
        signed_error = self.latest_corrected - actual
        absolute_error = abs(signed_error)
        relative_error = absolute_error / actual * 100
        if self.comparison_path is None:
            folder = Path(__file__).resolve().parents[1] / 'rangefinder_test_logs'
            folder.mkdir(exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.comparison_path = folder / f'distance_comparison_{stamp}.csv'
        fields = ['wall_time', 'actual_mm', 'raw_mm', 'measured_mm', 'offset_mm',
                  'signed_error_mm', 'absolute_error_mm', 'relative_error_percent',
                  'measurement_elapsed_s', 'data_age_s']
        new_file = not self.comparison_path.exists()
        with self.comparison_path.open('a', newline='', encoding='utf-8-sig') as log:
            writer = csv.DictWriter(log, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow({
                'wall_time': datetime.now().isoformat(timespec='milliseconds'),
                'actual_mm': f'{actual:.3f}',
                'raw_mm': f'{self.latest_raw:.3f}',
                'measured_mm': f'{self.latest_corrected:.3f}',
                'offset_mm': f'{self.latest_offset:.3f}',
                'signed_error_mm': f'{signed_error:.3f}',
                'absolute_error_mm': f'{absolute_error:.3f}',
                'relative_error_percent': f'{relative_error:.4f}',
                'measurement_elapsed_s': f'{self.latest_elapsed:.6f}',
                'data_age_s': f'{age:.6f}',
            })
        self.comparison_count += 1
        self.comparison_result.setText(
            f'已记录第 {self.comparison_count} 点｜真实 {actual:.1f} mm｜测量 {self.latest_corrected:.1f} mm｜'
            f'误差 {signed_error:+.1f} mm｜绝对误差 {absolute_error:.1f} mm｜相对误差 {relative_error:.2f}%\n'
            f'对照记录：{self.comparison_path}'
        )

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        for data in (self.times, self.raw_values, self.corrected_values, self.arrivals):
            data.clear()
        self.total = self.outside = self.errors = self.long_gaps = 0
        self.max_gap = 0.0
        self.last_rx = None
        self.latest_raw = None
        self.latest_corrected = None
        self.latest_offset = None
        self.latest_elapsed = None
        self.latest_received = None
        self.comparison_path = None
        self.comparison_count = 0
        self.record_comparison_button.setEnabled(False)
        self.comparison_result.setText('输入尺具测得的真实距离，收到有效数据后可记录误差。')
        self._set_value_text('— mm')
        self.raw_label.setText('原始距离：—')
        self.raw_curve.clear()
        self.corrected_curve.clear()
        self.config.setText(self.config_text)
        self.stop_event.clear()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker = threading.Thread(target=self.capture_fn, args=(self.args, self.events, self.stop_event), daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.stop_button.setEnabled(False)
        self.status.setText('正在停止并保存…')

    def refresh(self):
        while not self.events.empty():
            event = self.events.get_nowait()
            if event[0] == 'status':
                self.status.setText(event[1])
            elif event[0] == 'config_text':
                self.config.setText(event[1])
            elif event[0] == 'config':
                requested_frequency = getattr(self.args, 'frequency', 10)
                result = f'{requested_frequency} Hz设置已确认' if event[1] else f'{requested_frequency} Hz设置未确认'
                self.config.setText(
                    f'{self.args.port} | {result} | '
                    f'≤{self.args.offset_full_until_mm:g} mm补偿 {self.args.offset_mm:+g} mm，'
                    f'至{self.args.offset_zero_from_mm:g} mm归零'
                )
            elif event[0] == 'finished':
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.record_comparison_button.setEnabled(False)
                reason = event[2] if len(event) > 2 and event[2] else ''
                self.log_label.setText('记录：' + event[1] + (f'；{reason}' if reason else ''))
                self._set_value_text('已停止')
                if self.close_pending:
                    QtCore.QTimer.singleShot(100, self.close)
            elif event[0] == 'frame':
                _, elapsed, received, raw, corrected, status, code, bad = event
                gap = received-self.last_rx if self.last_rx is not None else 0.0
                if gap > .25:
                    self.times.append(elapsed-.000001)
                    self.raw_values.append(np.nan)
                    self.corrected_values.append(np.nan)
                    self.long_gaps += 1
                self.max_gap = max(self.max_gap, gap)
                self.last_rx = received
                self.arrivals.append(received)
                self.total += 1
                self.times.append(elapsed)
                in_range = status == 'ok'
                self.raw_values.append(raw if in_range else np.nan)
                self.corrected_values.append(corrected if in_range else np.nan)
                self.raw_label.setText(f'原始距离：{raw:.1f} mm' if raw is not None else '原始距离：—')
                self._set_value_text(f'{corrected:.1f} mm' if corrected is not None else '无有效距离')
                if raw is not None and corrected is not None:
                    self.latest_raw = raw
                    self.latest_corrected = corrected
                    self.latest_offset = corrected - raw
                    self.latest_elapsed = elapsed
                    self.latest_received = received
                    self.record_comparison_button.setEnabled(True)
                else:
                    self.record_comparison_button.setEnabled(False)
                self.outside += status == 'outside_display_range'
                self.errors += status == 'device_error'
                message = '正在采集' if in_range else ('超出显示范围，曲线断开，原始记录保留' if raw is not None else f'模块错误 ERR-{code}')
                self.status.setText(f'{message} | 校验失败：{bad}')
        now = time.monotonic()
        while self.arrivals and now-self.arrivals[0] > 3:
            self.arrivals.popleft()
        rate = ((len(self.arrivals)-1)/(self.arrivals[-1]-self.arrivals[0])
                if len(self.arrivals)>1 and self.arrivals[-1]>self.arrivals[0] else 0)
        self.metrics.setText(f'近3秒回复频率：{rate:.1f} Hz | >0.25秒间断：{self.long_gaps} | '
                             f'最大间断：{self.max_gap:.2f}秒 | 总回复：{self.total} | '
                             f'超出显示范围：{self.outside} | 错误回复：{self.errors}')
        if self.worker and self.worker.is_alive() and self.last_rx is not None and now-self.last_rx > .5:
            self._set_value_text('等待新数据…')
            self.record_comparison_button.setEnabled(False)
            self.status.setText(f'已 {now-self.last_rx:.1f} 秒未收到完整距离回复')
        if self.times:
            end = self.times[-1] + (now-self.last_rx if self.worker and self.worker.is_alive() else 0)
            while self.times and self.times[0] < end-self.args.window_s:
                self.times.popleft(); self.raw_values.popleft(); self.corrected_values.popleft()
            self.raw_curve.setData(list(self.times), list(self.raw_values), connect='finite')
            self.corrected_curve.setData(list(self.times), list(self.corrected_values), connect='finite')
            self.plot.setXRange(max(0,end-self.args.window_s), max(self.args.window_s,end), padding=0)

    def closeEvent(self, event):
        if self.worker and self.worker.is_alive():
            self.close_pending = True
            self.stop()
            event.ignore()
            QtCore.QTimer.singleShot(100, self.close)
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser(description='100米模块实时曲线与原始记录')
    parser.add_argument('--port', default='COM3')
    parser.add_argument('--frequency', type=int, choices=sorted(FREQUENCY_CODES), default=10,
                        help='模块连续测量频率；动态连续性测试默认10 Hz')
    parser.add_argument('--offset-mm', type=float, default=2.0,
                        help='近距离最大软件补偿，默认+2 mm；0关闭')
    parser.add_argument('--offset-full-until-mm', type=float, default=60.0,
                        help='完整应用近距离补偿的原始距离上限，默认60 mm')
    parser.add_argument('--offset-zero-from-mm', type=float, default=70.0,
                        help='补偿线性衰减至0的原始距离，默认70 mm')
    parser.add_argument('--max-mm', type=float, default=500, help='曲线显示上限，非模块量程')
    parser.add_argument('--window-s', type=float, default=15)
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.offset_mm, args.offset_full_until_mm,
                                          args.offset_zero_from_mm, args.max_mm, args.window_s)) \
            or min(args.max_mm, args.window_s) <= 0:
        parser.error('参数须为有限数字，显示范围及窗口时长须大于0')
    if args.offset_full_until_mm < 0 or args.offset_zero_from_mm <= args.offset_full_until_mm:
        parser.error('补偿区间必须满足0 <= 完整补偿上限 < 补偿归零距离')
    if serial is None:
        parser.error('缺少pyserial，请安装后重试')
    app = QtWidgets.QApplication([])
    window = LiveWindow(args)
    window.show()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
