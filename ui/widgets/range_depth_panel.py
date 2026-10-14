"""100m 测距采集、滤波和相对深度输出。"""
import math
import queue
import statistics
import threading
import time
from collections import deque
from types import SimpleNamespace

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)
from serial.tools import list_ports
from tools.laser100m_live import capture
from ui.widgets.ui_helpers import (
    apply_panel_chrome,
    set_button_variant,
    set_label_role,
)


class RangeDepthPanel(QFrame):
    # 设备偶尔会有较长测量间隔。短时没有新帧时冻结深度即可，
    # 不清除用户已经记录的零点；恢复后下一条有效帧自动继续。
    STALE_TIMEOUT_S = 2.0

    sensor_state_changed = pyqtSignal(str, str)
    range_sample = pyqtSignal(object, object, object, float)
    range_validity_changed = pyqtSignal(bool, str)
    baseline_changed = pyqtSignal(bool, object)

    def __init__(self, occupied_port, parent=None):
        super().__init__(parent)
        self.occupied_port = occupied_port
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.baseline = None
        self.latest = None
        self.filtered_distance = None
        self.received = None
        self.history = deque(maxlen=10)
        self._median_history = deque(maxlen=3)
        self._filtered_history = deque(maxlen=10)
        self._data_valid = False
        self.lost = False
        self._sensor_state = 'pending'
        self._transport_connected = False
        self.setObjectName('RangeDepthCard')
        apply_panel_chrome(self)
        # This card is in the fixed real-time area of the right sidebar.  Its
        # controls need their full vertical layout; allowing the parent to
        # compress it makes the metrics and buttons overlap.
        self.setMinimumHeight(270)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        header = QHBoxLayout()
        title = QLabel('实时穿刺深度')
        title.setObjectName('SectionTitle')
        header.addWidget(title)
        header.addStretch()
        self.state_badge = QLabel('未连接')
        self.state_badge.setObjectName('RangeStateBadge')
        self.state_badge.setProperty('rangeState', self._sensor_state)
        header.addWidget(self.state_badge)
        layout.addLayout(header)

        self.depth = QLabel('— mm')
        self.depth.setObjectName('RangeDepthValue')
        self.depth.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.depth.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.depth.setFixedHeight(40)
        layout.addWidget(self.depth)

        metrics = QGridLayout()
        metrics.setContentsMargins(0, 0, 0, 0)
        metrics.setHorizontalSpacing(5)
        metrics.addWidget(self._make_metric_card('原始距离', 'distance'), 0, 0)
        metrics.addWidget(self._make_metric_card('滤波距离', 'filtered'), 0, 1)
        metrics.addWidget(self._make_metric_card('起始距离', 'reference'), 1, 0, 1, 2)
        layout.addLayout(metrics)

        row = QHBoxLayout()
        self.ports = QComboBox()
        row.addWidget(self.ports, 1)
        refresh = QPushButton('刷新')
        set_button_variant(refresh, 'ghost')
        refresh.setFixedHeight(30)
        refresh.clicked.connect(self.refresh_ports)
        row.addWidget(refresh)
        layout.addLayout(row)

        row = QHBoxLayout()
        self.connect_button = QPushButton('连接测距')
        set_button_variant(self.connect_button, 'primary')
        self.connect_button.setFixedHeight(30)
        self.connect_button.clicked.connect(self.start)
        row.addWidget(self.connect_button)
        self.stop_button = QPushButton('断开')
        set_button_variant(self.stop_button, 'danger')
        self.stop_button.setFixedHeight(30)
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        row.addWidget(self.stop_button)
        layout.addLayout(row)

        self.zero = QPushButton('针尖就位后 · 记录零点')
        set_button_variant(self.zero, 'secondary')
        self.zero.setFixedHeight(30)
        self.zero.clicked.connect(self.set_zero)
        self.zero.setEnabled(False)
        layout.addWidget(self.zero)

        self.status = QLabel('独立串口 9600 baud；等待连接')
        self.status.setObjectName('RangeStatus')
        self.status.setWordWrap(True)
        self.status.setMaximumHeight(34)
        layout.addWidget(self.status)

        note = QLabel('负值表示回退；深度用于进度与 3D 训练显示，不代表 CT 配准后的真实针尖位置。')
        note.setObjectName('RangeNote')
        note.setWordWrap(True)
        note.setMaximumHeight(28)
        layout.addWidget(note)

        self.refresh_ports()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)

    def _make_metric_card(self, title, attr_name):
        card = QFrame()
        card.setObjectName('RangeMetricCard')
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 6, 8, 6)
        card_layout.setSpacing(2)

        label = QLabel(title)
        set_label_role(label, 'muted')
        value = QLabel('— mm' if attr_name in ('distance', 'filtered') else '未记录')
        value.setObjectName('RangeMetricValue')
        card_layout.addWidget(label)
        card_layout.addWidget(value)
        setattr(self, attr_name, value)
        return card

    def _set_sensor_state(self, state, text):
        if self._sensor_state == state and self.state_badge.text() == text:
            return
        self._sensor_state = state
        self.state_badge.setText(text)
        self.state_badge.setProperty('rangeState', state)
        self.state_badge.style().unpolish(self.state_badge)
        self.state_badge.style().polish(self.state_badge)
        self.sensor_state_changed.emit(state, text)

    def _set_data_valid(self, valid, reason=''):
        valid = bool(valid)
        if self._data_valid == valid and valid:
            return
        self._data_valid = valid
        self.range_validity_changed.emit(valid, reason)

    def _set_baseline(self, baseline):
        self.baseline = baseline
        if baseline is None:
            self.reference.setText('未记录')
            self.baseline_changed.emit(False, None)
        else:
            self.reference.setText(f'{baseline:.1f} mm')
            self.baseline_changed.emit(True, float(baseline))

    def refresh_ports(self):
        if self.worker and self.worker.is_alive():
            return
        self.ports.clear()
        for port in list_ports.comports():
            self.ports.addItem(port.device, port.device)

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        port = self.ports.currentData()
        if not port or port == self.occupied_port():
            self.status.setText('请选择未被 IMU 占用的独立串口')
            return
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self._set_baseline(None)
        self._transport_connected = False
        self.latest = self.received = None
        self.filtered_distance = None
        self.history.clear()
        self._median_history.clear()
        self._filtered_history.clear()
        self._set_data_valid(False, '等待测距数据')
        self.lost = False
        self.depth.setText('— mm')
        self.distance.setText('— mm')
        self.filtered.setText('— mm')
        self.reference.setText('未记录')
        args = SimpleNamespace(port=port, frequency=10, offset_mm=0.,
                               offset_full_until_mm=60., offset_zero_from_mm=70., max_mm=500.)
        self.worker = threading.Thread(target=capture, args=(args, self.events, self.stop_event), daemon=True)
        self.ports.setEnabled(False)
        self.connect_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText('正在连接…')
        self._set_sensor_state('active', '连接中')
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self._transport_connected = False
        self.distance.setText('— mm')
        self.filtered.setText('— mm')
        self.invalidate('已停止，等待串口释放', clear_baseline=True,
                        reset_measurement=True)
        self.stop_button.setEnabled(False)
        self._set_sensor_state('pending', '未连接')

    def invalidate(self, message, clear_baseline=False, reset_measurement=False):
        """暂停深度应用。

        瞬时坏帧只使当前数据无效，保留零点和最后可信滤波值；只有停止
        或采集线程结束时才清空测量显示，避免界面在坏帧/好帧之间闪烁。
        """
        if reset_measurement:
            self.latest = None
            self.filtered_distance = None
            self.history.clear()
            self._median_history.clear()
            self._filtered_history.clear()
            self.depth.setText('— mm')
            self.filtered.setText('— mm')
        if self.baseline is None:
            self.zero.setEnabled(False)
        self.status.setText(message)
        self._set_data_valid(False, message)
        if clear_baseline:
            self._set_baseline(None)

    def accept_sample(self, raw, status, received):
        if self.received is not None and received - self.received > self.STALE_TIMEOUT_S:
            # 长间隔后重新建立滤波窗口，但不动用户已经记录的零点。
            self.history.clear()
            self._median_history.clear()
            self._filtered_history.clear()
            self.filtered_distance = None
        self.received = received
        if status != 'ok' or raw is None or not math.isfinite(raw) or not 0 < raw <= 500:
            if raw is not None and math.isfinite(raw):
                self.distance.setText(f'{raw:.1f} mm')
            else:
                self.distance.setText('无效')
            self.invalidate('测距暂失；深度暂停，零点保留')
            return

        # 不再因相邻帧大跳变反复进入“异常/重新记录”状态。3 点中值和
        # EMA 继续用于控制输入；超出训练量程的值已在上面被拦截。
        self.latest = raw
        self.history.append((received, raw))
        self._median_history.append(float(raw))
        median_distance = statistics.median(self._median_history)
        if self.filtered_distance is None:
            self.filtered_distance = median_distance
        else:
            self.filtered_distance = 0.6 * median_distance + 0.4 * self.filtered_distance
        self._filtered_history.append(self.filtered_distance)
        self.distance.setText(f'{raw:.1f} mm')
        self.filtered.setText(f'{self.filtered_distance:.1f} mm')
        # 首个新鲜有效读数即可由用户主动记录零点，不再强制等待 10 帧。
        self.zero.setEnabled(True)
        if self._sensor_state == 'active':
            self._set_sensor_state('done', '已连接')
        self._set_data_valid(True, '')
        if self.baseline is not None:
            measured_depth = self.baseline - self.filtered_distance
            self.depth.setText(f'{measured_depth:+.1f} mm')
            self.status.setText('实时更新 · 滤波距离差分')
        else:
            measured_depth = None
            self.status.setText('已有有效读数，可直接记录零点')
        age_s = max(0.0, time.monotonic() - received)
        self.range_sample.emit(raw, self.filtered_distance, measured_depth, age_s)

    def set_zero(self):
        if (not self.zero.isEnabled() or self.received is None
                or time.monotonic() - self.received > self.STALE_TIMEOUT_S):
            return
        if self.filtered_distance is None or not self._filtered_history:
            return
        self._set_baseline(statistics.median(self._filtered_history))
        self.depth.setText(f'{self.baseline - self.filtered_distance:+.1f} mm')

    def clear_baseline(self):
        """清除零点但保留当前测距采集，供重置/重选流程调用。"""
        if self.baseline is None:
            self.reference.setText('未记录')
            self.depth.setText('— mm')
            self.baseline_changed.emit(False, None)
            return
        self._set_baseline(None)
        self.depth.setText('— mm')

    def poll(self):
        for _ in range(200):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == 'finished':
                self._transport_connected = False
                reason = event[2] if len(event) > 2 and event[2] else '采集结束'
                self.invalidate(reason + '；记录：' + event[1],
                                clear_baseline=True, reset_measurement=True)
                self.ports.setEnabled(True)
                self.connect_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self._set_sensor_state('pending', '未连接')
            elif not self.stop_event.is_set():
                if event[0] == 'connected':
                    self._transport_connected = True
                    self.status.setText(event[1])
                elif event[0] == 'frame':
                    self.accept_sample(event[3], event[5], event[2])
                elif event[0] in ('status', 'config_text'):
                    self.status.setText(event[1])
        if self.latest is not None and self.received is not None and time.monotonic() - self.received > self.STALE_TIMEOUT_S:
            self.distance.setText('超时')
            self.invalidate('测距暂时无新帧；深度暂停，零点保留')

    def shutdown(self):
        self.stop()
        self.timer.stop()
        if self.worker:
            self.worker.join(timeout=3)
