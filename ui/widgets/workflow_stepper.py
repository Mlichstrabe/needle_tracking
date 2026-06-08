"""顶栏训练流程步骤条。"""
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt


class WorkflowStepBar(QWidget):
    """四步：加载 CT → 选 Entry → 连接 IMU → 对准 Target。"""

    STEPS = ("加载 CT", "选 Entry", "连接 IMU", "对准 Target")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._chip_labels = []
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        for i, name in enumerate(self.STEPS):
            if i > 0:
                sep = QLabel("›")
                sep.setObjectName("StepSeparator")
                row.addWidget(sep)

            chip = QFrame()
            chip.setObjectName("WorkflowStep")
            chip.setProperty("stepState", "pending")
            chip_layout = QHBoxLayout(chip)
            chip_layout.setContentsMargins(8, 4, 8, 4)
            chip_layout.setSpacing(6)

            num = QLabel(str(i + 1))
            num.setObjectName("StepNumber")
            text = QLabel(name)
            text.setObjectName("StepText")
            chip_layout.addWidget(num)
            chip_layout.addWidget(text)
            row.addWidget(chip)
            self._chip_labels.append(chip)

        row.addStretch()

    def set_step_state(self, index, state):
        """state: pending | active | done"""
        if 0 <= index < len(self._chip_labels):
            chip = self._chip_labels[index]
            chip.setProperty("stepState", state)
            chip.style().unpolish(chip)
            chip.style().polish(chip)

    def set_states(self, states):
        for i, state in enumerate(states):
            self.set_step_state(i, state)
