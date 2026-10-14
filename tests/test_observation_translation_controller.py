import unittest

import numpy as np

from core.observation_translation_controller import ObservationTranslationController


class ObservationTranslationControllerTests(unittest.TestCase):
    def _ready(self):
        c = ObservationTranslationController()
        c.set_imu_connected(True)
        c.set_range_status(True, True)
        c.set_baseline(True, 100.0)
        c.update_axis([0, 0, -1])
        c.update_gyro([0, 0, 0], calibrated=True, now=0.0)
        c.update_gyro([0, 0, 0], calibrated=True, now=0.3)
        c.update_range(12.0, True)
        return c

    def test_space_requires_stable_pose_and_locks_axis(self):
        c = ObservationTranslationController()
        c.set_imu_connected(True)
        c.set_range_status(True, True)
        c.set_baseline(True, 100.0)
        c.update_axis([0, 0, -1])
        c.update_gyro([0, 0, 0], calibrated=True, now=0.0)
        self.assertFalse(c.toggle_lock(now=0.1))
        c.update_gyro([0, 0, 0], calibrated=True, now=0.3)
        self.assertTrue(c.toggle_lock(now=0.31))
        self.assertTrue(np.allclose(c.locked_axis, [0, 0, -1]))

    def test_axis_stability_allows_lock_without_gyro_calibration(self):
        now = [0.0]
        c = ObservationTranslationController(clock=lambda: now[0])
        c.set_imu_connected(True)
        c.set_range_status(True, True)
        c.set_baseline(True, 100.0)
        c.update_axis([0, 0, -1])
        now[0] = 0.1
        c.update_axis([0, 0, -1])
        self.assertFalse(c.toggle_lock())
        now[0] = 0.45
        c.update_axis([0, 0, -1])
        self.assertTrue(c.toggle_lock())
        self.assertTrue(c.locked)

    def test_space_locks_axis_before_range_but_blocks_displacement(self):
        c = ObservationTranslationController()
        c.set_imu_connected(True)
        c.update_axis([0, 0, -1])
        c.update_gyro([0, 0, 0], calibrated=True, now=0.0)
        self.assertTrue(c.toggle_lock(now=0.35))
        self.assertEqual(c.state, c.RANGE_BLOCKED)
        self.assertEqual(c.applied_displacement_mm, 0.0)
        c.set_baseline(False)
        self.assertTrue(c.locked)
        c.set_range_status(True, True)
        c.set_baseline(True, 100.0)
        c.update_range(25.0, True)
        c.update_axis([0, 0, -1])
        c.tick(now=0.5)
        self.assertEqual(c.state, c.LOCKED)
        self.assertEqual(c.applied_displacement_mm, 25.0)

    def test_angle_pause_keeps_lock_and_resumes_with_latest_distance(self):
        c = self._ready()
        self.assertTrue(c.toggle_lock(now=0.31))
        c.update_range(20.0, True)
        c.update_axis([1, 0, 0])
        c.tick(now=0.4)
        c.tick(now=0.61)
        self.assertEqual(c.state, c.ANGLE_PAUSED)
        self.assertEqual(c.applied_displacement_mm, 20.0)
        c.update_range(35.0, True)
        c.update_axis([0.05, 0, -1])
        c.tick(now=0.7)
        c.tick(now=1.01)
        self.assertEqual(c.state, c.LOCKED)
        self.assertEqual(c.applied_displacement_mm, 35.0)

    def test_invalid_range_freezes_and_zero_loss_exits_lock(self):
        c = self._ready()
        self.assertTrue(c.toggle_lock(now=0.31))
        c.update_range(10.0, True)
        c.update_range(None, False)
        self.assertEqual(c.state, c.RANGE_BLOCKED)
        self.assertEqual(c.applied_displacement_mm, 10.0)
        c.set_baseline(False)
        self.assertFalse(c.locked)
        self.assertEqual(c.state, c.WAITING_ZERO)

    def test_signed_displacement_moves_on_both_sides_of_zero(self):
        c = self._ready()
        self.assertTrue(c.toggle_lock(now=0.31))
        c.update_range(-12.0, True)
        self.assertEqual(c.measured_displacement_mm, -12.0)
        self.assertEqual(c.applied_displacement_mm, -12.0)


if __name__ == "__main__":
    unittest.main()
