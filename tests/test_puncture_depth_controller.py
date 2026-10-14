import unittest

from core.puncture_depth_controller import PunctureDepthController


class PunctureDepthControllerTests(unittest.TestCase):
    def _ready(self):
        controller = PunctureDepthController()
        controller.set_plan_depth(100.0)
        controller.set_imu_connected(True)
        controller.set_range_status(True, valid=True)
        controller.set_baseline(True, 100.0)
        controller.update_range(100.0, 100.0, 0.0)
        return controller

    def test_angle_requires_hold_and_no_baseline_blocks_puncture(self):
        controller = PunctureDepthController()
        controller.set_plan_depth(100.0)
        controller.set_imu_connected(True)
        controller.set_range_status(True, valid=True)
        controller.update_range(100.0, 100.0, 0.0)

        controller.update_angle(2.0, now=0.0)
        controller.update_angle(2.0, now=0.5)
        self.assertNotEqual(controller.state, PunctureDepthController.PUNCTURING)

        controller.set_baseline(True, 100.0)
        controller.update_range(100.0, 100.0, 0.0)
        controller.update_angle(2.0, now=1.0)
        events = controller.update_angle(2.0, now=1.3)
        self.assertEqual(controller.state, PunctureDepthController.PUNCTURING)
        self.assertTrue(any(e["type"] == "depth_applied" for e in events))

    def test_angle_pause_hysteresis_and_resume(self):
        controller = self._ready()
        controller.update_angle(2.0, now=0.0)
        controller.update_angle(2.0, now=0.3)
        self.assertEqual(controller.state, PunctureDepthController.PUNCTURING)

        controller.update_range(90.0, 90.0, 10.0)
        controller.update_angle(8.0, now=0.4)
        self.assertEqual(controller.state, PunctureDepthController.PUNCTURING)
        controller.update_angle(8.1, now=0.5)
        self.assertEqual(controller.state, PunctureDepthController.ANGLE_PAUSED)

        paused_events = controller.update_range(80.0, 80.0, 20.0)
        self.assertFalse(any(e["type"] == "depth_applied" for e in paused_events))
        controller.update_angle(5.1, now=0.8)
        self.assertEqual(controller.state, PunctureDepthController.ANGLE_PAUSED)
        controller.update_angle(5.0, now=1.0)
        events = controller.update_angle(5.0, now=1.3)
        self.assertEqual(controller.state, PunctureDepthController.PUNCTURING)
        self.assertTrue(any(e["type"] == "depth_applied" for e in events))
        self.assertEqual(controller.applied_depth_mm, 20.0)

    def test_completion_requires_three_valid_samples_and_clamps_retreat(self):
        controller = self._ready()
        controller.update_angle(2.0, now=0.0)
        controller.update_angle(2.0, now=0.3)
        for i in range(2):
            events = controller.update_range(0.0, 0.0, 100.0, now=i)
            self.assertFalse(any(e["type"] == "completed" for e in events))
        events = controller.update_range(0.0, 0.0, 100.0, now=2.0)
        self.assertEqual(controller.state, PunctureDepthController.COMPLETE)
        self.assertEqual(sum(e["type"] == "completed" for e in events), 1)

        controller = self._ready()
        controller.update_angle(2.0, now=0.0)
        controller.update_angle(2.0, now=0.3)
        controller.update_range(105.0, 105.0, -5.0)
        self.assertEqual(controller.applied_depth_mm, 0.0)
        self.assertEqual(controller.state, PunctureDepthController.PUNCTURING)


if __name__ == "__main__":
    unittest.main()
