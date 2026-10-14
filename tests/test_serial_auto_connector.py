import unittest

from core.serial_auto_connector import _looks_like_imu, _looks_like_range, _range_command


class SerialAutoConnectorTests(unittest.TestCase):
    def test_protocol_signatures(self):
        frame = bytearray([0x55, 0x59] + [0] * 8)
        frame.append(sum(frame) & 0xFF)
        self.assertTrue(_looks_like_imu(bytes(frame)))
        self.assertFalse(_looks_like_imu(bytes(frame[:-1]) + b"\x01"))
        self.assertTrue(_looks_like_range(b"noise\x80\x06\x83\x31\x30\x30\x2e\x30\x30\x30"))
        command = _range_command(0x80, 0x06, 0x03)
        self.assertEqual(sum(command) & 0xFF, 0)


if __name__ == "__main__":
    unittest.main()
