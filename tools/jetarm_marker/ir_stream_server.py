#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetArm 侧：订阅 IR + depth，经 TCP 推送 JARD 配对帧。

  bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh
"""
from __future__ import annotations

import argparse
import socket
import struct
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

MAGIC_PAIR = b"JARD"
TOPIC_IR = "/depth_cam/ir/image_raw"
TOPIC_DEPTH = "/depth_cam/depth/image_raw"


def ir_msg_to_gray(msg: Image) -> np.ndarray:
    h, w = msg.height, msg.width
    if msg.encoding in ("mono8", "8UC1"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w).copy()
    if msg.encoding in ("mono16", "16UC1"):
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32)
        lo = float(np.percentile(raw, 1.0))
        hi = float(np.percentile(raw, 99.7))
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((raw - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    if msg.encoding == "rgb8":
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"unsupported IR encoding: {msg.encoding}")


def depth_msg_to_u16(msg: Image) -> np.ndarray:
    h, w = msg.height, msg.width
    if msg.encoding in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).copy()
    if msg.encoding == "32FC1":
        meters = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        return np.clip(meters * 1000.0, 0, 65535).astype(np.uint16)
    raise ValueError(f"unsupported depth encoding: {msg.encoding}")


class IrDepthStreamServer(Node):
    def __init__(self, *, host: str, port: int, quality: int) -> None:
        super().__init__("ir_depth_stream_server")
        self._quality = quality
        self._client: socket.socket | None = None
        self._client_lock = threading.Lock()
        self._depth_lock = threading.Lock()
        self._latest_depth: np.ndarray | None = None
        self._depth_wh: tuple[int, int] = (0, 0)
        self._sent_frames = 0
        self.create_subscription(Image, TOPIC_IR, self._on_image, 10)
        self.create_subscription(Image, TOPIC_DEPTH, self._on_depth, 10)
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(1)
        self.get_logger().info(f"IR+depth stream on {host}:{port} (JARD)")
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while rclpy.ok():
            try:
                client, addr = self._server.accept()
            except OSError:
                break
            try:
                client.setsockopt(socket.IPPROTO_TCP, socket.SO_SNDBUF, 262144)
            except OSError:
                pass
            self.get_logger().info(f"client connected: {addr}")
            with self._client_lock:
                if self._client is not None:
                    try:
                        self._client.close()
                    except OSError:
                        pass
                self._client = client

    def _on_depth(self, msg: Image) -> None:
        try:
            depth = depth_msg_to_u16(msg)
        except Exception as exc:
            self.get_logger().warning(f"depth decode failed: {exc}")
            return
        with self._depth_lock:
            self._latest_depth = depth
            self._depth_wh = (depth.shape[1], depth.shape[0])

    def _on_image(self, msg: Image) -> None:
        try:
            gray = ir_msg_to_gray(msg)
        except Exception as exc:
            self.get_logger().warning(f"IR decode failed: {exc}")
            return

        ok, encoded = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if not ok:
            return
        ir_bytes = encoded.tobytes()

        with self._depth_lock:
            depth = None if self._latest_depth is None else self._latest_depth.copy()
            dw, dh = self._depth_wh

        if depth is not None:
            depth_bytes = depth.astype(np.uint16).tobytes()
            meta = struct.pack(">IHH", len(depth_bytes), dw, dh)
            payload = meta + ir_bytes + depth_bytes
            packet = struct.pack(">4sI", MAGIC_PAIR, len(payload)) + payload
        else:
            meta = struct.pack(">IHH", 0, 0, 0)
            payload = meta + ir_bytes
            packet = struct.pack(">4sI", MAGIC_PAIR, len(payload)) + payload

        with self._client_lock:
            client = self._client
        if client is None:
            return
        try:
            client.sendall(packet)
            self._sent_frames += 1
            if self._sent_frames in (1, 30, 300):
                self.get_logger().info(f"sent frames: {self._sent_frames}")
        except OSError as exc:
            self.get_logger().warning(f"send failed: {exc}")
            with self._client_lock:
                if self._client is client:
                    self._client = None
            try:
                client.close()
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="JetArm IR+depth -> TCP JARD stream")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()

    rclpy.init()
    node = IrDepthStreamServer(host=args.host, port=args.port, quality=args.quality)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
