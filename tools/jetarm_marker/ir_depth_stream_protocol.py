#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JetArm IR + depth 配对帧 TCP 协议（JARD）。"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from tools.jetarm_marker.ir_stream_protocol import connect_stream, recv_exact

MAGIC_IR = b"JARM"
MAGIC_PAIR = b"JARD"
_HEADER_PAIR_FMT = ">4sIIHH"
_HEADER_PAIR_SIZE = struct.calcsize(_HEADER_PAIR_FMT)


@dataclass
class IrDepthFrame:
    gray: np.ndarray
    depth: Optional[np.ndarray]
    depth_encoding: str = "16UC1"


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("jpeg decode failed")
    return gray


def iter_ir_depth_frames(sock: socket.socket) -> Iterator[IrDepthFrame]:
    while True:
        header = recv_exact(sock, 8)
        magic = header[:4]
        length = struct.unpack(">I", header[4:8])[0]

        if magic == MAGIC_IR:
            payload = recv_exact(sock, length)
            yield IrDepthFrame(gray=_decode_jpeg(payload), depth=None)
            continue

        if magic != MAGIC_PAIR:
            raise ValueError(f"bad stream magic: {magic!r}")

        if length < 4:
            raise ValueError("short JARD header")
        rest = recv_exact(sock, length)
        depth_len, w, h = struct.unpack(">IHH", rest[:8])
        ir_len = length - 8 - depth_len
        if ir_len < 0:
            raise ValueError("invalid JARD lengths")
        ir_bytes = rest[8 : 8 + ir_len]
        gray = _decode_jpeg(ir_bytes)
        depth = None
        if depth_len > 0 and w > 0 and h > 0:
            depth_bytes = rest[8 + ir_len : 8 + ir_len + depth_len]
            depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(h, w).copy()
        yield IrDepthFrame(gray=gray, depth=depth)


def connect_ir_depth_stream(host: str, port: int, *, timeout_s: float = 8.0) -> socket.socket:
    return connect_stream(host, port, timeout_s=timeout_s)
