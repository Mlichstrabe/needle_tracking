#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JetArm IR + depth TCP 流协议（JARM 单 IR JPEG，JARD IR+depth 配对帧）。"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

MAGIC_IR = b"JARM"
MAGIC_PAIR = b"JARD"
_HEADER_JARM_FMT = ">4sI"
_HEADER_JARM_SIZE = struct.calcsize(_HEADER_JARM_FMT)


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("stream closed")
        buf.extend(chunk)
    return bytes(buf)


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("jpeg decode failed")
    return gray


def connect_stream(host: str, port: int, *, timeout_s: float = 5.0) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=timeout_s)
    sock.settimeout(10.0)
    return sock


def encode_jpeg_frame(gray_u8: np.ndarray, *, quality: int = 80) -> bytes:
    ok, encoded = cv2.imencode(".jpg", gray_u8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return encoded.tobytes()


def pack_jarm_frame(jpeg_bytes: bytes) -> bytes:
    return struct.pack(_HEADER_JARM_FMT, MAGIC_IR, len(jpeg_bytes)) + jpeg_bytes


def iter_jpeg_frames(sock: socket.socket) -> Iterator[np.ndarray]:
    while True:
        header = recv_exact(sock, _HEADER_JARM_SIZE)
        magic, length = struct.unpack(_HEADER_JARM_FMT, header)
        if magic != MAGIC_IR:
            raise ValueError(f"bad stream magic: {magic!r}")
        payload = recv_exact(sock, length)
        yield decode_jpeg(payload)


@dataclass
class IrDepthFrame:
    gray: np.ndarray
    depth: Optional[np.ndarray]
    depth_encoding: str = "16UC1"


def iter_ir_depth_frames(sock: socket.socket) -> Iterator[IrDepthFrame]:
    while True:
        header = recv_exact(sock, 8)
        magic = header[:4]
        length = struct.unpack(">I", header[4:8])[0]

        if magic == MAGIC_IR:
            payload = recv_exact(sock, length)
            yield IrDepthFrame(gray=decode_jpeg(payload), depth=None)
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
        gray = decode_jpeg(ir_bytes)
        depth = None
        if depth_len > 0 and w > 0 and h > 0:
            depth_bytes = rest[8 + ir_len : 8 + ir_len + depth_len]
            depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(h, w).copy()
        yield IrDepthFrame(gray=gray, depth=depth)


def connect_ir_depth_stream(host: str, port: int, *, timeout_s: float = 8.0) -> socket.socket:
    return connect_stream(host, port, timeout_s=timeout_s)
