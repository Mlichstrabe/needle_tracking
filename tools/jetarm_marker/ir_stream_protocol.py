#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JetArm IR JPEG 流：长度前缀帧协议（TCP）。"""
from __future__ import annotations

import socket
import struct
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

MAGIC = b"JARM"
HEADER_FMT = ">4sI"  # magic + uint32 payload length
HEADER_SIZE = struct.calcsize(HEADER_FMT)


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


def iter_jpeg_frames(sock: socket.socket) -> Iterator[np.ndarray]:
  while True:
    header = recv_exact(sock, HEADER_SIZE)
    magic, length = struct.unpack(HEADER_FMT, header)
    if magic != MAGIC:
      raise ValueError(f"bad stream magic: {magic!r}")
    payload = recv_exact(sock, length)
    yield decode_jpeg(payload)


def connect_stream(host: str, port: int, *, timeout_s: float = 5.0) -> socket.socket:
  sock = socket.create_connection((host, port), timeout=timeout_s)
  sock.settimeout(10.0)
  return sock


def encode_jpeg_frame(gray_u8: np.ndarray, *, quality: int = 80) -> bytes:
  ok, encoded = cv2.imencode(".jpg", gray_u8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
  if not ok:
    raise RuntimeError("jpeg encode failed")
  return encoded.tobytes()


def pack_frame(jpeg_bytes: bytes) -> bytes:
  return struct.pack(HEADER_FMT, MAGIC, len(jpeg_bytes)) + jpeg_bytes
