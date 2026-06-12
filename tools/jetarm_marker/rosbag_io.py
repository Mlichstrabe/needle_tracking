"""ROS2 bag 读取辅助（依赖 rosbags，无需本机 ROS2）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError as exc:  # pragma: no cover
    Reader = None  # type: ignore
    get_typestore = None  # type: ignore
    Stores = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

# V1 主用 topic（与 JetArm Marker Tracking Data.md 一致）
TOPIC_RGB = "/depth_cam/rgb/image_raw"
TOPIC_RGB_INFO = "/depth_cam/rgb/camera_info"
TOPIC_DEPTH = "/depth_cam/depth/image_raw"
TOPIC_DEPTH_INFO = "/depth_cam/depth/camera_info"
TOPIC_IR = "/depth_cam/ir/image_raw"
TOPIC_IR_INFO = "/depth_cam/ir/camera_info"

DEFAULT_TOPICS = (TOPIC_RGB, TOPIC_RGB_INFO, TOPIC_DEPTH, TOPIC_DEPTH_INFO, TOPIC_IR, TOPIC_IR_INFO)


def require_rosbags() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "需要安装 rosbags：pip install -r requirements-jetarm-marker.txt"
        ) from _IMPORT_ERROR


def _typestore():
    require_rosbags()
    return get_typestore(Stores.ROS2_HUMBLE)


@dataclass
class TopicSummary:
    topic: str
    msgtype: str
    count: int


@dataclass
class BagProbeResult:
    bag_path: str
    topics: List[TopicSummary] = field(default_factory=list)
    duration_ns: Optional[int] = None
    start_time_ns: Optional[int] = None
    end_time_ns: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bag_path": self.bag_path,
            "duration_s": (self.duration_ns / 1e9) if self.duration_ns else None,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "topics": [
                {"topic": t.topic, "msgtype": t.msgtype, "count": t.count}
                for t in self.topics
            ],
        }


def list_bag_topics(bag_dir: Path) -> List[TopicSummary]:
    """仅列出 topic 元数据（不统计消息数时可更快，此处与 probe 一致）。"""
    return probe_bag(bag_dir).topics


def probe_bag(bag_dir: Path) -> BagProbeResult:
    """列出 bag 内 topic、消息数与时间跨度。"""
    require_rosbags()
    bag_dir = Path(bag_dir)
    result = BagProbeResult(bag_path=str(bag_dir.resolve()))

    counts: Dict[str, Tuple[str, int]] = {}
    t_min: Optional[int] = None
    t_max: Optional[int] = None

    with Reader(bag_dir) as reader:
        for conn in reader.connections:
            counts[conn.topic] = (conn.msgtype, 0)

        for conn, timestamp, _raw in reader.messages():
            msgtype, n = counts[conn.topic]
            counts[conn.topic] = (msgtype, n + 1)
            if t_min is None or timestamp < t_min:
                t_min = timestamp
            if t_max is None or timestamp > t_max:
                t_max = timestamp

    for topic, (msgtype, count) in sorted(counts.items()):
        result.topics.append(TopicSummary(topic=topic, msgtype=msgtype, count=count))

    result.start_time_ns = t_min
    result.end_time_ns = t_max
    if t_min is not None and t_max is not None:
        result.duration_ns = t_max - t_min
    return result


def save_probe_summary(result: BagProbeResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class FrameRecord:
    topic: str
    timestamp_ns: int
    encoding: str
    width: int
    height: int
    array: np.ndarray  # HxW or HxWx3, uint8 or uint16/float


def _image_msg_to_array(msg) -> Tuple[np.ndarray, str]:
    enc = msg.encoding
    h, w = int(msg.height), int(msg.width)
    step = int(msg.step)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        row = raw.reshape(h, step)
        img = row[:, : w * 3].reshape(h, w, 3)
        if enc == "bgr8":
            img = img[:, :, ::-1].copy()
        return img, enc

    if enc in ("mono8", "8UC1"):
        row = raw.reshape(h, step)
        return row[:, :w].copy(), enc

    if enc in ("mono16", "16UC1"):
        row = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, step // 2)
        return row[:, :w].copy(), enc

    if enc == "32FC1":
        row = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(h, step // 4)
        return row[:, :w].copy(), enc

    raise ValueError(f"不支持的 encoding: {enc}")


def iter_topic_messages(bag_dir: Path, topic: str) -> Iterator[Tuple[int, Any]]:
    """按时间顺序 yield (timestamp_ns, deserialized_msg)。"""
    require_rosbags()
    ts = _typestore()
    with Reader(Path(bag_dir)) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise KeyError(f"bag 中无 topic: {topic}")
        for conn, timestamp, raw in reader.messages(connections=conns):
            yield timestamp, ts.deserialize_cdr(raw, conn.msgtype)


def collect_timestamps(bag_dir: Path, topic: str) -> List[int]:
    return [t for t, _ in iter_topic_messages(bag_dir, topic)]


def read_message_at_index(bag_dir: Path, topic: str, index: int):
    for i, (_t, msg) in enumerate(iter_topic_messages(bag_dir, topic)):
        if i == index:
            return msg
    raise IndexError(f"topic {topic} 仅有 {i + 1} 帧，请求 index={index}")


def load_image_frames(
    bag_dir: Path,
    topic: str,
    *,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> List[FrameRecord]:
    """Load decoded image frames from one topic.

    `end_index` is inclusive.  The returned records keep the source topic and
    ROS bag timestamp for later RGB/depth synchronization.
    """
    frames: List[FrameRecord] = []
    for i, (timestamp_ns, msg) in enumerate(iter_topic_messages(bag_dir, topic)):
        if i < start_index:
            continue
        if end_index is not None and i > end_index:
            break
        rec = decode_image_msg(msg)
        rec.topic = topic
        rec.timestamp_ns = timestamp_ns
        frames.append(rec)
    return frames


def decode_image_msg(msg) -> FrameRecord:
    arr, enc = _image_msg_to_array(msg)
    return FrameRecord(
        topic="",
        timestamp_ns=0,
        encoding=enc,
        width=int(msg.width),
        height=int(msg.height),
        array=arr,
    )


def camera_info_to_dict(msg) -> Dict[str, Any]:
    k = list(msg.k)
    return {
        "width": int(msg.width),
        "height": int(msg.height),
        "frame_id": str(msg.header.frame_id),
        "fx": float(k[0]),
        "fy": float(k[4]),
        "cx": float(k[2]),
        "cy": float(k[5]),
        "distortion_model": str(msg.distortion_model),
        "d": [float(x) for x in msg.d],
    }


def pick_frame_indices(n_frames: int, n_samples: int) -> List[int]:
    if n_frames <= 0:
        return []
    if n_samples >= n_frames:
        return list(range(n_frames))
    if n_samples <= 1:
        return [n_frames // 2]
    step = (n_frames - 1) / (n_samples - 1)
    return sorted({int(round(i * step)) for i in range(n_samples)})
