"""姿态关键点增强测试（M1-3 可选增强）。

覆盖：锚点估计、信息不足时的退回、构图分档对锚点的使用范围、
检测器关键点解析、以及流水线端到端的比例/尺寸不变量。
"""

from __future__ import annotations

import numpy as np
import pytest

from video2personvideo.core.framing import (
    MODE_CLOSEUP,
    MODE_FULLBODY,
    MODE_HALFBODY,
    compute_target_box,
    mode_label,
)
from video2personvideo.core.pipeline import CropPipeline
from video2personvideo.core.pose import (
    DEFAULT_KEYPOINT_CONF,
    FACE_POINTS,
    HEAD_TO_BODY_RATIO,
    HIP_POINTS,
    LEFT_HIP,
    NOSE,
    BodyAnchors,
    build_anchors,
    estimate_head_height,
    mean_y,
    min_y,
    shoulder_width,
)
from video2personvideo.core.ratio import parse_ratio
from video2personvideo.core.subject import Detection, select_subject

RATIO = parse_ratio("9:16")
FRAME = (1920.0, 1080.0)
HALFBODY_BBOX = (860.0, 200.0, 1060.0, 900.0)
NO_ANCHOR_BOX = compute_target_box(HALFBODY_BBOX, FRAME, RATIO)


# ------------------------------------------------------------ 锚点估计
def test_build_anchors_from_synthetic_pose(pose_keypoints) -> None:
    bbox = (100.0, 100.0, 200.0, 400.0)  # 宽 100 高 300
    keypoints = pose_keypoints(bbox)
    anchors = build_anchors(keypoints, bbox)

    assert anchors is not None
    # 面部最上点 = 眼睛 y = 100 + 0.06*300 = 118；肩宽 = 0.22*100 = 22 → 头高 19.8
    assert anchors.head_top == pytest.approx(118.0 - 0.5 * 0.9 * 22.0)
    assert anchors.waist == pytest.approx(100.0 + 0.5 * 300.0)
    assert anchors.span == pytest.approx(anchors.waist - anchors.head_top)
    assert anchors.center == pytest.approx((anchors.head_top + anchors.waist) / 2)


def test_head_top_is_pulled_back_when_bbox_includes_hands(pose_keypoints) -> None:
    """抬手时 bbox 顶边会高出一截，关键点应把锚点拉回真正的头顶。"""
    bbox = (100.0, 0.0, 200.0, 400.0)
    anchors = build_anchors(pose_keypoints(bbox), bbox)

    assert anchors is not None
    assert anchors.head_top > bbox[1]  # 不再贴着被抬高的 bbox 顶边


def test_missing_hips_falls_back(pose_keypoints) -> None:
    bbox = (100.0, 100.0, 200.0, 400.0)
    keypoints = list(pose_keypoints(bbox))
    for index in HIP_POINTS:
        keypoints[index] = (0.0, 0.0, 0.0)

    assert build_anchors(tuple(keypoints), bbox) is None


def test_missing_face_falls_back(pose_keypoints) -> None:
    bbox = (100.0, 100.0, 200.0, 400.0)
    keypoints = list(pose_keypoints(bbox))
    for index in FACE_POINTS:
        keypoints[index] = (0.0, 0.0, 0.0)

    assert build_anchors(tuple(keypoints), bbox) is None


def test_low_confidence_points_are_ignored(pose_keypoints) -> None:
    bbox = (100.0, 100.0, 200.0, 400.0)
    keypoints = list(pose_keypoints(bbox, confidence=0.3))
    assert build_anchors(tuple(keypoints), bbox) is None

    keypoints = list(pose_keypoints(bbox, confidence=DEFAULT_KEYPOINT_CONF))
    assert build_anchors(tuple(keypoints), bbox) is not None


def test_none_and_empty_keypoints() -> None:
    assert build_anchors(None, HALFBODY_BBOX) is None
    assert build_anchors((), HALFBODY_BBOX) is None
    assert min_y(None, FACE_POINTS) is None
    assert mean_y(None, HIP_POINTS) is None


def test_shoulder_width_needs_both_sides(pose_keypoints) -> None:
    bbox = (100.0, 100.0, 200.0, 400.0)
    keypoints = list(pose_keypoints(bbox))
    assert shoulder_width(tuple(keypoints)) == pytest.approx(0.22 * 100.0)

    keypoints[6] = (0.0, 0.0, 0.0)
    assert shoulder_width(tuple(keypoints)) is None


def test_head_height_falls_back_to_body_ratio(pose_keypoints) -> None:
    bbox = (100.0, 100.0, 200.0, 400.0)
    keypoints = list(pose_keypoints(bbox))
    keypoints[5] = (0.0, 0.0, 0.0)
    keypoints[6] = (0.0, 0.0, 0.0)

    assert estimate_head_height(tuple(keypoints), bbox) == pytest.approx(
        HEAD_TO_BODY_RATIO * 300.0
    )


def test_mean_y_uses_only_confident_points(pose_keypoints) -> None:
    bbox = (100.0, 100.0, 200.0, 400.0)
    keypoints = list(pose_keypoints(bbox))
    keypoints[LEFT_HIP] = (0.0, 0.0, 0.0)  # 只剩右胯

    assert mean_y(tuple(keypoints), HIP_POINTS) == pytest.approx(100.0 + 0.5 * 300.0)


# ------------------------------------------------------------ 构图联动
def test_anchors_make_halfbody_crop_tighter(pose_keypoints) -> None:
    anchors = build_anchors(pose_keypoints(HALFBODY_BBOX), HALFBODY_BBOX)
    assert anchors is not None

    box = compute_target_box(HALFBODY_BBOX, FRAME, RATIO, anchors=anchors)

    assert box.mode == MODE_HALFBODY
    assert box.matches_ratio(RATIO, tol=1e-9)
    assert box.h < NO_ANCHOR_BOX.h  # 半身应当比"整条 bbox"更紧
    assert box.h == pytest.approx(anchors.span / 0.70)
    # 框顶落在头顶之上 headroom 个框高处
    assert box.y == pytest.approx(anchors.head_top - 0.08 * box.h, abs=0.5)


def test_anchors_do_not_apply_to_fullbody(pose_keypoints) -> None:
    bbox = (950.0, 500.0, 970.0, 530.0)  # 极小人物 → 全身档
    anchors = build_anchors(pose_keypoints(bbox), bbox)
    assert anchors is not None

    plain = compute_target_box(bbox, FRAME, RATIO)
    with_anchors = compute_target_box(bbox, FRAME, RATIO, anchors=anchors)

    assert plain.mode == MODE_FULLBODY
    assert with_anchors.h == pytest.approx(plain.h)
    assert with_anchors.y == pytest.approx(plain.y)


def test_anchors_do_not_apply_to_closeup(pose_keypoints) -> None:
    bbox = (700.0, 10.0, 1200.0, 1075.0)  # 占满画面 → 头像/大特写
    anchors = build_anchors(pose_keypoints(bbox), bbox)
    assert anchors is not None

    plain = compute_target_box(bbox, FRAME, RATIO)
    with_anchors = compute_target_box(bbox, FRAME, RATIO, anchors=anchors)

    assert plain.mode == MODE_CLOSEUP
    assert with_anchors.h == pytest.approx(plain.h)


@pytest.mark.parametrize("name", ["16:9", "9:16", "1:1", "4:5", "2:1"])
def test_anchors_keep_ratio_and_inside_frame(name: str, pose_keypoints) -> None:
    ratio = parse_ratio(name)
    anchors = build_anchors(pose_keypoints(HALFBODY_BBOX), HALFBODY_BBOX)

    box = compute_target_box(HALFBODY_BBOX, FRAME, ratio, anchors=anchors)

    assert box.matches_ratio(ratio, tol=1e-6)
    assert box.x >= -1e-6 and box.y >= -1e-6
    assert box.x + box.w <= FRAME[0] + 1e-6
    assert box.y + box.h <= FRAME[1] + 1e-6


def test_invalid_anchors_are_ignored(pose_keypoints) -> None:
    """锚点跨度为 0（头顶与腰重合）时退回 bbox 逻辑。"""
    degenerate = BodyAnchors(head_top=500.0, waist=500.0)
    box = compute_target_box(HALFBODY_BBOX, FRAME, RATIO, anchors=degenerate)
    assert box.h == pytest.approx(NO_ANCHOR_BOX.h)


def test_mode_label_still_reports_halfbody() -> None:
    assert mode_label(MODE_HALFBODY) == "半身"


# ------------------------------------------------------- 检测器关键点解析
class _KeypointTensor:
    def __init__(self, xy, conf) -> None:
        self.xy = xy
        self.conf = conf


class _Result:
    def __init__(self, boxes, keypoints=None) -> None:
        self.boxes = boxes
        self.keypoints = keypoints


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
        self.conf = np.array([0.9, 0.8])
        self.cls = np.array([0, 0])


def _pose_result() -> _Result:
    xy = np.zeros((2, 17, 2), dtype=float)
    conf = np.zeros((2, 17), dtype=float)
    xy[:, NOSE] = (12.0, 25.0)
    conf[:, NOSE] = 0.9
    xy[:, HIP_POINTS[0]] = (14.0, 38.0)
    conf[:, HIP_POINTS[0]] = 0.8
    return _Result(_Boxes(), _KeypointTensor(xy, conf))


def test_detector_parses_keypoints() -> None:
    from video2personvideo.core.detector import PersonDetector

    detections = PersonDetector.to_detections(_pose_result())

    assert len(detections) == 2
    assert detections[0].keypoints is not None and len(detections[0].keypoints) == 17
    assert detections[0].keypoints[NOSE] == pytest.approx((12.0, 25.0, 0.9))


def test_detector_can_skip_keypoints() -> None:
    from video2personvideo.core.detector import PersonDetector

    detections = PersonDetector.to_detections(_pose_result(), with_keypoints=False)
    assert all(item.keypoints is None for item in detections)


def test_detector_without_keypoints_field() -> None:
    from video2personvideo.core.detector import PersonDetector

    detections = PersonDetector.to_detections(_Result(_Boxes()))
    assert all(item.keypoints is None for item in detections)
    assert PersonDetector.to_keypoints(_Result(_Boxes())) is None


# ------------------------------------------------------------- 流水线联动
#: 640x480 画面里一个占高 50% 的人 → 半身档，且远大于 min_box_px，便于比较取景框大小
PIPE_BBOX = (240.0, 120.0, 400.0, 360.0)


def _pipe_frames(count: int = 6) -> list[np.ndarray]:
    return [np.full((480, 640, 3), 100, dtype=np.uint8) for _ in range(count)]


def _run(pipeline: CropPipeline, frames: list[np.ndarray]):
    outcomes = []
    for index, frame in enumerate(frames):
        outcomes.extend(pipeline.process(frame, index))
    outcomes.extend(pipeline.flush())
    return outcomes


def test_pipeline_uses_keypoints_and_keeps_invariants(fake_detector, pose_keypoints) -> None:
    def provider(frame, index):
        return [PIPE_BBOX]

    def keypoints(frame, index):
        return [pose_keypoints(PIPE_BBOX)]

    plain = _run(CropPipeline(RATIO, fake_detector(provider)), _pipe_frames())
    posed = _run(
        CropPipeline(RATIO, fake_detector(provider, keypoints)), _pipe_frames()
    )

    assert len(plain) == len(posed) == 6
    # 不变量 2：输出尺寸全程恒定
    assert {item.frame.shape[:2] for item in posed} == {(1920, 1080)}
    for outcome in posed:
        # 不变量 3：取景框比例恒定
        assert outcome.box.matches_ratio(RATIO, tol=1e-6)
        assert outcome.mode == MODE_HALFBODY
    # 关键点让半身取景更紧（下边界落在腰上而不是整条 bbox 底边）
    assert posed[0].box.h < plain[0].box.h


def test_pipeline_ignores_unusable_keypoints(fake_detector) -> None:
    def provider(frame, index):
        return [PIPE_BBOX]

    def broken_keypoints(frame, index):
        # 全 0 置信度 → 锚点不可用，应退回 bbox 逻辑
        return [tuple((0.0, 0.0, 0.0) for _ in range(17))]

    plain = _run(CropPipeline(RATIO, fake_detector(provider)), _pipe_frames(3))
    broken = _run(
        CropPipeline(RATIO, fake_detector(provider, broken_keypoints)), _pipe_frames(3)
    )

    assert [item.mode for item in broken] == [item.mode for item in plain]
    for left, right in zip(plain, broken, strict=True):
        assert left.box.h == pytest.approx(right.box.h)
        assert left.box.y == pytest.approx(right.box.y)


def test_select_subject_carries_keypoints(pose_keypoints) -> None:
    bbox = (860.0, 200.0, 1060.0, 900.0)
    detection = Detection(bbox=bbox, confidence=0.9, keypoints=pose_keypoints(bbox))
    subject = select_subject([detection], FRAME)

    assert subject is not None
    assert subject.keypoints is not None
    assert build_anchors(subject.keypoints, subject.bbox) is not None
