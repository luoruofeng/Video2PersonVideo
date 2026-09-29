"""检测器结构化输出测试（不加载模型，用桩对象模拟 ultralytics 的 Results）。"""

from __future__ import annotations

import numpy as np
import pytest

from video2personvideo.core.detector import PersonDetector


class _Boxes:
    def __init__(self, xyxy, conf=None, cls=None) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class _Result:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class _Tensor:
    """模拟 torch.Tensor 的 ``.cpu().numpy()``。"""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def test_to_detections_parses_boxes() -> None:
    result = _Result(
        _Boxes(
            xyxy=np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]),
            conf=np.array([0.9, 0.4]),
            cls=np.array([0, 0]),
        )
    )
    detections = PersonDetector.to_detections(result)

    assert len(detections) == 2
    assert detections[0].bbox == (10.0, 20.0, 30.0, 40.0)
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].class_id == 0
    assert detections[1].area == pytest.approx(400.0)


def test_to_detections_accepts_tensors() -> None:
    result = _Result(
        _Boxes(
            xyxy=_Tensor(np.array([[0.0, 0.0, 10.0, 10.0]])),
            conf=_Tensor(np.array([0.75])),
            cls=_Tensor(np.array([0])),
        )
    )
    detections = PersonDetector.to_detections(result)
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.75)


def test_to_detections_handles_missing_or_empty() -> None:
    assert PersonDetector.to_detections(_Result(None)) == []
    assert PersonDetector.to_detections(_Result(_Boxes(xyxy=np.zeros((0, 4))))) == []


def test_to_detections_uses_defaults_without_conf_and_cls() -> None:
    result = _Result(_Boxes(xyxy=np.array([[1.0, 2.0, 3.0, 4.0]])))
    detection = PersonDetector.to_detections(result)[0]
    assert detection.confidence == 1.0
    assert detection.class_id == 0


def test_to_detections_skips_malformed_rows() -> None:
    result = _Result(_Boxes(xyxy=np.array([[1.0, 2.0]])))
    assert PersonDetector.to_detections(result) == []
