"""YOLO 人像检测器：模型加载、推理、画框。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DEFAULT_MODEL, PERSON_CLASS_ID
from ..utils.device import resolve_device
from ..utils.logger import get_logger
from .subject import Detection

logger = get_logger(__name__)


def _to_numpy(value: Any) -> np.ndarray | None:
    """把 torch.Tensor / ndarray 统一成 numpy 数组。"""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        return value.cpu().numpy()
    except AttributeError:
        return np.asarray(value)


def _warn_if_weights_missing(model_path: str) -> None:
    """首次运行时提示权重会自动下载（或可离线放置到 assets/models/）。"""
    if Path(model_path).exists():
        return
    if not model_path.endswith((".pt", ".onnx", ".engine")):
        return  # 非权重文件名，交给 ultralytics 自己处理
    logger.info(
        "未找到本地权重 %s：首次运行会自动联网下载（约 5MB）。"
        "若无网络，可手动下载后放到 assets/models/ 并用 -m assets/models/%s 指定。",
        model_path,
        Path(model_path).name,
    )


class PersonDetector:
    """对 ultralytics YOLO 的轻量封装，专注"检测画面中的人"。

    - ``predict``：跑一次推理，返回 ultralytics 的 ``Results`` 列表
    - ``annotate``：把检测框/标签/置信度画到图上（YOLO 自带绘图）
    - ``count``：统计该帧的人数
    """

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        classes: Iterable[int] | None = (PERSON_CLASS_ID,),
        max_det: int = 300,
        device: str | None = None,
        verbose: bool = False,
        use_keypoints: bool = True,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - 未安装依赖时才触发
            raise RuntimeError(
                "未安装 ultralytics，请先执行：pip install -r requirements.txt"
            ) from exc

        self.model_path = str(model)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.classes = [int(c) for c in classes] if classes else None
        self.device = resolve_device(device)
        self.verbose = verbose
        self.use_keypoints = bool(use_keypoints)
        #: 模型是否自带姿态关键点（``*-pose.pt``）
        self.has_keypoints = "pose" in Path(self.model_path).stem.lower()

        _warn_if_weights_missing(self.model_path)
        logger.info("加载模型 %s（device=%s, imgsz=%s, conf=%s）", self.model_path, self.device, self.imgsz, self.conf)
        self.model = YOLO(self.model_path)
        self.names: dict[int, str] = dict(getattr(self.model, "names", {}) or {})
        logger.debug("模型类别：%s", self.names)
        if self.keypoints_enabled:
            logger.info("检测到姿态模型：将用关键点精修半身构图（头顶 / 腰部锚点）")

    @property
    def keypoints_enabled(self) -> bool:
        """是否真的会解析并使用关键点。"""
        return self.use_keypoints and self.has_keypoints

    # ------------------------------------------------------------------ 推理
    def predict(self, frame: np.ndarray | Sequence[np.ndarray], *, stream: bool = False) -> Any:
        """对单帧（BGR ndarray）或一组帧做推理。

        ``stream=False`` 返回 ``Results`` 列表；``stream=True`` 返回生成器，
        配合 ``batch`` 可以把多帧塞进同一次前向计算（见 :meth:`detect_boxes_batch`）。
        """
        return self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            max_det=self.max_det,
            device=self.device,
            verbose=self.verbose,
            stream=bool(stream),
            batch=len(frame) if isinstance(frame, (list, tuple)) else 1,
        )

    def annotate(self, result: Any) -> np.ndarray:
        """把检测结果画到图像上（YOLO 内置绘图：框 + 标签 + 置信度）。"""
        return result.plot()

    @staticmethod
    def count(result: Any) -> int:
        """统计一帧里的目标数量。"""
        boxes = getattr(result, "boxes", None)
        return 0 if boxes is None else len(boxes)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """一次调用完成"推理 + 画框"，返回 (标注后的图, 人数)。"""
        results = self.predict(frame)
        if not results:
            return frame, 0
        result = results[0]
        return self.annotate(result), self.count(result)

    # ------------------------------------------------------- 结构化检测结果
    def detect_boxes(self, frame: np.ndarray) -> list[Detection]:
        """只做推理，返回结构化检测结果（bbox / 置信度 / 类别），不做绘图。

        裁剪构图走这条路径，不依赖 YOLO ``plot()`` 的渲染结果。
        """
        results = self.predict(frame)
        if not results:
            return []
        return self.to_detections(results[0], with_keypoints=self.use_keypoints)

    def detect_boxes_batch(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        """批量推理（M1-7）：把多帧塞进同一次前向计算，返回与输入等长的结果列表。

        与逐帧调用 :meth:`detect_boxes` 的结果语义完全一致，只是吞吐更高。
        """
        if not frames:
            return []
        if len(frames) == 1:
            return [self.detect_boxes(frames[0])]

        # stream=True 配合 batch=N：一次前向算完这一批帧
        results = self.predict(list(frames), stream=True)
        detections = [
            self.to_detections(result, with_keypoints=self.use_keypoints) for result in results
        ]

        # 数量对不上时补齐/截断，保证调用方能按位置一一对应
        if len(detections) < len(frames):  # pragma: no cover - 极端兼容分支
            logger.warning("批量推理返回 %d 个结果，少于输入的 %d 帧", len(detections), len(frames))
            detections.extend([] for _ in range(len(frames) - len(detections)))
        return detections[: len(frames)]

    @staticmethod
    def to_detections(result: Any, *, with_keypoints: bool = True) -> list[Detection]:
        """把 ultralytics 的 ``Results`` 转成 :class:`Detection` 列表。"""
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = _to_numpy(getattr(boxes, "xyxy", None))
        if xyxy is None or len(xyxy) == 0:
            return []
        conf = _to_numpy(getattr(boxes, "conf", None))
        cls = _to_numpy(getattr(boxes, "cls", None))
        keypoints = PersonDetector.to_keypoints(result) if with_keypoints else None

        detections: list[Detection] = []
        for index, row in enumerate(xyxy):
            if len(row) < 4:
                continue
            confidence = float(conf[index]) if conf is not None and index < len(conf) else 1.0
            class_id = int(cls[index]) if cls is not None and index < len(cls) else 0
            detections.append(
                Detection(
                    bbox=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                    confidence=confidence,
                    class_id=class_id,
                    keypoints=(
                        keypoints[index]
                        if keypoints is not None and index < len(keypoints)
                        else None
                    ),
                )
            )
        return detections

    @staticmethod
    def to_keypoints(result: Any) -> tuple[tuple[tuple[float, float, float], ...], ...] | None:
        """把姿态模型的 ``Results.keypoints`` 转成 ``((x, y, conf), ...)`` 嵌套元组。

        非姿态模型（没有 ``keypoints`` 字段）返回 ``None``。
        """
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return None
        xy = _to_numpy(getattr(keypoints, "xy", None))
        if xy is None or len(xy) == 0:
            return None
        conf = _to_numpy(getattr(keypoints, "conf", None))

        parsed: list[tuple[tuple[float, float, float], ...]] = []
        for index in range(len(xy)):
            points: list[tuple[float, float, float]] = []
            for joint in range(len(xy[index])):
                point = xy[index][joint]
                if len(point) < 2:
                    continue
                if conf is not None and index < len(conf) and joint < len(conf[index]):
                    score = float(conf[index][joint])
                else:
                    score = 1.0
                points.append((float(point[0]), float(point[1]), score))
            parsed.append(tuple(points))
        return tuple(parsed) if parsed else None

    # ------------------------------------------------------------------ 信息
    @property
    def class_names(self) -> Sequence[str]:
        return [self.names[key] for key in sorted(self.names)]

    def warmup(self, imgsz: int | None = None) -> None:
        """用一张空白图预热模型，避免首帧卡顿。"""
        size = int(imgsz or self.imgsz)
        blank = np.zeros((size, size, 3), dtype=np.uint8)
        try:
            self.predict(blank)
            logger.debug("模型预热完成（%sx%s）", size, size)
        except Exception as exc:  # noqa: BLE001 - 预热失败不影响主流程
            logger.warning("模型预热失败：%s", exc)
