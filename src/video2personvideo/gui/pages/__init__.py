"""向导页面：只做展示与事件转发，业务逻辑在 ``core`` 与 ``gui.controller``。"""

from __future__ import annotations

from .input_page import InputPage
from .ratio_page import RatioPage
from .result_page import ResultPage
from .run_page import RunPage
from .setup_page import SetupPage

__all__ = ["InputPage", "RatioPage", "ResultPage", "RunPage", "SetupPage"]
