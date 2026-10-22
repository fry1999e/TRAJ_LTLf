"""
输出与格式化辅助函数。

这些函数不参与 LTLf 语义判断，只负责把内部数据打印成人容易阅读的文本。
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from ltlf_checker import Trace


def format_truth(value: bool) -> str:
    """把 True/False 转成人类可读的中文标签。"""

    return "满足" if value else "不满足"


def format_trace(trace: Trace) -> str:
    """
    把有限轨迹格式化为：
        t0:{request} -> t1:{} -> t2:{grant}
    """

    pieces: list[str] = []
    for index, step in enumerate(trace):
        if step:
            atoms = ", ".join(sorted(step))
            pieces.append(f"t{index}:{{{atoms}}}")
        else:
            pieces.append(f"t{index}:{{}}")
    return " -> ".join(pieces)


def ratio_text(value: float) -> str:
    """把 0.666666 这类小数变成更适合阅读的百分比。"""

    return f"{value * 100:.1f}%"


def write_result_file(
    text: str,
    output_path: str = "outputs/result.txt",
    *,
    preserve_existing: bool = False,
) -> Path:
    """
    保存程序运行结果。

    pathlib.Path 会自动处理 Windows 和 Linux 路径差异。
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if preserve_existing and path.exists():
        path = _next_timestamped_output_path(path)

    path.write_text(text, encoding="utf-8")
    return path


def clear_directory_contents(
    directory_path: str,
    *,
    preserve_names: set[str] | None = None,
) -> Path:
    """
    清空目录内容，但保留目录本身。

    如果目录不存在，会先创建它。
    """

    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    preserved = preserve_names or set()
    for child in path.iterdir():
        if child.name in preserved:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    return path


def _next_timestamped_output_path(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    if not candidate.exists():
        return candidate

    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{timestamp}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1

