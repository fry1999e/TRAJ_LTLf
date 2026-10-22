"""
有限轨迹展示格式化工具。

程序内部使用 list[set[str]] 表示有限轨迹。人工审核使用时间为行、
AP 为列的 true/false 表格，完整保留未发生事件的时间步。
以下事件列表格式继续供已有报告和调用方使用：

    t0: request

    t1:

    t2: grant

这个文件专门负责把“程序能算的轨迹”转换成“人能直接读懂的轨迹”。
"""

from __future__ import annotations

import unicodedata

from ltlf_checker import Trace


def format_trace_truth_table(trace: Trace, ap_names=()) -> str:
    """Render actual valuations, including false cells and empty steps."""
    columns = tuple(dict.fromkeys((*ap_names, *sorted({name for step in trace for name in step}))))
    rows = [["时刻", *columns], *[
        [f"t{index}", *("true" if name in step else "false" for name in columns)]
        for index, step in enumerate(trace)
    ]]

    def width(value):
        return sum(0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
                   for char in value)

    widths = [max(width(row[index]) for row in rows) for index in range(len(rows[0]))]

    def render(row):
        return " | ".join(value + " " * (size - width(value)) for value, size in zip(row, widths)).rstrip()

    return "\n".join([render(rows[0]), "-+-".join("-" * size for size in widths),
                       *(render(row) for row in rows[1:])])


def format_trace_for_beginner(trace: Trace) -> str:
    """
    把有限轨迹转换为适合零基础阅读的多行文本。

    规则：
      1. 每一个时间步单独显示一行。
      2. t0、t1、t2 表示时间顺序。
      3. 如果该时间步没有事件，冒号后面留空。
      4. 如果同一时间步有多个事件，用逗号分隔。
      5. 时间步之间空一行，效果接近用户上传图片中的格式。
    """

    lines: list[str] = []
    for index, step in enumerate(trace):
        atoms = ", ".join(sorted(step))
        if atoms:
            lines.append(f"t{index}: {atoms}")
        else:
            lines.append(f"t{index}:")
    return "\n\n".join(lines)


def trace_to_serializable(trace: Trace) -> list[list[str]]:
    """
    把 list[set[str]] 转换成可以写入 JSON 的 list[list[str]]。

    JSON 不认识 Python 的 set，所以保存评审记录前需要做这个转换。
    """

    return [sorted(step) for step in trace]
