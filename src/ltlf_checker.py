"""
LTLf 有限轨迹语义检查器。

这个文件是 Traj-LTLf 原型程序的核心：它判断一条有限轨迹是否满足某个
LTLf 公式。

有限轨迹示例：
    [
        {"request"},
        set(),
        {"grant"}
    ]

含义：
    t0 时刻 request 为真
    t1 时刻没有原子命题为真
    t2 时刻 grant 为真
"""

from __future__ import annotations

from ltlf_parser import Formula


Trace = list[set[str]]


def check_trace(trace: Trace, formula: Formula) -> bool:
    """
    判断整条有限轨迹是否满足公式。

    LTLf 通常从第 0 个位置开始判断整条轨迹，所以这里固定从 position=0
    开始调用 satisfies。
    """

    if not trace:
        raise ValueError("本演示程序暂不处理空轨迹，请至少提供一个时间步。")
    return satisfies(trace, formula, position=0)


def satisfies(trace: Trace, formula: Formula, position: int = 0) -> bool:
    """
    判断 trace 在 position 位置是否满足 formula。

    参数解释：
      - trace: 有限轨迹，也就是若干个时间步组成的列表
      - formula: 要检查的 LTLf 公式
      - position: 当前检查的是第几个时间步

    注意：
      在 LTLf 中，轨迹是有限的。因此 X phi 在最后一个时间步一定为假，
      因为最后一步后面没有“下一步”。
    """

    op = formula.op

    if op == "true":
        return True
    if op == "false":
        return False

    # 如果当前位置已经超过轨迹长度，普通公式都无法继续被满足。
    if position >= len(trace):
        return False

    if op == "ap":
        # 原子命题 p 成立，当且仅当 p 出现在当前时间步的集合中。
        return formula.name in trace[position]

    if op == "not":
        return not satisfies(trace, formula.args[0], position)

    if op == "and":
        left, right = formula.args
        return satisfies(trace, left, position) and satisfies(trace, right, position)

    if op == "or":
        left, right = formula.args
        return satisfies(trace, left, position) or satisfies(trace, right, position)

    if op == "implies":
        left, right = formula.args
        # A -> B 等价于：A 不成立，或者 B 成立。
        return (not satisfies(trace, left, position)) or satisfies(trace, right, position)

    if op == "X":
        # 强 next：必须真的存在下一步，并且下一步满足子公式。
        next_position = position + 1
        return next_position < len(trace) and satisfies(trace, formula.args[0], next_position)

    if op == "F":
        # F phi 表示 eventually phi：
        # 从当前位置开始，一直到有限轨迹结束，只要有一个位置满足 phi 就可以。
        child = formula.args[0]
        for next_position in range(position, len(trace)):
            if satisfies(trace, child, next_position):
                return True
        return False

    if op == "G":
        # G phi 表示 globally phi：
        # 从当前位置开始，一直到有限轨迹结束，每一个位置都必须满足 phi。
        child = formula.args[0]
        for next_position in range(position, len(trace)):
            if not satisfies(trace, child, next_position):
                return False
        return True

    if op == "U":
        # phi U psi 表示 phi 一直保持，直到某个位置 psi 成立。
        # 需要找到一个 j，使得 j 位置满足 psi，并且 position 到 j-1 都满足 phi。
        left, right = formula.args
        for target_position in range(position, len(trace)):
            right_holds = satisfies(trace, right, target_position)
            left_holds_before = all(
                satisfies(trace, left, before_position)
                for before_position in range(position, target_position)
            )
            if right_holds and left_holds_before:
                return True
        return False

    raise ValueError(f"未知公式节点类型: {op}")
