"""
LTLf 公式的简单表示与解析工具。

这个文件不依赖任何第三方库。为了让零基础读者也能看懂，程序没有直接使用
复杂的语法树框架，而是用一个很小的 Formula 类表示公式。

例如：
    G(request -> F grant)

会被表示为：
    Formula("G", (Formula("implies", (...)),))
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class Formula:
    """
    一个 LTLf 公式节点。

    op 表示这个节点的类型：
      - "ap"       : 原子命题，例如 request
      - "not"      : 取反
      - "and"      : 合取
      - "or"       : 析取
      - "implies"  : 蕴含
      - "X"        : 下一步
      - "F"        : 最终
      - "G"        : 始终
      - "U"        : 直到

    args 保存子公式。比如 G(phi) 只有一个子公式，phi -> psi 有两个子公式。
    name 只在 op == "ap" 时使用，用来保存原子命题名字。
    """

    op: str
    args: tuple["Formula", ...] = ()
    name: str = ""

    def size(self) -> int:
        """
        计算公式大小。

        这里的大小不是字符串长度，而是公式树里节点的数量。它用于在候选公式
        得分相同的时候，优先选择更简洁的公式。
        """

        return 1 + sum(child.size() for child in self.args)

    def __str__(self) -> str:
        return formula_to_text(self)


def Atom(name: str) -> Formula:
    """创建一个原子命题公式，例如 Atom("request")。"""

    return Formula("ap", name=name)


def TRUE() -> Formula:
    """创建恒真公式。"""

    return Formula("true")


def FALSE() -> Formula:
    """创建恒假公式。"""

    return Formula("false")


def Not(phi: Formula) -> Formula:
    return Formula("not", (phi,))


def And(left: Formula, right: Formula) -> Formula:
    return Formula("and", (left, right))


def Or(left: Formula, right: Formula) -> Formula:
    return Formula("or", (left, right))


def Implies(left: Formula, right: Formula) -> Formula:
    return Formula("implies", (left, right))


def Iff(left: Formula, right: Formula) -> Formula:
    """创建双向蕴含，用两个普通蕴含展开，避免增加新的求值节点。"""

    return And(Implies(left, right), Implies(right, left))


def X(phi: Formula) -> Formula:
    return Formula("X", (phi,))


def F(phi: Formula) -> Formula:
    return Formula("F", (phi,))


def G(phi: Formula) -> Formula:
    return Formula("G", (phi,))


def U(left: Formula, right: Formula) -> Formula:
    return Formula("U", (left, right))


def simplify_boolean_identities(formula: Formula) -> Formula:
    """Remove Boolean neutral elements without changing temporal operators.

    Rewriting children is safe under every operator.  In particular, a
    standalone ``X true`` or ``true U p`` is retained because the constant
    can carry finite-trace meaning there.
    """

    if not formula.args:
        return formula
    children = tuple(simplify_boolean_identities(child) for child in formula.args)
    if formula.op == "and":
        if children[0].op == "true":
            return children[1]
        if children[1].op == "true":
            return children[0]
    if formula.op == "or":
        if children[0].op == "false":
            return children[1]
        if children[1].op == "false":
            return children[0]
    return Formula(formula.op, children, formula.name)


def formula_to_text(formula: Formula) -> str:
    """
    把公式树转回人能读懂的文本。

    这个函数主要用于输出结果，例如把内部结构显示成：
        G(request -> F grant)
    """

    if formula.op == "ap":
        return formula.name
    if formula.op == "true":
        return "true"
    if formula.op == "false":
        return "false"

    if formula.op == "not":
        child = formula.args[0]
        if child.op in {"and", "or", "implies", "U"}:
            return f"!({formula_to_text(child)})"
        return f"!{formula_to_text(child)}"

    if formula.op in {"X", "F", "G"}:
        child = formula.args[0]
        child_text = formula_to_text(child)
        if child.op in {"and", "or", "implies", "U"}:
            return f"{formula.op}({child_text})"
        return f"{formula.op} {child_text}"

    if formula.op in {"and", "or", "implies", "U"}:
        left, right = formula.args
        symbol = {
            "and": " & ",
            "or": " | ",
            "implies": " -> ",
            "U": " U ",
        }[formula.op]

        left_text = formula_to_text(left)
        right_text = formula_to_text(right)

        if _binary_left_needs_parentheses(formula.op, left.op):
            left_text = f"({left_text})"
        if right.op in {"and", "or", "implies", "U"}:
            right_text = f"({right_text})"

        return f"{left_text}{symbol}{right_text}"

    raise ValueError(f"未知公式节点类型: {formula.op}")


def _binary_left_needs_parentheses(parent_op: str, child_op: str) -> bool:
    """Keep printed formulas parseable to the same tree."""

    if child_op == "implies":
        return True
    if parent_op == "and" and child_op == "or":
        return True
    if parent_op == "U" and child_op in {"and", "or"}:
        return True
    return False


class FormulaParser:
    """
    一个很小的递归下降解析器。

    它支持的输入形式包括：
      - request
      - F grant
      - X F grant
      - G(request -> F grant)
      - safe U goal

    这个解析器用于演示和调试，不追求覆盖所有工业级 LTLf 语法。
    """

    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.index = 0

    def parse(self) -> Formula:
        formula = self._parse_iff()
        if self._peek() is not None:
            raise ValueError(f"公式末尾有无法解析的内容: {self._peek()}")
        return formula

    def _tokenize(self, text: str) -> list[str]:
        """
        把字符串切成 token。

        token 可以粗略理解为公式中的“单词或符号”，比如 G、request、->、(、)。
        """

        text = unicodedata.normalize('NFKC', text).replace("\\/", "|").replace("/\\", "&")
        pattern = r"\s*(<->|<=>|->|&&|\|\||[()!&|]|¬|∧|∨|→|↔|[^\W\d]\w*|.)"
        tokens = re.findall(pattern, text)
        return [token for token in tokens if token.strip()]

    def _peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("公式还没有写完，解析时已经到达末尾。")
        self.index += 1
        return token

    def _match_symbol(self, *symbols: str) -> bool:
        token = self._peek()
        if token in symbols:
            self.index += 1
            return True
        return False

    def _match_word(self, *words: str) -> bool:
        token = self._peek()
        if token is not None and token.lower() in words:
            self.index += 1
            return True
        return False

    def _parse_iff(self) -> Formula:
        formula = self._parse_implies()
        while self._match_symbol("<->", "<=>", "↔") or self._match_word("iff", "equiv"):
            formula = Iff(formula, self._parse_implies())
        return formula

    def _parse_implies(self) -> Formula:
        left = self._parse_or()
        if self._match_symbol("->", "→") or self._match_word("implies"):
            right = self._parse_implies()
            return Implies(left, right)
        return left

    def _parse_or(self) -> Formula:
        formula = self._parse_and()
        while self._match_symbol("|", "||", "∨") or self._match_word("or"):
            formula = Or(formula, self._parse_and())
        return formula

    def _parse_and(self) -> Formula:
        formula = self._parse_until()
        while self._match_symbol("&", "&&", "∧") or self._match_word("and"):
            formula = And(formula, self._parse_until())
        return formula

    def _parse_until(self) -> Formula:
        formula = self._parse_unary()
        while self._match_word("u", "until"):
            formula = U(formula, self._parse_unary())
        return formula

    def _parse_unary(self) -> Formula:
        if self._match_symbol("!", "¬") or self._match_word("not"):
            return Not(self._parse_unary())

        token = self._peek()
        if token is not None:
            temporal_aliases = {
                "next": "X",
                "eventually": "F",
                "finally": "F",
                "always": "G",
                "globally": "G",
            }
            alias_operator = temporal_aliases.get(token.lower())
            if alias_operator is not None:
                self.index += 1
                child = self._parse_unary()
                if alias_operator == "X":
                    return X(child)
                if alias_operator == "F":
                    return F(child)
                return G(child)

        if token is not None and token.upper() in {"X", "F", "G"}:
            self.index += 1
            operator = token.upper()
            child = self._parse_unary()
            if operator == "X":
                return X(child)
            if operator == "F":
                return F(child)
            return G(child)

        return self._parse_atom()

    def _parse_atom(self) -> Formula:
        if self._match_symbol("("):
            formula = self._parse_implies()
            if not self._match_symbol(")"):
                raise ValueError("缺少右括号 )")
            return formula

        token = self._take()
        lower_token = token.lower()
        if lower_token == "true":
            return TRUE()
        if lower_token == "false":
            return FALSE()
        if re.fullmatch(r"[^\W\d]\w*", token):
            return Atom(token)

        raise ValueError(f"无法识别的公式 token: {token}")


def parse_formula(text: str) -> Formula:
    """把公式文本解析成 Formula 对象。"""

    return FormulaParser(text).parse()
