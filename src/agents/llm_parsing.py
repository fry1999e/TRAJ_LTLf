"""Validation helpers for JSON returned by LLM-backed agents."""

from __future__ import annotations

import json
import re
from typing import Optional

from temporal_language import operator_atoms_for_alias

AP_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ALLOWED_TRACE_KINDS = {"positive", "negative", "boundary"}
ALLOWED_COMPLETION_OPS = {"epsilon", "X", "F", "G", "XF"}
ATOMIC_COMPLETION_OPS = {"X", "F", "G"}


def require_mapping(value: object, context: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} 必须是 JSON object。")
    return value


def require_list(value: object, context: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{context} 必须是 JSON array。")
    return value


def coerce_list_field(
    mapping: dict,
    field: str,
    context: str | None = None,
    *,
    aliases: tuple[str, ...] = (),
    single_object_keys: tuple[str, ...] = (),
) -> list:
    """Return a top-level LLM list field, accepting common wrapper shapes."""

    context = context or field
    value = None
    found = False
    for key in (field, *aliases):
        if key in mapping:
            value = mapping[key]
            found = True
            break
    if not found and _looks_like_single_object(mapping, single_object_keys):
        return [mapping]
    if not found:
        return coerce_list_value(
            mapping,
            context,
            single_object_keys=single_object_keys,
        )
    return coerce_list_value(
        value,
        context,
        single_object_keys=single_object_keys,
    )


def coerce_list_value(
    value: object,
    context: str,
    *,
    single_object_keys: tuple[str, ...] = (),
) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = _parse_json_like_string(value)
        if parsed is not None:
            return coerce_list_value(
                parsed,
                context,
                single_object_keys=single_object_keys,
            )
    if isinstance(value, dict):
        if _looks_like_single_object(value, single_object_keys):
            return [value]
        for wrapper_key in (
            "items",
            "item",
            "result",
            "results",
            "data",
            "values",
            "list",
            "records",
            "output",
            "candidates",
            "candidate",
            "candidate_list",
            "templates",
            "template_candidates",
            "template_list",
            "hole_candidates",
            "hole_candidate",
            "hole_templates",
            "outputs",
        ):
            nested = value.get(wrapper_key)
            if isinstance(nested, list):
                return nested
            if isinstance(nested, (dict, str)):
                try:
                    return coerce_list_value(
                        nested,
                        context,
                        single_object_keys=single_object_keys,
                    )
                except ValueError:
                    pass
        for nested in value.values():
            if isinstance(nested, list):
                return nested
            if isinstance(nested, (dict, str)):
                try:
                    return coerce_list_value(
                        nested,
                        context,
                        single_object_keys=single_object_keys,
                    )
                except ValueError:
                    pass
        if value and all(isinstance(item, dict) for item in value.values()):
            return list(value.values())
    raise ValueError(
        f"{context} 必须是 JSON array。收到 {_describe_json_shape(value)}。"
    )


def _parse_json_like_string(value: str) -> object | None:
    text = value.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _describe_json_shape(value: object) -> str:
    if isinstance(value, dict):
        keys = ", ".join(str(key) for key in list(value)[:8])
        suffix = ", ..." if len(value) > 8 else ""
        return f"JSON object(keys=[{keys}{suffix}])"
    if isinstance(value, list):
        return "JSON array"
    return type(value).__name__


def _looks_like_single_object(
    mapping: dict,
    keys: tuple[str, ...],
) -> bool:
    return bool(keys) and any(key in mapping for key in keys)


def require_string(mapping: dict, field: str, context: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{field} 必须是非空字符串。")
    return value.strip()


def optional_string(mapping: dict, field: str, default: str = "") -> str:
    value = mapping.get(field, default)
    if value is None:
        return default
    if not isinstance(value, str):
        return default
    return value.strip()


def require_ap_name(value: str, context: str) -> str:
    value = value.strip()
    if not AP_PATTERN.fullmatch(value):
        raise ValueError(
            f"{context} 必须是合法 AP 名称，只能使用英文字母、数字和下划线，"
            "且不能以数字开头。"
        )
    return value


def coerce_bool(value: object, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "满足", "通过"}:
            return True
        if normalized in {"false", "no", "n", "0", "不满足", "失败"}:
            return False
    raise ValueError(f"{context} 必须是布尔值。")


def coerce_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def normalize_operator(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} 必须是字符串算子。")
    atoms = _completion_operator_atoms(value)
    if atoms is None:
        raise ValueError(f"{context} 包含不支持的算子: {value}")
    if not atoms:
        return "epsilon"
    if atoms == ("X", "F"):
        return "XF"
    if len(atoms) == 1:
        return atoms[0]
    return " ".join(atoms)


def normalize_operator_list(value: object, context: str) -> tuple[str, ...]:
    operators = require_list(value, context)
    normalized = tuple(normalize_operator(operator, context) for operator in operators)
    if not normalized:
        raise ValueError(f"{context} 至少需要一个允许算子。")
    for operator in normalized:
        if operator == "epsilon":
            continue
        atoms = completion_operator_atoms(operator)
        if not atoms or any(atom not in ATOMIC_COMPLETION_OPS for atom in atoms):
            raise ValueError(f"{context} 包含不支持的算子: {operator}")
    return normalized


def completion_operator_atoms(operator: str) -> tuple[str, ...]:
    atoms = _completion_operator_atoms(operator)
    if atoms is None:
        raise ValueError(f"operator 包含不支持的算子: {operator}")
    return atoms


def _completion_operator_atoms(value: str) -> Optional[tuple[str, ...]]:
    text = value.strip()
    compact = re.sub(r"[\s_-]+", "", text).upper()
    if compact in {"", "NONE", "EPSILON", "IDENTITY", "NOOP", "NOOPERATOR"}:
        return ()

    grounded = operator_atoms_for_alias(text)
    if grounded is not None:
        return grounded
    if re.fullmatch(r"[XFG]+", compact):
        return tuple(compact)

    atoms: list[str] = []
    parts = [part for part in re.split(r"[\s,;+/]+", text) if part.strip()]
    if not parts:
        return ()
    for part in parts:
        normalized = re.sub(r"[_-]+", "", part.strip()).upper()
        part_atoms = operator_atoms_for_alias(part)
        if part_atoms is not None and len(part_atoms) == 1:
            atoms.extend(part_atoms)
            continue
        if re.fullmatch(r"[XFG]+", normalized):
            atoms.extend(normalized)
            continue
        return None
    return tuple(atoms)
