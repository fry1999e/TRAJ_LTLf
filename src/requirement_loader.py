"""
Load natural-language requirements from XLSX or Markdown files.

The batch entrypoint treats each Markdown bullet or numbered item as one
requirement. Headings, blank lines, and fenced code blocks are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path

from ap_identity import (
    controlled_symbolic_profile,
    create_declared_ap_catalog,
    declared_ap_catalog_from_mapping,
    declared_catalog_profile,
    natural_language_profile,
    offline_formula_assisted_profile,
    undetermined_profile,
)
from schemas import APRepresentationProfile, DeclaredAPCatalog
from semantic_source import EffectiveSemanticSource, natural_language_source


DEFAULT_REQUIREMENTS_FILE = "data/input/example.xlsx"


@dataclass(frozen=True)
class RequirementRecord:
    """One stable requirement loaded from XLSX or Markdown."""

    requirement_id: str
    text: str
    ap_representation_profile: APRepresentationProfile = field(
        default_factory=undetermined_profile
    )
    declared_ap_catalog: DeclaredAPCatalog | None = None
    # Programmatic callers predating runtime remain a frozen v1 compatibility
    # surface. File/CLI loaders set this true, including untagged sources whose
    # explicit v2 profile is ``undetermined``.
    ap_profile_declared: bool = field(default=False, compare=False)
    effective_semantic_source: EffectiveSemanticSource | None = field(
        default=None,
        compare=False,
    )

    def __post_init__(self) -> None:
        source = self.effective_semantic_source
        if source is None:
            source = natural_language_source(
                requirement_id=self.requirement_id,
                source_text=self.text,
            )
            object.__setattr__(self, "effective_semantic_source", source)
        if (
            source.requirement_id != self.requirement_id
            or source.source_text != self.text
        ):
            raise ValueError(
                "RequirementRecord effective semantic source lineage mismatch"
            )


def load_requirement_records(path: str | Path = DEFAULT_REQUIREMENTS_FILE) -> list[RequirementRecord]:
    """Dispatch by file type; XLSX inputs carry text and stable IDs only."""
    if Path(path).suffix.lower() == ".xlsx":
        from dataset_workbook import normalized_requirement_id, read_input_workbook

        return [
            RequirementRecord(normalized_requirement_id(source_id), text, ap_profile_declared=True)
            for source_id, text in read_input_workbook(path)
        ]
    if Path(path).suffix.lower() != ".md":
        raise ValueError("需求文件须为 .xlsx 或 .md")
    return load_requirement_records_from_markdown(path)


def load_requirements(path: str | Path = DEFAULT_REQUIREMENTS_FILE) -> list[str]:
    return [record.text for record in load_requirement_records(path)]


def load_requirements_from_markdown(path: str | Path = DEFAULT_REQUIREMENTS_FILE) -> list[str]:
    """Return all requirements listed as Markdown bullets or numbered items."""

    return [record.text for record in load_requirement_records_from_markdown(path)]


def load_requirement_records_from_markdown(
    path: str | Path = DEFAULT_REQUIREMENTS_FILE,
) -> list[RequirementRecord]:
    """Load stable ``## REQ-*`` records, with legacy list fallback.

    The explicit format keeps IDs stable when requirements are inserted or
    reordered.  Legacy Markdown lists remain readable and receive the historic
    ``REQ-001`` sequence.
    """

    if Path(path).suffix.lower() == ".xlsx":
        return load_requirement_records(path)
    markdown_path = Path(path)
    if not markdown_path.exists():
        raise FileNotFoundError(
            f"未找到需求文件: {markdown_path}. 请创建该文件，"
            "并用 Markdown 列表写入待测试的自然语言需求。"
        )

    text = markdown_path.read_text(encoding="utf-8")
    explicit = _load_explicit_requirement_records(
        text, base_dir=markdown_path.parent
    )
    if explicit:
        return explicit

    requirements: list[str] = []
    in_code_block = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not line:
            continue

        bullet_match = re.match(r"^(?:[-*+]|\d+[.)])\s+(.*)$", line)
        if not bullet_match:
            continue

        item = re.sub(r"^\[[ xX]\]\s+", "", bullet_match.group(1).strip())
        if item:
            requirements.append(item)

    if not requirements:
        raise ValueError(
            f"需求文件 {markdown_path} 中没有有效需求。"
            "请使用 '- requirement text' 或 '1. requirement text' 格式。"
        )
    return [
        RequirementRecord(
            f"REQ-{index:03d}",
            requirement,
            ap_representation_profile=undetermined_profile(),
            ap_profile_declared=True,
        )
        for index, requirement in enumerate(requirements, start=1)
    ]


def _load_explicit_requirement_records(
    text: str,
    *,
    base_dir: Path | None = None,
) -> list[RequirementRecord]:
    default_profile_match = re.search(
        r"^default_ap_profile\s*:\s*([A-Za-z0-9_-]+)\s*$",
        text,
        re.MULTILINE,
    )
    default_profile_mode = (
        default_profile_match.group(1) if default_profile_match else "undetermined"
    )
    default_grammar_match = re.search(
        r"^default_ap_grammar\s*:\s*([A-Za-z0-9_-]+)\s*$",
        text,
        re.MULTILINE,
    )
    default_ap_grammar = (
        default_grammar_match.group(1) if default_grammar_match else ""
    )
    lines = text.splitlines()
    records: list[RequirementRecord] = []
    seen: set[str] = set()
    index = 0
    in_fence = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence:
            index += 1
            continue
        match = re.match(r"^##\s+([A-Za-z][A-Za-z0-9_-]*-\d+)\s*$", line)
        if not match:
            index += 1
            continue
        requirement_id = match.group(1).upper()
        if requirement_id in seen:
            raise ValueError(f"需求文件包含重复 requirement_id: {requirement_id}")
        seen.add(requirement_id)
        index += 1
        metadata_lines: list[str] = []
        while index < len(lines) and not lines[index].strip().startswith("```"):
            if re.match(r"^##\s+", lines[index]):
                raise ValueError(f"{requirement_id} 缺少 fenced text 原文。")
            metadata_lines.append(lines[index])
            index += 1
        if index >= len(lines):
            raise ValueError(f"{requirement_id} 缺少 fenced text 原文。")
        index += 1
        content: list[str] = []
        while index < len(lines) and not lines[index].strip().startswith("```"):
            content.append(lines[index])
            index += 1
        if index >= len(lines):
            raise ValueError(f"{requirement_id} 的 fenced text 未闭合。")
        value = "\n".join(content)
        if not value.strip():
            raise ValueError(f"{requirement_id} 的原文不能为空。")
        profile, catalog = _ap_metadata(
            metadata_lines,
            requirement_id=requirement_id,
            default_profile_mode=default_profile_mode,
            default_ap_grammar=default_ap_grammar,
            base_dir=base_dir,
        )
        records.append(
            RequirementRecord(
                requirement_id,
                value,
                ap_representation_profile=profile,
                declared_ap_catalog=catalog,
                ap_profile_declared=True,
            )
        )
        index += 1
    return records


def _ap_metadata(
    lines: list[str],
    *,
    requirement_id: str,
    default_profile_mode: str = "undetermined",
    default_ap_grammar: str = "",
    base_dir: Path | None = None,
) -> tuple[APRepresentationProfile, DeclaredAPCatalog | None]:
    fields: dict[str, str] = {}
    for raw in lines:
        match = re.match(
            r"^\s*(ap_profile|ap_profile_id|ap_grammar|ap_language|ap_domain|ap_catalog_json|ap_catalog_file)\s*:\s*(.*?)\s*$",
            raw,
        )
        if match:
            fields[match.group(1)] = match.group(2)
    mode = fields.get("ap_profile", default_profile_mode).strip()
    profile_id = fields.get("ap_profile_id", "").strip()
    language = fields.get("ap_language", "").strip()
    domain = fields.get("ap_domain", "").strip()
    symbolic_grammar_id = fields.get(
        "ap_grammar",
        default_ap_grammar,
    ).strip()
    provenance = (
        f"requirement_metadata:{requirement_id}"
        if "ap_profile" in fields
        else (
            "requirement_file_default_profile"
            if default_profile_mode != "undetermined"
            else "implicit_untagged"
        )
    )
    catalog: DeclaredAPCatalog | None = None
    if mode == "controlled_symbolic":
        profile = controlled_symbolic_profile(
            profile_id=profile_id or f"AP-PROFILE-{requirement_id}-symbolic-v1",
            provenance=provenance,
            language=language or "symbolic",
            domain=domain or "synthetic",
            symbolic_grammar_id=(
                symbolic_grammar_id or "single_lower_letter_v1"
            ),
        )
    elif mode == "natural_language_grounded":
        profile = natural_language_profile(
            profile_id=profile_id or f"AP-PROFILE-{requirement_id}-natural-v1",
            provenance=provenance,
            language=language,
            domain=domain,
        )
    elif mode == "declared_catalog":
        raw_catalog = fields.get("ap_catalog_json", "")
        catalog_file = fields.get("ap_catalog_file", "").strip()
        if raw_catalog and catalog_file:
            raise ValueError(
                f"{requirement_id} 不能同时声明 ap_catalog_json 和 ap_catalog_file。"
            )
        if catalog_file:
            catalog_path = Path(catalog_file)
            if not catalog_path.is_absolute():
                catalog_path = (base_dir or Path.cwd()) / catalog_path
            try:
                raw_catalog = catalog_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(
                    f"{requirement_id} 无法读取 ap_catalog_file: {catalog_path}"
                ) from exc
        if not raw_catalog:
            raise ValueError(
                f"{requirement_id} declared_catalog 缺少 ap_catalog_json/ap_catalog_file。"
            )
        try:
            payload = json.loads(raw_catalog)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{requirement_id} ap_catalog_json 不是有效 JSON。"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{requirement_id} ap_catalog_json 必须是对象。")
        if payload.get("catalog_hash"):
            catalog = declared_ap_catalog_from_mapping(payload)
        else:
            catalog = create_declared_ap_catalog(
                catalog_id=str(payload.get("catalog_id", "")),
                catalog_version=str(payload.get("catalog_version", "")),
                entries=payload.get("entries", ()),
                provenance=str(payload.get("provenance", provenance)),
                source=str(payload.get("source", "declared_catalog")),
            )
        profile = declared_catalog_profile(
            catalog,
            profile_id=profile_id,
            provenance=provenance,
            language=language,
            domain=domain,
        )
    elif mode == "offline_formula_assisted":
        profile = offline_formula_assisted_profile(
            profile_id=(
                profile_id or f"AP-PROFILE-{requirement_id}-offline-formula-v1"
            ),
            provenance=provenance,
            language=language,
            domain=domain,
        )
    elif mode == "undetermined":
        profile = undetermined_profile(
            profile_id=(
                profile_id
                or (
                    f"AP-PROFILE-{requirement_id}-undetermined-v1"
                    if "ap_profile" in fields or default_profile_mode != "undetermined"
                    else "AP-PROFILE-undetermined-v1"
                )
            ),
            provenance=provenance,
            language=language,
            domain=domain,
        )
    else:
        raise ValueError(f"{requirement_id} 不支持 ap_profile={mode}。")
    return profile, catalog


__all__ = [
    "DEFAULT_REQUIREMENTS_FILE",
    "RequirementRecord",
    "load_requirement_records",
    "load_requirements",
    "load_requirement_records_from_markdown",
    "load_requirements_from_markdown",
]
