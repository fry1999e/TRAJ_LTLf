"""Versioned Markdown loader for offline human surface segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from pathlib import Path
from typing import Mapping

from schemas import ArtifactGenerator, SurfaceSegmentation
from surface_segmentation import DEFAULT_CUT_MARKER, build_surface_segmentation


DEFAULT_HUMAN_SEGMENTATION_FILE = "data/input/human_segmentation.md"
HUMAN_SEGMENTATION_FORMAT = "traj-ltlf-human-segmentation"
HUMAN_SEGMENTATION_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class HumanSegmentationLoadResult:
    segmentations: dict[str, SurfaceSegmentation]
    errors: dict[str, str]
    missing_requirement_ids: tuple[str, ...]


@dataclass(frozen=True)
class SegmentationComparisonReport:
    requirement_id: str
    llm_segmentation_id: str
    human_segmentation_id: str
    comparison_status: str
    boundary_relation: str
    llm_cut_offsets: tuple[int, ...]
    human_cut_offsets: tuple[int, ...]
    shared_offsets: tuple[int, ...]
    llm_only_offsets: tuple[int, ...]
    human_only_offsets: tuple[int, ...]
    review_recommended: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_human_segmentations_from_markdown(
    path: str | Path,
    *,
    expected_requirements: Mapping[str, str],
    marker: str = DEFAULT_CUT_MARKER,
    data_split: str = "test",
) -> HumanSegmentationLoadResult:
    """Load every valid item and isolate item-level preprocessing errors."""

    source_path = Path(path)
    if not source_path.exists():
        return HumanSegmentationLoadResult(
            segmentations={},
            errors={"__file__": f"未找到人工切分文件: {source_path}"},
            missing_requirement_ids=tuple(expected_requirements),
        )
    text = source_path.read_text(encoding="utf-8")
    declared_format = _read_field(text, "format")
    declared_version = _read_field(text, "schema_version")
    if declared_format != HUMAN_SEGMENTATION_FORMAT:
        raise ValueError(
            f"人工切分文件 format 必须为 {HUMAN_SEGMENTATION_FORMAT}。"
        )
    if declared_version != HUMAN_SEGMENTATION_SCHEMA_VERSION:
        raise ValueError(
            "不支持的人工切分 schema_version=" + (declared_version or "<missing>")
        )

    sections = _requirement_sections(text)
    segmentations: dict[str, SurfaceSegmentation] = {}
    errors: dict[str, str] = {}
    for requirement_id, section in sections.items():
        expected = expected_requirements.get(requirement_id)
        if expected is None:
            errors[requirement_id] = "人工切分引用了原文文件中不存在的 requirement_id。"
            continue
        try:
            source_text = _extract_fenced_block(
                section, "source_text:start", "source_text:end"
            )
            marked_text = _extract_fenced_block(
                section, "human_marked_text:start", "human_marked_text:end"
            )
            if source_text != expected:
                raise ValueError("Source Text 与当前需求原文不完全一致。")
            segmentations[requirement_id] = build_surface_segmentation(
                requirement_id=requirement_id,
                source_text=source_text,
                marked_text=marked_text,
                marker=marker,
                status="surface_reviewed",
                generator=ArtifactGenerator(
                    mode="human",
                    prompt_version="offline-human-presegmentation-v1",
                ),
                experiment_mode="manual",
                data_split=data_split,
                label_source="human",
                review_basis="offline_human_presegmentation",
            )
        except Exception as exc:
            errors[requirement_id] = str(exc)

    missing = tuple(
        requirement_id
        for requirement_id in expected_requirements
        if requirement_id not in segmentations and requirement_id not in errors
    )
    return HumanSegmentationLoadResult(segmentations, errors, missing)


def compare_segmentations(
    llm: SurfaceSegmentation,
    human: SurfaceSegmentation,
) -> SegmentationComparisonReport:
    if llm.requirement_id != human.requirement_id:
        raise ValueError("cannot compare segmentations from different requirements")
    if llm.source_hash != human.source_hash or llm.source_text != human.source_text:
        raise ValueError("cannot compare segmentations from different source text")
    llm_offsets = _cut_offsets(llm)
    human_offsets = _cut_offsets(human)
    llm_set = set(llm_offsets)
    human_set = set(human_offsets)
    if llm_set == human_set:
        relation = "exact_match"
    elif llm_set < human_set:
        relation = "human_is_finer"
    elif human_set < llm_set:
        relation = "llm_is_finer"
    else:
        relation = "cross_boundary_disagreement"
    return SegmentationComparisonReport(
        requirement_id=llm.requirement_id,
        llm_segmentation_id=llm.segmentation_id,
        human_segmentation_id=human.segmentation_id,
        comparison_status="same" if relation == "exact_match" else "different",
        boundary_relation=relation,
        llm_cut_offsets=llm_offsets,
        human_cut_offsets=human_offsets,
        shared_offsets=tuple(sorted(llm_set & human_set)),
        llm_only_offsets=tuple(sorted(llm_set - human_set)),
        human_only_offsets=tuple(sorted(human_set - llm_set)),
        review_recommended=relation != "exact_match",
    )


def _cut_offsets(segmentation: SurfaceSegmentation) -> tuple[int, ...]:
    return tuple(segment.end for segment in segmentation.segments[:-1])


def _read_field(text: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}\s*:\s*(.*?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _requirement_sections(text: str) -> dict[str, str]:
    matches = list(
        re.finditer(r"^##\s+([A-Za-z][A-Za-z0-9_-]*-\d+)\s*$", text, re.MULTILINE)
    )
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        requirement_id = match.group(1).upper()
        if requirement_id in sections:
            raise ValueError(f"人工切分文件包含重复 requirement_id: {requirement_id}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[requirement_id] = text[match.end() : end]
    return sections


def _extract_fenced_block(section: str, start_name: str, end_name: str) -> str:
    start_marker = f"<!-- {start_name} -->"
    end_marker = f"<!-- {end_name} -->"
    start = section.find(start_marker)
    end = section.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0:
        raise ValueError(f"缺少 {start_marker} / {end_marker}。")
    envelope = section[start + len(start_marker) : end]
    match = re.search(r"```(?:text)?\s*\r?\n([\s\S]*?)\r?\n```", envelope)
    if not match:
        raise ValueError(f"{start_name} 必须包含 fenced text。")
    return match.group(1).replace("\r\n", "\n")


__all__ = [
    "DEFAULT_HUMAN_SEGMENTATION_FILE",
    "HUMAN_SEGMENTATION_FORMAT",
    "HUMAN_SEGMENTATION_SCHEMA_VERSION",
    "HumanSegmentationLoadResult",
    "SegmentationComparisonReport",
    "compare_segmentations",
    "load_human_segmentations_from_markdown",
]
