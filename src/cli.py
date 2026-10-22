"""
Traj-LTLf 原型程序入口。

运行方式：
    python main.py

人工审核：
    python main.py --review manual --experiment-mode human --llm api \
        --segmentation-review manual --plain-reading-review manual

默认配置来自 config/manual.json。人工只审核切分、AP 和轨迹是否符合原文。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import sys

from boundary_trace_fixture import (
    BoundaryTraceFixtureError,
    load_boundary_trace_fixture,
)

from batch_pipeline import (
    BatchPipelineItemResult,
    batch_classification_counts,
    batch_unique_result_counts,
    format_final_unique_batch_results,
    format_final_unique_batch_summary,
    format_batch_pipeline_results,
    format_batch_pipeline_summary,
    run_batch_traj_ltlf_pipeline,
    write_classified_batch_outputs,
)
from checkpointing import (
    DEFAULT_CHECKPOINT_DIR,
    migrate_v1_pending_semantic_overrides,
    prepare_checkpoint_run,
)
from artifact_audit import audit_ap_identity_artifacts
from ap_preparation import APPreparationError
from source_graph_resolution import SourceGraphResolutionRequired
from contract_versions import faithful_ir_runtime_contract
from experiment_modes import (
    implicit_auto_human_warning,
    is_sample_reference_formula,
    resolve_experiment_mode,
)
from llm import LLMClientError
from pipeline import (
    build_degraded_result_record,
    format_pipeline_result,
    format_pipeline_summary,
    run_traj_ltlf_pipeline,
)
from pre_ir_review import requires_interactive_pre_ir_review
from plain_reading import PlainReadingSemanticResolutionRequired
from pre_ir_batch_review import (
    PRE_IR_BATCHES_DIR,
    PreIRStateConsistencyError,
    ReviewFileInputError,
    _checkpoint_fingerprint,
    apply_ambiguity_review,
    apply_no_ambiguity_review,
    apply_semantic_fallback_review,
    apply_semantic_override_ap_inventory_review,
    apply_semantic_override_ap_mapping_review,
    apply_pending_replacement_review,
    ensure_deferred_pre_ir_batch,
    export_pending_replacement_review,
    mark_pre_ir_downstream_started,
    reopen_pre_ir_item,
    review_pending_replacement_pre_ir,
)
from rag.persistent_memory import DEFAULT_APPROVED_MEMORY_FILE
from human_segmentation_loader import DEFAULT_HUMAN_SEGMENTATION_FILE
from requirement_loader import (
    DEFAULT_REQUIREMENTS_FILE,
    RequirementRecord,
    load_requirement_records,
    load_requirements,
)
from schemas import TraceEvidencePolicy, WarmupLearningConfig
from terminal_progress import create_progress, progress_session
from utils import clear_directory_contents, write_result_file
from warmup_loader import load_warmup_examples_from_markdown
from runtime_config import DEFAULT_CONFIG, load_runtime_config


RESULT_OUTPUT_FILE = "data/output/manual/outputs/result.txt"
SUMMARY_OUTPUT_FILE = "data/output/manual/outputs/final_candidates.txt"
REVIEW_RECORDS_DIR = "data/output/manual/review_records"
ARTIFACT_DIRS_TO_CLEAR = ("data/output/manual/outputs", REVIEW_RECORDS_DIR)

EXAMPLE_ARTIFACT_DIR = "data/output/warmup"
EXAMPLE_RESULT_OUTPUT_FILE = f"{EXAMPLE_ARTIFACT_DIR}/Example_result.txt"
EXAMPLE_SUMMARY_OUTPUT_FILE = f"{EXAMPLE_ARTIFACT_DIR}/Example_final_candidates.txt"
EXAMPLE_REVIEW_RECORDS_DIR = f"{EXAMPLE_ARTIFACT_DIR}/Example_review_records"


@progress_session()
def main() -> None:
    """
    程序入口函数。

    初学者可以把它理解为：Python 从这里开始执行整个 Traj-LTLf 流程。
    """

    # Windows 终端有时会默认使用非 UTF-8 编码。这里显式设置一下，方便显示中文。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_args, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(description="运行 Traj-LTLf 人工审核流程")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="运行默认配置；命令行参数优先。")
    parser.add_argument("--delivery-policy", choices=["strict"], default="strict",
                        help="人工审核完成后才交付正式结果。")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="隔离本次输出、审核记录和断点，避免覆盖其他运行。")
    parser.add_argument(
        "requirement",
        nargs="*",
        help="自然语言需求，例如：Whenever request occurs, grant should eventually occur.",
    )
    parser.add_argument(
        "--review",
        choices=["manual"],
        default="manual",
        help="用户审核表面切分、AP 含义和轨迹；语义结构及技术字段由主模型处理。",
    )
    parser.add_argument(
        "--llm",
        choices=["rules", "api"],
        default=None,
        help="Agent 生成模式。rules 使用本地规则；api 调用真实 LLM API。",
    )
    parser.add_argument(
        "--experiment-mode",
        choices=["human"],
        default="human",
        help="由用户判断轨迹是否符合原文，主模型整理标签、置信度和理由。",
    )
    parser.add_argument(
        "--learning-mode",
        choices=[
            "stage_guided",
            "offline_formula_validation",
        ],
        default=None,
        help=(
            "warmup/sample 学习模式。默认 stage_guided。"
        ),
    )
    parser.add_argument(
        "--offline-validation",
        choices=["none", "final_formula"],
        default="none",
        help="warmup-only 离线验算来源；final_formula 允许读取样本最终公式做校验。",
    )
    parser.add_argument(
        "--data-split",
        choices=["train", "validation", "test"],
        default="test",
        help="当前需求所属数据划分；test 结果默认不写入持久 RAG。",
    )
    parser.set_defaults(reference_formula=None, boundary_traces_file=None)
    parser.add_argument(
        "--max-review-rounds",
        type=int,
        default=3,
        help="语义结构或轨迹评审不通过时，Agent 最多重新生成的轮数。",
    )
    parser.add_argument(
        "--boundary-trace-policy",
        choices=["generated", "disabled"],
        default="generated",
        help=(
            "边界轨迹证据来源。generated 在线生成后由用户审核；disabled 用于无边界消融。"
        ),
    )
    parser.add_argument(
        "--segmentation-review",
        choices=["manual"],
        default="manual",
        help="表面切分由用户审核。",
    )
    parser.add_argument(
        "--plain-reading-review",
        choices=["manual"],
        default="manual",
        help=(
            "保留 manual 命令兼容性；普通读法和语义结构由主模型检查，用户无需填写专业表单。"
        ),
    )
    parser.add_argument(
        "--max-segmentation-retries",
        type=int,
        default=2,
        help="切分字符重构失败后的自动机械修复重试次数。",
    )
    parser.add_argument(
        "--max-surface-review-rounds",
        type=int,
        default=3,
        help="表面边界审核的独立最大轮数。",
    )
    parser.add_argument(
        "--max-plain-reading-rounds",
        type=int,
        default=2,
        help="普通语言解释审核的独立最大轮数。",
    )
    parser.add_argument(
        "--max-plain-reading-revision-rounds",
        type=int,
        default=2,
        help="人工补充或修订普通读法的独立最大轮数。",
    )
    parser.add_argument(
        "--segmentation-prompt-version",
        default="plain-segmentation-v1",
        help="切分 Agent prompt 版本，写入 artifact/checkpoint 审计。",
    )
    parser.add_argument(
        "--plain-reading-prompt-version",
        default="plain-reading-v5",
        help="普通读法 Agent prompt 版本，写入 artifact/checkpoint 审计。",
    )
    parser.add_argument(
        "--plain-reading-semantic-fallback",
        choices=["enabled", "disabled"],
        default="enabled",
        help=(
            "runtime 灰度开关：enabled 时两次 artifact 后的纯语义失败转普通审核者；"
            "disabled 时保持技术阻断。"
        ),
    )
    parser.add_argument(
        "--requirements-file",
        default=DEFAULT_REQUIREMENTS_FILE,
        help="未在命令行传入需求时，从该 XLSX 或 Markdown 文件读取批量需求。",
    )
    parser.add_argument(
        "--human-segmentation-file",
        default=DEFAULT_HUMAN_SEGMENTATION_FILE,
        help="大批量延迟 Pre-IR 审核使用的人工预切分 Markdown 文件。",
    )
    parser.add_argument(
        "--pre-ir-mode", choices=["interactive", "deferred"], default="interactive",
        help="默认逐项交互审核；deferred 显式使用人工预切分文件和批量审核文档。",
    )
    parser.add_argument(
        "--warmup-size",
        type=int,
        default=0,
        help="批量模式下先人工学习的正确小样本数量，0 表示不分热身批次。",
    )
    parser.add_argument(
        "--warmup-file",
        default=None,
        help="独立小样本热身 Markdown 文件；通过后会作为完整正确过程写入 RAG。",
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="只运行 warmup 文件，不读取主体需求文件。",
    )
    parser.add_argument(
        "--rag-memory-file",
        default=DEFAULT_APPROVED_MEMORY_FILE,
        help=(
            "持久化 RAG 记忆文件。默认会自动加载并写回；"
            "传空字符串可关闭持久化记忆。"
        ),
    )
    parser.add_argument(
        "--progress",
        choices=["auto", "plain", "off"],
        default=None,
        help=(
            "终端进度显示方式。默认 auto：交互终端显示转动标识，"
            "重定向输出时使用文本提示；plain 强制文本提示；off 关闭。"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="批量 API 运行的断点目录；传空字符串可关闭断点续跑。",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "清空断点、outputs及普通review记录后从头运行；Pre-IR仅清理当前"
            "checkpoint的namespace，保留共享records目录中的其他批次。"
        ),
    )
    parser.add_argument(
        "--review-feedback-file",
        default=None,
        help=(
            "读取结构化 review_feedback.json，并按每项 restart_stage 从最近安全阶段重跑。"
        ),
    )
    parser.add_argument(
        "--require-complete-batch",
        action="store_true",
        help="部分结果落盘后，若仍有需求没有最终结果则以非零状态退出。",
    )
    parser.add_argument(
        "--apply-no-ambiguity-review",
        default=None,
        metavar="FILE",
        help="应用‘LLM 未发现歧义’批量抽查 Markdown；应用后正常退出。",
    )
    parser.add_argument(
        "--apply-ambiguity-review",
        default=None,
        metavar="FILE",
        help="应用统一歧义裁决 Markdown；若产生下一轮会明确提示并正常退出。",
    )
    parser.add_argument(
        "--apply-semantic-fallback-review",
        default=None,
        metavar="FILE",
        help="应用两次artifact后无安全歧义轴的Y/R/C审核或replacement确认。",
    )
    parser.add_argument(
        "--apply-semantic-override-ap-inventory-review",
        default=None,
        metavar="FILE",
        help="应用semantic override两侧AP inventory人工审核Markdown。",
    )
    parser.add_argument(
        "--apply-semantic-override-ap-mapping-review",
        default=None,
        metavar="FILE",
        help="应用semantic override AP identity mapping人工审核Markdown。",
    )
    parser.add_argument(
        "--reopen-pre-ir",
        default=None,
        metavar="REQ_ID",
        help=(
            "将一个稳定 ID 重新带回交互式切分和歧义处理；下游启动后"
            "仅允许重开 deferred 且没有 Semantic IR 后代的项目。"
        ),
    )
    parser.add_argument(
        "--review-pending-replacements",
        action="store_true",
        help=(
            "连续处理checkpoint中全部已激活且等待surface gate的replacement；"
            "可单独用于中断续跑，或与--apply-semantic-fallback-review组合。"
        ),
    )
    parser.add_argument(
        "--export-pending-replacement-review",
        action="store_true",
        help=(
            "为全部已激活replacement生成独立的批量Pre-IR Markdown；"
            "只写draft，不批准项目或启动下游。"
        ),
    )
    parser.add_argument(
        "--apply-pending-replacement-review",
        default=None,
        metavar="FILE",
        help="全文件预检后逐项事务化应用replacement Pre-IR Markdown。",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="仅与--export-pending-replacement-review配合，强制重新生成草稿。",
    )
    parser.add_argument(
        "--audit-ap-identity",
        nargs="?",
        const="-",
        default=None,
        metavar="OUTPUT_JSON",
        help=(
            "只读审计AP inventory/comparison/override lineage；省略输出路径时打印JSON。"
        ),
    )
    parser.add_argument(
        "--migrate-v1-semantic-overrides",
        choices=["dry-run", "commit"],
        default=None,
        help="显式检查或迁移v1 extraction-unknown pending override；不会自动激活。",
    )
    try:
        parser.set_defaults(**load_runtime_config(config_args.config))
    except (OSError, ValueError) as exc:
        parser.error(f"配置读取失败: {exc}")
    args = parser.parse_args()
    if args.run_dir is not None:
        args.checkpoint_dir = str(args.run_dir / "checkpoints")


    if args.max_segmentation_retries < 0:
        parser.error("--max-segmentation-retries 不能小于 0")
    if args.max_surface_review_rounds < 1:
        parser.error("--max-surface-review-rounds 必须至少为 1")
    if args.max_plain_reading_rounds < 1:
        parser.error("--max-plain-reading-rounds 必须至少为 1")
    if args.max_plain_reading_revision_rounds < 1:
        parser.error("--max-plain-reading-revision-rounds 必须至少为 1")

    apply_review_commands = [
        value
        for value in (
            args.apply_no_ambiguity_review,
            args.apply_semantic_fallback_review,
            args.apply_semantic_override_ap_inventory_review,
            args.apply_semantic_override_ap_mapping_review,
            args.apply_ambiguity_review,
            args.apply_pending_replacement_review,
        )
        if value
    ]
    if len(apply_review_commands) > 1:
        parser.error("--apply-*-review 一次只能使用一种")
    maintenance_commands = [
        value
        for value in (
            args.audit_ap_identity,
            args.migrate_v1_semantic_overrides,
        )
        if value
    ]
    if len(maintenance_commands) > 1:
        parser.error("--audit-ap-identity 与 --migrate-v1-semantic-overrides 不能同时使用")
    if maintenance_commands and (
        apply_review_commands
        or args.reopen_pre_ir
        or args.review_pending_replacements
        or args.export_pending_replacement_review
    ):
        parser.error("AP审计/迁移命令不能与Pre-IR审核命令同时使用")
    if args.reopen_pre_ir and apply_review_commands:
        parser.error("--apply-*-review 与 --reopen-pre-ir 一次只能使用一种")
    if args.reopen_pre_ir and args.review_pending_replacements:
        parser.error(
            "--reopen-pre-ir 与 --review-pending-replacements 一次只能使用一种"
        )
    if args.export_pending_replacement_review and (
        apply_review_commands
        or args.reopen_pre_ir
        or args.review_pending_replacements
        or maintenance_commands
    ):
        parser.error(
            "--export-pending-replacement-review 必须作为独立审核命令运行"
        )
    if args.review_pending_replacements and (
        args.apply_no_ambiguity_review
        or args.apply_ambiguity_review
        or args.apply_semantic_override_ap_inventory_review
        or args.apply_semantic_override_ap_mapping_review
        or args.apply_pending_replacement_review
    ):
        parser.error(
            "--review-pending-replacements 只能单独使用，或与"
            " --apply-semantic-fallback-review 组合"
        )
    review_command_requested = bool(
        apply_review_commands
        or maintenance_commands
        or args.reopen_pre_ir
        or args.review_pending_replacements
        or args.export_pending_replacement_review
    )
    if review_command_requested:
        if args.requirement:
            parser.error("审核命令不能同时传入命令行需求")
        if args.fresh:
            parser.error("审核命令不能与 --fresh 同时使用")
        if not args.checkpoint_dir:
            parser.error("审核命令需要非空 --checkpoint-dir")
        if (
            args.reopen_pre_ir or args.review_pending_replacements
        ) and not sys.stdin.isatty():
            parser.error("交互式Pre-IR审核需要可交互终端")
        if args.review_pending_replacements and args.llm == "rules":
            parser.error("--review-pending-replacements 需要 --llm api")
        if args.export_pending_replacement_review and args.llm == "rules":
            parser.error("--export-pending-replacement-review 需要 --llm api")
        if args.refresh and not args.export_pending_replacement_review:
            parser.error("--refresh 只能与 --export-pending-replacement-review 配合")
        _run_pre_ir_review_command(args, parser)
        return
    if args.refresh:
        parser.error("--refresh 只能与 --export-pending-replacement-review 配合")

    single_requirement = " ".join(args.requirement) if args.requirement else None
    if single_requirement is not None and args.review_feedback_file:
        parser.error("--review-feedback-file 当前仅支持批量运行")
    if args.warmup_only and not args.warmup_file:
        parser.error("--warmup-only 需要同时提供 --warmup-file")
    if args.warmup_only and single_requirement is not None:
        parser.error("--warmup-only 不能和命令行单条需求同时使用")
    warmup_only = args.warmup_only
    deferred_pre_ir_mode = (
        single_requirement is None
        and not warmup_only
        and args.pre_ir_mode == "deferred"
        and args.segmentation_review == "manual"
        and args.plain_reading_review == "manual"
    )
    if deferred_pre_ir_mode and not args.checkpoint_dir:
        parser.error("批量延迟 Pre-IR 审核需要非空 --checkpoint-dir")

    # 非交互环境不能隐式切换为自动通过。
    review_mode = args.review
    if review_mode is None:
        review_mode = "manual"
    from delivery_policy import resolve_delivery_policy
    if args.delivery_policy is None and (warmup_only or args.warmup_file or args.warmup_size):
        args.delivery_policy = "strict"
    try:
        args.delivery_policy = resolve_delivery_policy(args.delivery_policy, review_mode, args.experiment_mode or "human")
    except ValueError as exc:
        parser.error(str(exc))
    if args.delivery_policy == "always_output":
        deferred_pre_ir_mode = False
    if requires_interactive_pre_ir_review(
        args.segmentation_review,
        args.plain_reading_review,
    ) and not sys.stdin.isatty() and not deferred_pre_ir_mode and args.delivery_policy == "strict":
        parser.error(
            "人工审核需要可交互终端。请在终端执行此命令；"
            "已准备人工预切分文件时可显式使用 --pre-ir-mode deferred。"
        )
    try:
        experiment_mode = resolve_experiment_mode(
            review_mode,
            args.experiment_mode,
        )
    except ValueError as exc:
        parser.error(str(exc))
    trace_evidence_policy = _resolve_trace_evidence_policy(
        args,
        parser,
        experiment_mode=experiment_mode,
        review_mode=review_mode,
    )
    learning_mode = _resolve_learning_mode(args)
    if experiment_mode == "formula_oracle" and learning_mode != "legacy_formula_oracle":
        parser.error(
            "--experiment-mode formula_oracle 是旧 warmup 兼容入口；"
            "请同时使用 --learning-mode legacy_formula_oracle，"
            "或改用 --learning-mode stage_guided。"
        )
    if args.reference_formula and learning_mode != "legacy_formula_oracle":
        parser.error(
            "--reference-formula 只允许 legacy_formula_oracle；"
            "stage_guided 请使用 --offline-validation final_formula。"
        )
    if learning_mode == "legacy_formula_oracle" and experiment_mode != "formula_oracle":
        parser.error(
            "legacy_formula_oracle 兼容模式需要 --experiment-mode formula_oracle。"
        )
    if experiment_mode == "formula_oracle" and not args.reference_formula:
        parser.error("--experiment-mode formula_oracle 需要提供 --reference-formula")
    if warmup_only and args.warmup_file:
        try:
            WarmupLearningConfig(
                warmup_file=args.warmup_file or "",
                warmup_only=warmup_only,
                reference_formula=args.reference_formula or "",
                data_split=args.data_split,
                learning_mode=learning_mode,
                offline_validation=args.offline_validation,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if experiment_mode == "formula_oracle":
        if single_requirement is not None:
            parser.error("formula_oracle 只允许 warmup-only 批量样本运行")
    if is_sample_reference_formula(args.reference_formula):
        if not args.warmup_file:
            parser.error("--reference-formula sample 需要同时提供 --warmup-file")
        if not warmup_only:
            parser.error("--reference-formula sample 当前只支持 --warmup-only 批量样本运行")
        if single_requirement is not None:
            parser.error("--reference-formula sample 不能和命令行单条需求同时使用")
    mode_warning = implicit_auto_human_warning(
        review_mode,
        args.experiment_mode,
        warmup_only=warmup_only,
    )
    if mode_warning and args.delivery_policy == "strict":
        print(mode_warning, file=sys.stderr)
    terminal_final_only = _should_print_terminal_final_only(args, review_mode)
    progress_mode = args.progress or "auto"
    progress = create_progress(progress_mode)
    progress.start_run(
        "Traj-LTLf run started",
        details=[
            f"Command: python -B {' '.join(sys.argv)}",
            (
                f"Mode: review={review_mode}, experiment={experiment_mode}, "
                f"learning={learning_mode}, llm={args.llm or 'env/default'}, "
                f"boundary_traces={trace_evidence_policy.boundary_source}"
            ),
            f"Checkpoint: {args.checkpoint_dir or 'disabled'}",
        ],
    )

    use_example_artifacts = (
        args.review in {"manual", "auto"}
        and args.llm == "api"
        and args.warmup_file is not None
    )
    final_unique_results = _should_use_final_unique_results(
        args,
        review_mode=review_mode,
        experiment_mode=experiment_mode,
    )
    if trace_evidence_policy.experimental_only:
        final_unique_results = False
    result_output_file = (
        EXAMPLE_RESULT_OUTPUT_FILE if use_example_artifacts else RESULT_OUTPUT_FILE
    )
    summary_output_file = (
        EXAMPLE_SUMMARY_OUTPUT_FILE if use_example_artifacts else SUMMARY_OUTPUT_FILE
    )
    review_records_dir = (
        EXAMPLE_REVIEW_RECORDS_DIR if use_example_artifacts else REVIEW_RECORDS_DIR
    )
    artifact_dirs_to_clear = (
        (EXAMPLE_ARTIFACT_DIR,) if use_example_artifacts else ARTIFACT_DIRS_TO_CLEAR
    )
    if args.run_dir is not None:
        result_output_file = str(args.run_dir / "outputs/result.txt")
        summary_output_file = str(args.run_dir / "outputs/final_candidates.txt")
        review_records_dir = str(args.run_dir / "review_records")
        artifact_dirs_to_clear = (str(args.run_dir / "outputs"), review_records_dir)
        args.checkpoint_dir = str(args.run_dir / "checkpoints")

    should_clear_artifacts = bool(
        args.fresh
        or (use_example_artifacts and not deferred_pre_ir_mode)
        or (
            single_requirement is not None
            and args.review in {"manual", "auto"}
            and args.llm == "api"
            and not args.review_feedback_file
        )
    )
    if args.delivery_policy == "always_output" and not args.fresh:
        should_clear_artifacts = False

    pipeline_output: PipelineResult | list[BatchPipelineItemResult] | None = None
    try:
        if single_requirement is not None:
            progress.info("Input: command-line requirement")
            _prepare_artifacts(
                should_clear=should_clear_artifacts,
                artifact_dirs=artifact_dirs_to_clear,
                review_records_dir=review_records_dir,
                use_example_artifacts=use_example_artifacts,
                preserve_degraded=False,
                fresh=bool(args.fresh),
                checkpoint_dir=args.checkpoint_dir or None,
                progress=progress,
            )
            result = run_traj_ltlf_pipeline(
                single_requirement,
                delivery_policy=args.delivery_policy,
                review_mode=review_mode,
                max_review_rounds=args.max_review_rounds,
                llm_mode=args.llm,
                records_dir=review_records_dir,
                approved_memory_file=args.rag_memory_file or None,
                progress=progress,
                progress_label="REQ-001",
                experiment_mode=experiment_mode,
                data_split=args.data_split,
                reference_formula=args.reference_formula,
                segmentation_review=args.segmentation_review,
                plain_reading_review=args.plain_reading_review,
                max_segmentation_retries=args.max_segmentation_retries,
                max_surface_review_rounds=args.max_surface_review_rounds,
                max_plain_reading_rounds=args.max_plain_reading_rounds,
                max_plain_reading_revision_rounds=(
                    args.max_plain_reading_revision_rounds
                ),
                segmentation_prompt_version=args.segmentation_prompt_version,
                plain_reading_prompt_version=args.plain_reading_prompt_version,
                plain_reading_semantic_fallback=(
                    args.plain_reading_semantic_fallback == "enabled"
                ),
                trace_evidence_policy=trace_evidence_policy,
            )
            report = format_pipeline_result(result)
            summary_report = format_pipeline_summary(result)
            if result.selection_method == "candidate_fallback":
                from contracts.records import write_json
                write_json(Path(review_records_dir) / "formula_fallback_REQ-001.json", {
                    "source_text": result.requirement, "source_hash": result.semantic.faithful_ir.source_hash,
                    "formula": str(result.provisional_candidate.formula), "approval_status": result.approval_status,
                    "record": next(record for record in reversed(result.review_records)
                                   if record.get("stage") == "formula_candidate_fallback"),
                })
            classification = (
                result.trace_coverage.status
                if result.trace_coverage.status
                in {"complete", "degraded_unproven", "degraded_proven"}
                else "rejected"
            )
            if trace_evidence_policy.experimental_only:
                classification = "needs_review"
            if result.delivery_policy == "always_output":
                classification = "complete"
            elif result.provisional_candidate is not None:
                classification = "complete_with_corrections"
            single_item = BatchPipelineItemResult(
                requirement_id="REQ-001",
                requirement=result.requirement,
                result=result,
                report_text=report,
                summary_text=summary_report,
                review_records=result.review_records,
                classification=classification,
                degraded_record=(
                    build_degraded_result_record("REQ-001", result)
                    if result.trace_coverage.degraded
                    else None
                ),
            )
            write_classified_batch_outputs(
                [single_item],
                output_dir=Path(result_output_file).parent,
                expected_items=[single_item],
            )
            if result.trace_coverage.degraded:
                report = format_batch_pipeline_results([single_item])
                summary_report = format_batch_pipeline_summary([single_item])
            pipeline_output = result
        else:
            progress.stage("Input loading")
            if warmup_only:
                requirements = []
                requirement_ids = []
                requirement_records = []
                ap_representation_profiles_by_id = {}
                declared_ap_catalogs_by_id = {}
                progress.info("Requirements file: skipped for warmup-only run")
            else:
                with progress.task(
                    f"Loading requirements file: {args.requirements_file}"
                ):
                    try:
                        try:
                            requirement_records = (
                                load_requirement_records(
                                    args.requirements_file
                                )
                            )
                        except FileNotFoundError:
                            # Compatibility seam for embedders that provide the
                            # historical text loader without a physical file.
                            requirements = load_requirements(
                                args.requirements_file
                            )
                            requirement_records = [
                                RequirementRecord(
                                    f"REQ-{index:03d}",
                                    text,
                                    ap_profile_declared=True,
                                )
                                for index, text in enumerate(requirements, start=1)
                            ]
                        else:
                            requirements = [
                                record.text for record in requirement_records
                            ]
                    except (FileNotFoundError, ValueError) as exc:
                        progress.error(f"需求文件读取失败: {exc}")
                        raise SystemExit(f"需求文件读取失败: {exc}") from exc
                requirement_ids = [
                    record.requirement_id for record in requirement_records
                ]
                ap_representation_profiles_by_id = {
                    record.requirement_id: record.ap_representation_profile
                    for record in requirement_records
                }
                declared_ap_catalogs_by_id = {
                    record.requirement_id: record.declared_ap_catalog
                    for record in requirement_records
                    if record.declared_ap_catalog is not None
                }
                progress.ok(
                    f"Loaded {args.requirements_file}: {len(requirements)} requirement(s)"
                )
            if args.warmup_file:
                with progress.task(f"Loading warmup file: {args.warmup_file}"):
                    try:
                        warmup_examples = load_warmup_examples_from_markdown(
                            args.warmup_file
                        )
                    except (FileNotFoundError, ValueError) as exc:
                        progress.error(f"warmup 文件读取失败: {exc}")
                        raise SystemExit(f"warmup 文件读取失败: {exc}") from exc
                progress.ok(
                    f"Loaded {args.warmup_file}: {len(warmup_examples)} example(s)"
                )
            else:
                warmup_examples = None
                progress.info("Warmup examples: 0")
            checkpoint_dir = None
            if args.checkpoint_dir:
                checkpoint_root = prepare_checkpoint_run(
                    args.checkpoint_dir,
                    manifest=_build_checkpoint_manifest(
                        requirements=requirements,
                        requirement_ids=requirement_ids,
                        review_mode=review_mode,
                        llm_mode=args.llm,
                        max_review_rounds=args.max_review_rounds,
                        requirements_file=args.requirements_file,
                        human_segmentation_file=args.human_segmentation_file,
                        warmup_examples=warmup_examples,
                        warmup_file=args.warmup_file,
                        warmup_only=warmup_only,
                        warmup_size=args.warmup_size,
                        approved_memory_file=args.rag_memory_file or None,
                        experiment_mode=experiment_mode,
                        data_split=args.data_split,
                        reference_formula=args.reference_formula,
                        learning_mode=learning_mode,
                        offline_validation=args.offline_validation,
                        segmentation_review=args.segmentation_review,
                        plain_reading_review=args.plain_reading_review,
                        max_segmentation_retries=args.max_segmentation_retries,
                        max_surface_review_rounds=args.max_surface_review_rounds,
                        max_plain_reading_rounds=args.max_plain_reading_rounds,
                        max_plain_reading_revision_rounds=(
                            args.max_plain_reading_revision_rounds
                        ),
                        segmentation_prompt_version=args.segmentation_prompt_version,
                        plain_reading_prompt_version=args.plain_reading_prompt_version,
                        plain_reading_semantic_fallback=(
                            args.plain_reading_semantic_fallback == "enabled"
                        ),
                        trace_evidence_policy=trace_evidence_policy,
                        delivery_policy=args.delivery_policy,
                    ),
                    fresh=args.fresh,
                )
                checkpoint_dir = str(checkpoint_root) if checkpoint_root is not None else None
                if checkpoint_dir:
                    progress.info(f"Checkpoint directory: {checkpoint_dir}")
            _prepare_artifacts(
                should_clear=should_clear_artifacts,
                artifact_dirs=artifact_dirs_to_clear,
                review_records_dir=review_records_dir,
                use_example_artifacts=use_example_artifacts,
                preserve_degraded=bool(checkpoint_dir and not args.fresh),
                fresh=bool(args.fresh),
                checkpoint_dir=checkpoint_dir or args.checkpoint_dir or None,
                progress=progress,
            )
            if deferred_pre_ir_mode:
                pre_ir_outcome = ensure_deferred_pre_ir_batch(
                    requirement_records,
                    human_segmentation_file=args.human_segmentation_file,
                    checkpoint_dir=checkpoint_dir or args.checkpoint_dir,
                    records_dir=review_records_dir,
                    llm_mode=args.llm,
                    approved_memory_file=args.rag_memory_file or None,
                    max_segmentation_retries=args.max_segmentation_retries,
                    segmentation_prompt_version=args.segmentation_prompt_version,
                    plain_reading_prompt_version=args.plain_reading_prompt_version,
                    data_split=args.data_split,
                    semantic_fallback_enabled=(
                        args.plain_reading_semantic_fallback == "enabled"
                    ),
                    progress=progress,
                )
                if pre_ir_outcome.should_stop:
                    progress.done()
                    progress.close()
                    print(pre_ir_outcome.message)
                    return
                mark_pre_ir_downstream_started(
                    checkpoint_dir or args.checkpoint_dir
                )
            batch_results = run_batch_traj_ltlf_pipeline(
                requirements,
                delivery_policy=args.delivery_policy,
                requirement_records=(
                    requirement_records if not warmup_only else None
                ),
                requirement_ids=requirement_ids,
                review_mode=review_mode,
                max_review_rounds=args.max_review_rounds,
                llm_mode=args.llm,
                warmup_size=args.warmup_size,
                warmup_examples=warmup_examples,
                warmup_only=warmup_only,
                approved_memory_file=args.rag_memory_file or None,
                records_dir=review_records_dir,
                checkpoint_dir=checkpoint_dir,
                partial_result_path=_partial_output_path(result_output_file),
                partial_summary_path=_partial_output_path(summary_output_file),
                classified_output_dir=str(Path(result_output_file).parent),
                progress=progress,
                experiment_mode=experiment_mode,
                data_split=args.data_split,
                reference_formula=args.reference_formula,
                learning_mode=learning_mode,
                offline_validation=args.offline_validation,
                final_unique_results=final_unique_results,
                require_complete_batch=args.require_complete_batch,
                review_feedback_file=args.review_feedback_file,
                segmentation_review=args.segmentation_review,
                plain_reading_review=args.plain_reading_review,
                max_segmentation_retries=args.max_segmentation_retries,
                max_surface_review_rounds=args.max_surface_review_rounds,
                max_plain_reading_rounds=args.max_plain_reading_rounds,
                max_plain_reading_revision_rounds=(
                    args.max_plain_reading_revision_rounds
                ),
                segmentation_prompt_version=args.segmentation_prompt_version,
                plain_reading_prompt_version=args.plain_reading_prompt_version,
                plain_reading_semantic_fallback=(
                    args.plain_reading_semantic_fallback == "enabled"
                ),
                trace_evidence_policy=trace_evidence_policy,
                ap_representation_profiles_by_id=(
                    ap_representation_profiles_by_id
                ),
                declared_ap_catalogs_by_id=declared_ap_catalogs_by_id,
            )
            report = (
                format_final_unique_batch_results(batch_results)
                if final_unique_results
                else format_batch_pipeline_results(batch_results)
            )
            summary_report = (
                format_final_unique_batch_summary(batch_results)
                if final_unique_results
                else format_batch_pipeline_summary(batch_results)
            )
            pipeline_output = batch_results
            if final_unique_results:
                counts = batch_unique_result_counts(batch_results)
                progress.info(
                    "最终统计: "
                    f"总需求数={counts['total']}, "
                    f"有效最终公式={counts['final_formula_count']}, "
                    f"provisional={counts['provisional_candidate_count']}, "
                    f"blocked={counts['blocked_error_count']}"
                )
            else:
                counts = batch_classification_counts(batch_results)
                progress.info(
                    "分类统计: "
                    f"正常结果={counts['normal']}, "
                    f"待审查/实验结果={counts['needs_review']}, "
                    f"降级待审查={counts['degraded']}, "
                    f"阻断错误={counts['blocked_error']}, "
                    f"失败结果={counts['rejected']}"
                )
    except PlainReadingSemanticResolutionRequired as exc:
        pending_path = Path(review_records_dir) / f"plain_reading_pending_{exc.requirement_id}_{exc.review.source_hash[:12]}.json"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(json.dumps({
            "status": "awaiting_candidate_adjudication", "reason_code": exc.code,
            "plain_reading_review": asdict(exc.review),
            "surface_segmentation": asdict(exc.segmentation) if exc.segmentation is not None else None,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress.enabled:
            progress.wait(exc.diagnostics["user_message"], details=[f"记录：{pending_path}"])
        else:
            print(exc.diagnostics["user_message"])
            print(f"记录：{pending_path}")
        return
    except SourceGraphResolutionRequired as exc:
        from contracts.records import write_json
        name = re.sub(r"[^A-Za-z0-9_-]", "_", exc.requirement_id)
        pending_path = Path(review_records_dir) / f"source_graph_pending_{name}_{exc.source_hash[:12]}.json"
        write_json(pending_path, {"status": "awaiting_source_semantic_resolution", **exc.diagnostics})
        if progress.enabled:
            progress.wait(str(exc), details=[f"记录：{pending_path}"])
        else:
            print(str(exc))
            print(f"记录：{pending_path}")
        return
    except APPreparationError as exc:
        message = str(exc)
        if exc.code == "AP_PROPOSAL_INVALID":
            from contracts.records import write_json
            name = re.sub(r"[^A-Za-z0-9_-]", "_", str(exc.diagnostics.get("requirement_id", "unknown")))
            source_digest = re.sub(r"[^A-Fa-f0-9]", "", str(exc.diagnostics.get("source_hash", "")))[:12]
            pending_path = Path(review_records_dir) / f"ap_pending_{name}_{source_digest or 'unknown'}.json"
            write_json(pending_path, {"status": "awaiting_ap_correction", "stage": "ap_grounding",
                                      "reason_code": exc.code, "diagnostics": exc.diagnostics})
            message += f"\n记录：{pending_path}"
        progress.error(message)
        raise SystemExit(message) from None
    except LLMClientError as exc:
        progress.error(f"LLM API 配置或调用失败: {exc}")
        raise SystemExit(f"LLM API 配置或调用失败: {exc}") from exc
    except Exception as exc:
        progress.error(f"运行失败: {exc}")
        raise

    progress.stage("Writing outputs")
    with progress.task("Writing full result"):
        full_output_path = write_result_file(report, output_path=result_output_file)
    progress.ok(f"Full result: {full_output_path}")
    with progress.task("Writing summary result"):
        summary_output_path = write_result_file(
            summary_report,
            output_path=summary_output_file,
        )
    progress.ok(f"Summary: {summary_output_path}")
    progress.done()
    progress.close()

    if terminal_final_only:
        print(_format_terminal_final_ltlf(pipeline_output))
    else:
        display_report = (
            summary_report
            if review_mode == "manual" and args.llm == "api"
            else report
        )
        print(display_report)
        print("")
        print(f"完整结果已保存到: {full_output_path}")
        print(f"精简结果已保存到: {summary_output_path}")


def _should_print_terminal_final_only(args: argparse.Namespace, review_mode: str) -> bool:
    return args.llm == "api" and review_mode in {"auto", "manual"}


def _run_pre_ir_review_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Run deferred Pre-IR mutation and optional replacement queue stages."""

    del parser
    if args.audit_ap_identity is not None:
        try:
            report = audit_ap_identity_artifacts(args.checkpoint_dir)
            payload = json.dumps(
                report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            )
            if args.audit_ap_identity == "-":
                print(payload)
            else:
                output = Path(args.audit_ap_identity)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(payload + "\n", encoding="utf-8")
                print(f"AP identity audit已保存到: {output}")
            if not report.passed:
                raise PreIRStateConsistencyError(
                    "AP identity audit发现corrupt/stale artifact"
                )
        except Exception as exc:
            print(f"[ERR] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        return
    if args.migrate_v1_semantic_overrides is not None:
        try:
            report = migrate_v1_pending_semantic_overrides(
                args.checkpoint_dir,
                commit=args.migrate_v1_semantic_overrides == "commit",
                reviewer_id="cli-v1-migration-reviewer",
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        except Exception as exc:
            print(f"[ERR] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        return
    progress = None
    queue_progress = None
    review_file = (
        args.apply_no_ambiguity_review
        or args.apply_semantic_fallback_review
        or args.apply_semantic_override_ap_inventory_review
        or args.apply_semantic_override_ap_mapping_review
        or args.apply_ambiguity_review
        or args.apply_pending_replacement_review
    )
    if review_file or args.export_pending_replacement_review:
        progress = create_progress(args.progress or "off")
        if args.export_pending_replacement_review:
            review_kind = "replacement"
        elif args.apply_pending_replacement_review:
            review_kind = "replacement"
        elif args.apply_no_ambiguity_review:
            review_kind = "no-ambiguity"
        elif args.apply_semantic_fallback_review:
            review_kind = "semantic-fallback"
        elif args.apply_semantic_override_ap_inventory_review:
            review_kind = "semantic-override-ap-inventory"
        elif args.apply_semantic_override_ap_mapping_review:
            review_kind = "semantic-override-ap-mapping"
        else:
            review_kind = "ambiguity"
        operation = (
            "export started"
            if args.export_pending_replacement_review
            else "review apply started"
        )
        progress.start_run(
            f"Traj-LTLf {review_kind} {operation}",
            details=[
                *(
                    [f"Review file: {review_file}"]
                    if review_file
                    else []
                ),
                f"LLM: {args.llm or 'env/default'}",
                f"Checkpoint: {args.checkpoint_dir}",
            ],
        )

    try:
        if args.export_pending_replacement_review:
            export_outcome = export_pending_replacement_review(
                checkpoint_dir=args.checkpoint_dir,
                llm_mode=args.llm,
                approved_memory_file=args.rag_memory_file or None,
                max_segmentation_retries=args.max_segmentation_retries,
                segmentation_prompt_version=args.segmentation_prompt_version,
                plain_reading_prompt_version=args.plain_reading_prompt_version,
                semantic_fallback_enabled=(
                    args.plain_reading_semantic_fallback == "enabled"
                ),
                refresh=args.refresh,
                progress=progress,
            )
            if progress is not None:
                if export_outcome.failed_ids:
                    progress.wait(
                        "Replacement review exported with draft generation failures"
                    )
                else:
                    progress.done("Replacement review export finished")
            print(export_outcome.message)
            return

        apply_outcome = None
        if args.apply_no_ambiguity_review:
            apply_outcome = apply_no_ambiguity_review(
                args.apply_no_ambiguity_review,
                checkpoint_dir=args.checkpoint_dir,
                llm_mode=args.llm,
                approved_memory_file=args.rag_memory_file or None,
                plain_reading_prompt_version=args.plain_reading_prompt_version,
                semantic_fallback_enabled=(
                    args.plain_reading_semantic_fallback == "enabled"
                ),
                progress=progress,
            )
        elif args.apply_semantic_fallback_review:
            apply_outcome = apply_semantic_fallback_review(
                args.apply_semantic_fallback_review,
                checkpoint_dir=args.checkpoint_dir,
                llm_mode=args.llm,
                approved_memory_file=args.rag_memory_file or None,
                max_segmentation_retries=args.max_segmentation_retries,
                segmentation_prompt_version=args.segmentation_prompt_version,
                plain_reading_prompt_version=args.plain_reading_prompt_version,
                semantic_fallback_enabled=(
                    args.plain_reading_semantic_fallback == "enabled"
                ),
                progress=progress,
            )
        elif args.apply_semantic_override_ap_inventory_review:
            apply_outcome = apply_semantic_override_ap_inventory_review(
                args.apply_semantic_override_ap_inventory_review,
                checkpoint_dir=args.checkpoint_dir,
                progress=progress,
            )
        elif args.apply_semantic_override_ap_mapping_review:
            apply_outcome = apply_semantic_override_ap_mapping_review(
                args.apply_semantic_override_ap_mapping_review,
                checkpoint_dir=args.checkpoint_dir,
                progress=progress,
            )
        elif args.apply_ambiguity_review:
            apply_outcome = apply_ambiguity_review(
                args.apply_ambiguity_review,
                checkpoint_dir=args.checkpoint_dir,
                llm_mode=args.llm,
                approved_memory_file=args.rag_memory_file or None,
                max_segmentation_retries=args.max_segmentation_retries,
                segmentation_prompt_version=args.segmentation_prompt_version,
                plain_reading_prompt_version=args.plain_reading_prompt_version,
                semantic_fallback_enabled=(
                    args.plain_reading_semantic_fallback == "enabled"
                ),
                progress=progress,
            )
        elif args.apply_pending_replacement_review:
            apply_outcome = apply_pending_replacement_review(
                args.apply_pending_replacement_review,
                checkpoint_dir=args.checkpoint_dir,
                llm_mode=args.llm,
                approved_memory_file=args.rag_memory_file or None,
                max_segmentation_retries=args.max_segmentation_retries,
                segmentation_prompt_version=args.segmentation_prompt_version,
                plain_reading_prompt_version=args.plain_reading_prompt_version,
                semantic_fallback_enabled=(
                    args.plain_reading_semantic_fallback == "enabled"
                ),
                progress=progress,
            )
        elif args.reopen_pre_ir:
            apply_outcome = reopen_pre_ir_item(
                args.reopen_pre_ir,
                checkpoint_dir=args.checkpoint_dir,
                llm_mode=args.llm,
                approved_memory_file=args.rag_memory_file or None,
                max_segmentation_retries=args.max_segmentation_retries,
                max_surface_review_rounds=args.max_surface_review_rounds,
                max_plain_reading_rounds=args.max_plain_reading_rounds,
                max_plain_reading_revision_rounds=(
                    args.max_plain_reading_revision_rounds
                ),
                segmentation_prompt_version=args.segmentation_prompt_version,
                plain_reading_prompt_version=args.plain_reading_prompt_version,
                semantic_fallback_enabled=(
                    args.plain_reading_semantic_fallback == "enabled"
                ),
            )

        if apply_outcome is not None:
            if progress is not None:
                if apply_outcome.next_review_files:
                    progress.wait(
                        "Pre-IR review applied; another review round is required",
                        details=[
                            f"Next review file: {path}"
                            for path in apply_outcome.next_review_files
                        ],
                    )
                elif not apply_outcome.pre_ir_complete:
                    progress.wait(
                        "Pre-IR review applied; additional human review is required"
                    )
                else:
                    progress.done("Pre-IR review apply finished")
            print(apply_outcome.message)
            if apply_outcome.had_errors:
                raise SystemExit(1)

        if args.review_pending_replacements:
            if progress is not None:
                progress.close()
                progress = None
            queue_progress = create_progress(args.progress or "off")
            queue_progress.start_run(
                "Traj-LTLf pending replacement Pre-IR review started",
                details=[
                    f"LLM: {args.llm or 'env/default'}",
                    f"Checkpoint: {args.checkpoint_dir}",
                ],
            )
            queue_outcome = review_pending_replacement_pre_ir(
                checkpoint_dir=args.checkpoint_dir,
                llm_mode=args.llm,
                approved_memory_file=args.rag_memory_file or None,
                max_segmentation_retries=args.max_segmentation_retries,
                max_surface_review_rounds=args.max_surface_review_rounds,
                max_plain_reading_rounds=args.max_plain_reading_rounds,
                max_plain_reading_revision_rounds=(
                    args.max_plain_reading_revision_rounds
                ),
                segmentation_prompt_version=args.segmentation_prompt_version,
                plain_reading_prompt_version=args.plain_reading_prompt_version,
                semantic_fallback_enabled=(
                    args.plain_reading_semantic_fallback == "enabled"
                ),
                progress=queue_progress,
            )
            if queue_outcome.had_errors:
                queue_progress.error(
                    "Pending replacement Pre-IR review stopped with errors"
                )
            elif queue_outcome.halt_reason == "interrupted":
                queue_progress.wait(
                    "Pending replacement Pre-IR review was interrupted"
                )
            elif queue_outcome.remaining_ids:
                queue_progress.wait(
                    "Pending replacement Pre-IR review requires more human input"
                )
            else:
                queue_progress.done(
                    "Pending replacement Pre-IR review queue finished"
                )
            print(queue_outcome.message)
            if queue_outcome.had_errors:
                raise SystemExit(1)
            if queue_outcome.halt_reason == "interrupted":
                raise SystemExit(130)
    except ReviewFileInputError as exc:
        active_progress = queue_progress or progress
        if active_progress is not None:
            active_progress.error(str(exc))
        print(f"[ERR] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except (
        FileNotFoundError,
        ValueError,
        LLMClientError,
        PreIRStateConsistencyError,
    ) as exc:
        active_progress = queue_progress or progress
        if active_progress is not None:
            active_progress.error(str(exc))
        print(f"[ERR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt as exc:
        active_progress = queue_progress or progress
        if active_progress is not None:
            active_progress.wait("Pre-IR review was interrupted")
        recovery_command = (
            "--export-pending-replacement-review"
            if (
                args.export_pending_replacement_review
                or args.apply_pending_replacement_review
            )
            else "--review-pending-replacements"
        )
        print(
            "[WAIT] Pre-IR review was interrupted; rerun "
            + recovery_command
            + " with the same checkpoint.",
            file=sys.stderr,
        )
        raise SystemExit(130) from exc
    except Exception as exc:
        active_progress = queue_progress or progress
        if active_progress is not None:
            active_progress.error(f"{type(exc).__name__}: {exc}")
        print(
            f"[ERR] unexpected runtime failure ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    finally:
        if queue_progress is not None:
            queue_progress.close()
        if progress is not None:
            progress.close()


def _resolve_learning_mode(args: argparse.Namespace) -> str:
    if args.learning_mode:
        return args.learning_mode
    if args.experiment_mode == "formula_oracle" or is_sample_reference_formula(
        args.reference_formula
    ):
        return "legacy_formula_oracle"
    return "stage_guided"


def _should_use_final_unique_results(
    args: argparse.Namespace,
    *,
    review_mode: str,
    experiment_mode: str,
) -> bool:
    if review_mode == "auto" and experiment_mode == "independent_judge":
        return True
    return (
        review_mode == "manual"
        and experiment_mode == "human"
        and args.llm == "api"
    )


def _format_terminal_final_ltlf(
    pipeline_output: object,
) -> str:
    if isinstance(pipeline_output, list):
        lines = [
            _format_batch_item_terminal_final(item)
            for item in pipeline_output
        ]
        unique_counts = batch_unique_result_counts(pipeline_output)
        if unique_counts["final_formula_count"] == unique_counts["total"]:
            return "\n".join(lines)
        counts = batch_classification_counts(pipeline_output)
        if counts["needs_review"] or counts["degraded"] or counts["rejected"]:
            lines.extend(
                [
                    "",
                    f"正常结果: {counts['normal']}",
                    f"待审查结果: {counts['needs_review']}",
                    f"降级待审查: {counts['degraded']}",
                    f"失败结果: {counts['rejected']}",
                ]
            )
        return "\n".join(lines)
    return _selected_formula_text(pipeline_output)


def _format_batch_item_terminal_final(item: BatchPipelineItemResult) -> str:
    if item.result is not None:
        candidate = _selected_formula_text(item.result)
    else:
        candidate = _extract_final_candidate_from_summary(item.summary_text)
    return f"{item.requirement_id}: {candidate}"


def _selected_formula_text(result: object) -> str:
    selected = getattr(result, "selected", None) or getattr(result, "provisional_candidate", None)
    if selected is None:
        return "暂不输出（needs_review）"
    return str(selected.formula)


def _extract_final_candidate_from_summary(summary_text: str) -> str:
    lines = summary_text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "最终候选:":
            for candidate_line in lines[index + 1 :]:
                candidate = candidate_line.strip()
                if candidate:
                    return candidate
    return "暂不输出（checkpoint 缺少精简结果）"


def _partial_output_path(output_file: str) -> str:
    path = Path(output_file)
    return str(path.with_name(f"{path.stem}.partial{path.suffix}"))


def _resolve_trace_evidence_policy(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    experiment_mode: str,
    review_mode: str,
) -> TraceEvidencePolicy:
    source = str(args.boundary_trace_policy)
    fixture_path = args.boundary_traces_file
    if source != "frozen_file" and fixture_path:
        parser.error(
            "--boundary-traces-file 只能与 --boundary-trace-policy frozen_file 组合"
        )
    if source == "frozen_file" and not fixture_path:
        parser.error(
            "--boundary-trace-policy frozen_file 需要 --boundary-traces-file"
        )
    if source == "frozen_file" and not (
        review_mode == "auto" and experiment_mode == "independent_judge"
    ):
        parser.error(
            "frozen_file 自动标注边界证据要求 --review auto "
            "--experiment-mode independent_judge"
        )
    if source != "frozen_file":
        return TraceEvidencePolicy(boundary_source=source)
    try:
        fixture = load_boundary_trace_fixture(fixture_path)
    except BoundaryTraceFixtureError as exc:
        parser.error(f"{exc.code}: {exc}")
    return TraceEvidencePolicy(
        boundary_source="frozen_file",
        boundary_fixture_path=fixture.path,
        boundary_fixture_sha256=fixture.sha256,
        boundary_fixture_schema_version=fixture.schema_version,
    )


def _prepare_artifacts(
    *,
    should_clear: bool,
    artifact_dirs: tuple[str, ...],
    review_records_dir: str,
    use_example_artifacts: bool,
    preserve_degraded: bool,
    fresh: bool = False,
    checkpoint_dir: str | None = None,
    progress,
) -> None:
    if not should_clear:
        return
    progress.stage("Preparing artifacts")
    review_root = Path(review_records_dir).resolve(strict=False)
    review_records_prepared = False
    for artifact_dir in artifact_dirs:
        artifact_root = Path(artifact_dir).resolve(strict=False)
        preserve_names = {"degraded"} if preserve_degraded else set()
        if artifact_root == review_root:
            preserve_names.add(PRE_IR_BATCHES_DIR)
            review_records_prepared = True
        else:
            try:
                relative_review_root = review_root.relative_to(artifact_root)
            except ValueError:
                pass
            else:
                if relative_review_root.parts:
                    preserve_names.add(relative_review_root.parts[0])
        clear_directory_contents(
            artifact_dir,
            preserve_names=preserve_names or None,
        )
        progress.info(f"Cleared {artifact_dir}/")
    if use_example_artifacts or not review_records_prepared:
        clear_directory_contents(
            review_records_dir,
            preserve_names={PRE_IR_BATCHES_DIR},
        )
        progress.info(f"Prepared {review_records_dir}/")
    if fresh and checkpoint_dir:
        removed = _remove_current_pre_ir_review_namespace(
            review_records_dir=review_records_dir,
            checkpoint_dir=checkpoint_dir,
        )
        progress.info(f"Cleared current Pre-IR namespace: {removed}")


def _remove_current_pre_ir_review_namespace(
    *,
    review_records_dir: str,
    checkpoint_dir: str,
) -> Path:
    """Remove only the namespace owned by one checkpoint path."""

    records_root = Path(review_records_dir).resolve(strict=False)
    batches_candidate = records_root / PRE_IR_BATCHES_DIR
    if batches_candidate.is_symlink():
        raise RuntimeError("拒绝清理符号链接形式的 Pre-IR batch 根目录。")
    if batches_candidate.exists() and not batches_candidate.is_dir():
        raise RuntimeError("Pre-IR batch 根不是目录。")
    batches_root = batches_candidate.resolve(strict=False)
    try:
        batches_root.relative_to(records_root)
    except ValueError as exc:
        raise RuntimeError("Pre-IR batch 根目录逃逸 review_records。") from exc

    target_candidate = batches_root / _checkpoint_fingerprint(checkpoint_dir)
    if target_candidate.is_symlink():
        raise RuntimeError("拒绝清理符号链接形式的 Pre-IR checkpoint namespace。")
    target = target_candidate.resolve(strict=False)
    try:
        target.relative_to(records_root)
    except ValueError as exc:
        raise RuntimeError("Pre-IR checkpoint namespace 逃逸 review_records。") from exc
    if target.parent != batches_root:
        raise RuntimeError("Pre-IR checkpoint namespace 不是 batch 根的直接子目录。")
    if target.exists():
        if not target.is_dir():
            raise RuntimeError("Pre-IR checkpoint namespace 不是目录。")
        shutil.rmtree(target)
    return target


def _build_checkpoint_manifest(
    *,
    requirements: list[str],
    review_mode: str,
    llm_mode: str | None,
    max_review_rounds: int,
    requirements_file: str,
    warmup_examples: object,
    warmup_file: str | None,
    warmup_only: bool,
    warmup_size: int,
    approved_memory_file: str | None,
    experiment_mode: str,
    data_split: str,
    reference_formula: str | None,
    learning_mode: str,
    offline_validation: str,
    segmentation_review: str = "rules",
    plain_reading_review: str = "off",
    max_segmentation_retries: int = 2,
    max_surface_review_rounds: int = 3,
    max_plain_reading_rounds: int = 2,
    max_plain_reading_revision_rounds: int = 2,
    segmentation_prompt_version: str = "plain-segmentation-v1",
    plain_reading_prompt_version: str = "plain-reading-v5",
    plain_reading_semantic_fallback: bool = True,
    trace_evidence_policy: TraceEvidencePolicy = TraceEvidencePolicy(),
    requirement_ids: list[str] | None = None,
    human_segmentation_file: str | None = None,
    delivery_policy: str = "strict",
) -> dict:
    return {
        "delivery_policy": delivery_policy,
        "output_contract_version": "always-output-v1" if delivery_policy == "always_output" else "strict-v1",
        "requirements": list(requirements),
        "requirement_ids": list(requirement_ids or []),
        "review_mode": review_mode,
        "llm_mode": llm_mode,
        "max_review_rounds": max_review_rounds,
        "requirements_file": requirements_file,
        "human_segmentation_file": human_segmentation_file or "",
        "human_segmentation_content": _optional_file_content(
            human_segmentation_file
        ),
        "warmup_file": warmup_file,
        "warmup_only": warmup_only,
        "warmup_size": warmup_size,
        "approved_memory_file": approved_memory_file,
        "experiment_mode": experiment_mode,
        "learning_mode": learning_mode,
        "offline_validation": offline_validation,
        "segmentation_review": segmentation_review,
        "plain_reading_review": plain_reading_review,
        "max_segmentation_retries": max_segmentation_retries,
        "max_surface_review_rounds": max_surface_review_rounds,
        "max_plain_reading_rounds": max_plain_reading_rounds,
        "max_plain_reading_revision_rounds": max_plain_reading_revision_rounds,
        "segmentation_prompt_version": segmentation_prompt_version,
        "plain_reading_prompt_version": plain_reading_prompt_version,
        "plain_reading_semantic_fallback": plain_reading_semantic_fallback,
        "trace_evidence_policy": asdict(trace_evidence_policy),
        "data_split": data_split,
        "reference_formula_present": reference_formula is not None,
        "reference_formula_mode": (
            "sample"
            if is_sample_reference_formula(reference_formula)
            else "global"
            if reference_formula is not None
            else "none"
        ),
        "reference_formula_text": reference_formula or "",
        "faithful_ir_runtime_contract": faithful_ir_runtime_contract(),
        "warmup_examples": [
            {
                "requirement_id": getattr(example, "requirement_id", ""),
                "requirement": getattr(example, "requirement", ""),
                "process_text": getattr(example, "process_text", ""),
                "expected_semantic": (
                    asdict(example.expected_semantic)
                    if getattr(example, "expected_semantic", None) is not None
                    else None
                ),
                "expected_trace_count": len(getattr(example, "expected_traces", ())),
                "pattern_spec": (
                    asdict(example.pattern_spec)
                    if getattr(example, "pattern_spec", None) is not None
                    else None
                ),
            }
            for example in (warmup_examples or [])
        ],
    }


def _optional_file_content(path: str | None) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return "<missing>"
    return file_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
