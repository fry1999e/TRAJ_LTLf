"""
Traj-LTLf 的轻量知识库加载器。

第一版知识库使用本地 Markdown 文件，避免引入数据库或向量库。以后如果需要
接入真正的向量检索，可以保持 KnowledgeChunk 结构不变，只替换 Retriever。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowledgeChunk:
    """一个可被 RAG 检索的知识片段。"""

    chunk_id: str
    title: str
    text: str
    source: str


def load_default_kb(kb_dir: str = "data/examples/knowledge_base") -> list[KnowledgeChunk]:
    """
    加载默认知识库。

    如果 data/examples/knowledge_base 目录存在 Markdown 文件，就从文件读取；否则使用内置知识片段。
    """

    path = Path(kb_dir)
    markdown_files = sorted(path.glob("*.md")) if path.exists() else []
    chunks: list[KnowledgeChunk] = []

    for markdown_file in markdown_files:
        chunks.extend(_split_markdown_file(markdown_file))

    if chunks:
        return chunks

    return [
        KnowledgeChunk(
            chunk_id="builtin-response",
            title="Response 模式",
            text=(
                "Whenever A occurs, B should eventually occur 通常对应 "
                "G(A -> F B)。whenever 表示全局范围，eventually 表示 F。"
            ),
            source="builtin",
        ),
        KnowledgeChunk(
            chunk_id="builtin-trace",
            title="Response 轨迹示例",
            text=(
                "正轨迹：A 后面出现 B。反轨迹：A 出现后直到结束都没有 B。"
                "边界轨迹：A 与 B 同时出现，用来区分 F 与 X F。"
            ),
            source="builtin",
        ),
    ]


def _split_markdown_file(path: Path) -> list[KnowledgeChunk]:
    """
    把一个 Markdown 文件切成知识片段。

    为了简单可靠，当前按二级标题 ``##`` 切分；如果文件没有二级标题，就把
    整个文件作为一个知识片段。
    """

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    chunks: list[KnowledgeChunk] = []
    current_title = path.stem
    current_lines: list[str] = []

    def flush() -> None:
        if not current_lines:
            return
        chunk_text = "\n".join(current_lines).strip()
        if not chunk_text:
            return
        chunk_number = len(chunks) + 1
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{path.stem}-{chunk_number}",
                title=current_title,
                text=chunk_text,
                source=str(path),
            )
        )

    for line in lines:
        if line.startswith("## "):
            flush()
            current_title = line.replace("## ", "", 1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return chunks
