"""文件加载：把磁盘上的文档转成统一的 Document 列表。

支持格式：pdf / docx / md / txt / log / csv / json / html / htm
编码处理：显式声明 > UTF-8 > UTF-8-BOM > GB18030（覆盖 GBK/GB2312） > UTF-16
"""
from __future__ import annotations

import csv
import hashlib
import html as html_lib
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence

from .models import Document

SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".md",
    ".markdown",
    ".txt",
    ".log",
    ".csv",
    ".json",
    ".html",
    ".htm",
)

_DECODE_CANDIDATES = ("utf-8", "utf-8-sig", "gb18030", "utf-16")

_IGNORED_DIR_NAMES = {".git", ".idea", ".vscode", "__pycache__", "node_modules", ".venv", "venv", "data"}
_IGNORED_FILE_NAMES = {"thumbs.db", ".ds_store", "desktop.ini"}


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    """按块计算文件哈希，用于"同一文件不重复入库"。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text_file(path: Path, encoding: Optional[str] = None) -> str:
    """按候选编码逐个尝试解码，全部失败时以替换模式兜底（不会抛出）。"""
    raw = Path(path).read_bytes()
    candidates = [encoding] if encoding else []
    candidates.extend(_DECODE_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return raw.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _clean_text(text: str) -> str:
    """统一换行、去掉行尾空白、压缩多余空行，不改动语义内容。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------- 各格式
def _load_pdf(path: Path, doc_id: str) -> List[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的明确提示
        raise RuntimeError("处理 PDF 需要 pypdf，请先执行 pip install pypdf") from exc

    reader = PdfReader(str(path))
    documents: List[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # 个别损坏页不应中断整份文档
            text = ""
        text = _clean_text(text)
        if len(text) < 2:
            continue
        documents.append(Document(text=text, source=str(path), page=page_number, doc_id=doc_id))
    if not documents:
        raise RuntimeError(f"PDF 未提取到文本（可能是扫描件，需要 OCR）：{path.name}")
    return documents


def _load_docx(path: Path, doc_id: str) -> List[Document]:
    try:
        import docx
    except ImportError as exc:
        raise RuntimeError("处理 DOCX 需要 python-docx，请先执行 pip install python-docx") from exc

    document = docx.Document(str(path))
    blocks: List[str] = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            line = " | ".join(cell for cell in cells if cell)
            if line:
                blocks.append(line)
    text = _clean_text("\n".join(blocks))
    if not text:
        raise RuntimeError(f"DOCX 中未找到可用文本：{path.name}")
    return [Document(text=text, source=str(path), doc_id=doc_id)]


def _strip_html(raw: str) -> str:
    raw = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<(br|/p|/div|/li|/h\d)[^>]*>", "\n", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return html_lib.unescape(raw)


def _load_html(path: Path, doc_id: str, encoding: Optional[str]) -> List[Document]:
    raw = read_text_file(path, encoding)
    text = _clean_text(_strip_html(raw))
    if not text:
        raise RuntimeError(f"HTML 中未找到可用文本：{path.name}")
    return [Document(text=text, source=str(path), doc_id=doc_id)]


def _load_csv(path: Path, doc_id: str, encoding: Optional[str]) -> List[Document]:
    raw = read_text_file(path, encoding)
    rows = list(csv.reader(raw.splitlines()))
    if not rows:
        raise RuntimeError(f"CSV 为空：{path.name}")
    header = rows[0]
    lines: List[str] = []
    for row_index, row in enumerate(rows[1:], start=1):
        cells = []
        for column_index, cell in enumerate(row):
            name = header[column_index] if column_index < len(header) else f"列{column_index + 1}"
            if cell.strip():
                cells.append(f"{name}: {cell.strip()}")
        if cells:
            lines.append(f"第{row_index}行 " + "；".join(cells))
    text = _clean_text("\n".join(lines))
    if not text:
        raise RuntimeError(f"CSV 中没有数据行：{path.name}")
    return [Document(text=text, source=str(path), doc_id=doc_id)]


def _flatten_json(node: object, prefix: str = "") -> List[str]:
    lines: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            lines.extend(_flatten_json(value, f"{prefix}{key}."))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            lines.extend(_flatten_json(value, f"{prefix}[{index}]."))
    else:
        lines.append(f"{prefix.rstrip('.')}: {node}")
    return lines


def _load_json(path: Path, doc_id: str, encoding: Optional[str]) -> List[Document]:
    raw = read_text_file(path, encoding)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON 解析失败（{path.name}）：{exc}") from exc
    lines = _flatten_json(payload)
    text = _clean_text("\n".join(lines))
    if not text:
        raise RuntimeError(f"JSON 中没有可提取的内容：{path.name}")
    return [Document(text=text, source=str(path), doc_id=doc_id)]


def _load_plain_text(path: Path, doc_id: str, encoding: Optional[str]) -> List[Document]:
    text = _clean_text(read_text_file(path, encoding))
    if not text:
        raise RuntimeError(f"文件内容为空：{path.name}")
    return [Document(text=text, source=str(path), doc_id=doc_id)]


# --------------------------------------------------------------------- 对外 API
def load_file(path: str | Path, encoding: Optional[str] = None) -> List[Document]:
    """加载单个文件，返回按页/按块的 Document 列表。"""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在：{file_path}")
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型 {suffix}：{file_path.name}")

    digest = file_sha256(file_path)
    doc_id = digest[:12]
    meta = {
        "filename": file_path.name,
        "extension": suffix,
        "size": file_path.stat().st_size,
        "sha256": digest,
    }

    if suffix == ".pdf":
        documents = _load_pdf(file_path, doc_id)
    elif suffix == ".docx":
        documents = _load_docx(file_path, doc_id)
    elif suffix in (".html", ".htm"):
        documents = _load_html(file_path, doc_id, encoding)
    elif suffix == ".csv":
        documents = _load_csv(file_path, doc_id, encoding)
    elif suffix == ".json":
        documents = _load_json(file_path, doc_id, encoding)
    else:
        documents = _load_plain_text(file_path, doc_id, encoding)

    for document in documents:
        document.meta.update(meta)
        document.meta["sha256"] = digest
    return documents


def resolve_inputs(paths: Sequence[str | Path], recursive: bool = True) -> List[Path]:
    """把用户输入（文件/目录/通配）解析成待处理文件列表，自动跳过隐藏与临时文件。"""
    result: List[Path] = []
    seen = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            continue
        candidates: List[Path] = []
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            candidates = [p for p in path.glob(pattern) if p.is_file()]
        for candidate in candidates:
            if candidate.name.startswith((".", "~$")):
                continue
            if candidate.name.lower() in _IGNORED_FILE_NAMES:
                continue
            if any(part.lower() in _IGNORED_DIR_NAMES for part in candidate.parts[:-1]):
                continue
            if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            result.append(resolved)
    return sorted(result, key=lambda p: (p.parent.as_posix(), p.name))


def load_paths(paths: Sequence[str | Path], encoding: Optional[str] = None) -> List[Document]:
    """批量加载；单个文件失败不会影响其它文件（失败项直接跳过，需用 load_file 看具体原因）。"""
    documents: List[Document] = []
    for file_path in resolve_inputs(paths):
        documents.extend(load_file(file_path, encoding=encoding))
    return documents
