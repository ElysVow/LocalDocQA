"""检索质量评估：用固定问题集量化 hit@k 与关键词召回率。

用法：
    python scripts/eval_rag.py --collection demo
    python scripts/eval_rag.py --collection demo -k 3 --out docs/eval-report.json
    python scripts/eval_rag.py --collection demo --with-answer      # 同时评估回答关键词命中

评估文件格式（JSONL，每行一个问题）::

    {"question": "...", "keywords": ["..."], "expected_source": "02-计算机网络笔记.md"}

指标：
    hit@1    —— 第一条命中是否来自期望文档
    hit@k    —— 前 k 条里是否有期望文档
    kw       —— 期望关键词在召回文本中的覆盖率
    ans      —— （可选）期望关键词在生成回答中的覆盖率
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localdocqa.config import Config  # noqa: E402
from localdocqa.logging_utils import setup_logging  # noqa: E402
from localdocqa.rag_chain import RAGEngine  # noqa: E402


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON：{exc}") from exc
            if not case.get("question"):
                raise ValueError(f"第 {line_number} 行缺少 question 字段")
            cases.append(case)
    return cases


def keyword_coverage(keywords: List[str], text: str) -> float:
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword.lower() in text.lower())
    return hits / len(keywords)


def evaluate(cases: List[Dict[str, Any]], engine: RAGEngine, top_k: int, with_answer: bool) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        question = case["question"]
        expected = case.get("expected_source", "")
        keywords = list(case.get("keywords", []))

        hits = engine.retrieve(question, top_k=top_k)
        filenames = [hit.chunk.filename for hit in hits]
        merged_text = "\n".join(hit.chunk.text for hit in hits)

        row: Dict[str, Any] = {
            "question": question,
            "expected": expected,
            "top": filenames[0] if filenames else "-",
            "hit@1": bool(expected) and bool(filenames) and filenames[0] == expected,
            "hit@k": bool(expected) and expected in filenames,
            "kw_recall": keyword_coverage(keywords, merged_text),
            "answer_recall": None,
        }
        if with_answer:
            answer = engine.answer(question)
            row["answer_recall"] = keyword_coverage(keywords, answer.text)
        rows.append(row)

    with_expected = [row for row in rows if row["expected"]] or rows
    summary = {
        "questions": len(rows),
        "hit@1": round(mean(1.0 if row["hit@1"] else 0.0 for row in with_expected), 3),
        "hit@k": round(mean(1.0 if row["hit@k"] else 0.0 for row in with_expected), 3),
        "kw_recall": round(mean(row["kw_recall"] for row in rows), 3),
    }
    if with_answer:
        summary["answer_recall"] = round(mean(row["answer_recall"] or 0.0 for row in rows), 3)
    return {"summary": summary, "rows": rows, "top_k": top_k}


def print_report(result: Dict[str, Any]) -> None:
    print(f"评估集：{result['summary']['questions']} 个问题，top_k={result['top_k']}")
    print(f"{'问题':<34}{'hit@1':>7}{'hit@k':>7}{'关键词':>8}{'回答':>8}")
    for row in result["rows"]:
        answer_cell = "-" if row["answer_recall"] is None else f"{row['answer_recall']:.2f}"
        print(
            f"{row['question'][:32]:<34}"
            f"{'√' if row['hit@1'] else '×':>6}"
            f"{'√' if row['hit@k'] else '×':>7}"
            f"{row['kw_recall']:>8.2f}"
            f"{answer_cell:>8}"
        )
    summary = result["summary"]
    print("\n汇总：")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LocalDocQA 检索质量评估")
    parser.add_argument("--eval-file", default=str(ROOT / "examples" / "eval_set.jsonl"))
    parser.add_argument("-c", "--collection", default="demo")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--with-answer", action="store_true", help="同时评估生成回答（会调用模型）")
    parser.add_argument("--out", help="把结果写入 JSON 文件")
    parser.add_argument("--min-hit-k", type=float, default=0.0, help="hit@k 低于该值则以非零码退出")
    args = parser.parse_args()

    setup_logging("WARNING")
    config = Config.load()
    cases = load_cases(Path(args.eval_file))
    engine = RAGEngine(config, collection=args.collection, llm=None)
    result = evaluate(cases, engine, args.top_k, args.with_answer)
    print_report(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入：{out_path}")

    if args.min_hit_k and result["summary"]["hit@k"] < args.min_hit_k:
        print(f"\n[失败] hit@k={result['summary']['hit@k']} 低于阈值 {args.min_hit_k}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
