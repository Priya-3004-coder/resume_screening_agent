"""
demo_offline.py — Runs the screening pipeline using ONLY semantic similarity
(sentence-transformers), with NO OpenAI API key required.

Use this to verify the project works locally before adding your API key.
Scores will lack LLM reasoning but the ranking and similarity scores are real.

Usage:
    python demo_offline.py
    python demo_offline.py --jd job_description.txt --resumes resumes/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

from parser import load_all_resumes
from scorer import semantic_similarity


JD_FILE     = pathlib.Path("job_description.txt")
RESUMES_DIR = pathlib.Path("resumes")
RESULTS_DIR = pathlib.Path("results")


def run_offline(jd_path: str | None = None, resumes_dir: str | None = None,
                output_dir: str | None = None) -> None:
    jd_file    = pathlib.Path(jd_path)     if jd_path     else JD_FILE
    resume_dir = pathlib.Path(resumes_dir) if resumes_dir else RESUMES_DIR
    out_dir    = pathlib.Path(output_dir)  if output_dir  else RESULTS_DIR

    if not jd_file.exists():
        print(f"ERROR: Job description file not found: {jd_file}")
        sys.exit(1)

    print("\n" + "="*60)
    print("  Resume Screening Agent — OFFLINE DEMO (no API key needed)")
    print("="*60)

    jd_text = jd_file.read_text(encoding="utf-8").strip()
    print(f"\nJob description loaded ({len(jd_text)} chars)")

    print(f"\nLoading resumes from: {resume_dir}")
    resumes = load_all_resumes(str(resume_dir))
    print(f"Loaded {len(resumes)} resumes")

    print("\nComputing semantic similarity scores…")
    results = []
    for filename, resume_text in resumes.items():
        score = semantic_similarity(jd_text, resume_text)
        results.append({"filename": filename, "semantic_score": round(score, 4)})
        print(f"  {filename:<40} {score:.4f}")

    results.sort(key=lambda x: x["semantic_score"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = out_dir / "offline_ranked.csv"
    df.to_csv(csv_path, index=False)

    json_path = out_dir / "offline_ranked.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print("  OFFLINE RANKING (semantic similarity only)")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Score':>8}  {'File'}")
    print("-" * 55)
    for r in results:
        print(f"  #{r['rank']:<3} {r['semantic_score']:>8.4f}  {r['filename']}")
    print(f"\nResults saved to: {out_dir}/")
    print("\nTo get full LLM-powered scores, run: python agent.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", default=None)
    parser.add_argument("--resumes", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_offline(args.jd, args.resumes, args.output)
