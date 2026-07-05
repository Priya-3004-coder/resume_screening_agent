"""
agent.py — Orchestrates the full resume screening pipeline.

Flow:
  1. Load JD text
  2. Load all resumes from the resumes/ folder
  3. For each resume:
     a. Compute semantic similarity vs JD
     b. Call LLM for structured evaluation
     c. Compute final weighted score
  4. Sort candidates by final score (descending)
  5. Save ranked output to results/ranked_candidates.csv and .json
  6. Print a pretty summary table to stdout
"""

from __future__ import annotations

import json
import os
import sys
import time
import pathlib
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from parser import load_all_resumes, parse_resume
from scorer import semantic_similarity, llm_evaluate, compute_final_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "").lower()  # "groq" or "openai" (auto-detected if blank)

RESULTS_DIR    = pathlib.Path("results")
RESUMES_DIR    = pathlib.Path("resumes")
JD_FILE        = pathlib.Path("job_description.txt")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_screening(
    jd_path: str | None = None,
    resumes_dir: str | None = None,
    output_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Full screening pipeline. Returns the ranked list of candidate dicts.
    """
    # --- Setup paths -------------------------------------------------------
    jd_file     = pathlib.Path(jd_path)      if jd_path      else JD_FILE
    resume_dir  = pathlib.Path(resumes_dir)  if resumes_dir  else RESUMES_DIR
    out_dir     = pathlib.Path(output_dir)   if output_dir   else RESULTS_DIR

    # --- Validate inputs ---------------------------------------------------
    if not jd_file.exists():
        print(f"ERROR: Job description file not found: {jd_file}")
        sys.exit(1)

    # Auto-detect provider: use Groq if GROQ_API_KEY is set and provider not forced to openai
    use_groq = (LLM_PROVIDER == "groq") or (bool(GROQ_API_KEY) and LLM_PROVIDER != "openai")

    if use_groq:
        if not GROQ_API_KEY:
            print("ERROR: LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
            sys.exit(1)
        model = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant")
    else:
        if not OPENAI_API_KEY:
            print("ERROR: No LLM API key found.")
            print("  Option 1 (OpenAI): Add billing at https://platform.openai.com/settings/billing")
            print("  Option 2 (FREE):   Sign up at https://console.groq.com → get GROQ_API_KEY")
            print("                     Add to .env:  GROQ_API_KEY=gsk_...")
            print("                                   LLM_PROVIDER=groq")
            sys.exit(1)
        model = OPENAI_MODEL

    # --- Load JD -----------------------------------------------------------
    print(f"\n{'='*60}")
    print("  ROOMAN AI CHALLENGE — Resume Screening Agent")
    print(f"{'='*60}")
    print(f"\n[1/4] Loading job description from: {jd_file}")
    jd_text = jd_file.read_text(encoding="utf-8").strip()
    print(f"  JD length: {len(jd_text)} characters")

    # --- Load resumes ------------------------------------------------------
    print(f"\n[2/4] Loading resumes from: {resume_dir}")
    resumes = load_all_resumes(str(resume_dir))
    if not resumes:
        print("ERROR: No resumes found. Add .txt/.pdf/.docx files to the resumes/ folder.")
        sys.exit(1)
    print(f"  Total resumes loaded: {len(resumes)}")

    # --- Score each resume -------------------------------------------------
    provider_label = f"Groq ({model})" if use_groq else f"OpenAI ({model})"
    print(f"\n[3/4] Scoring resumes (provider: {provider_label})…")

    if use_groq:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)

    candidates = []
    for i, (filename, resume_text) in enumerate(resumes.items(), 1):
        print(f"\n  [{i}/{len(resumes)}] {filename}")

        # Semantic similarity
        sem_score = semantic_similarity(jd_text, resume_text)
        print(f"    Semantic similarity : {sem_score:.3f}")

        # LLM evaluation (with simple retry)
        llm_result = None
        for attempt in range(3):
            try:
                llm_result = llm_evaluate(jd_text, resume_text, client, model)
                break
            except Exception as e:
                print(f"    LLM attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)

        if llm_result is None:
            print(f"    WARNING: LLM evaluation failed for {filename}. Using zeros.")
            llm_result = {
                "skills_match": 0, "experience_relevance": 0,
                "education_fit": 0, "overall_impression": 0,
                "strengths": [], "gaps": ["Evaluation failed"],
                "summary": "Could not evaluate this resume."
            }

        # Final score
        final = compute_final_score(sem_score, llm_result)
        print(f"    LLM sub-scores      : skills={llm_result['skills_match']} "
              f"exp={llm_result['experience_relevance']} "
              f"edu={llm_result['education_fit']} "
              f"overall={llm_result['overall_impression']}")
        print(f"    FINAL SCORE         : {final}/100")

        candidates.append({
            "rank":                  0,              # filled after sorting
            "filename":              filename,
            "final_score":           final,
            "semantic_similarity":   round(sem_score, 4),
            "skills_match":          llm_result["skills_match"],
            "experience_relevance":  llm_result["experience_relevance"],
            "education_fit":         llm_result["education_fit"],
            "overall_impression":    llm_result["overall_impression"],
            "strengths":             "; ".join(llm_result.get("strengths", [])),
            "gaps":                  "; ".join(llm_result.get("gaps", [])),
            "summary":               llm_result.get("summary", ""),
        })

    # --- Rank & save -------------------------------------------------------
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, c in enumerate(candidates, 1):
        c["rank"] = rank

    print(f"\n[4/4] Saving results to: {out_dir}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    df = pd.DataFrame(candidates)
    csv_path = out_dir / "ranked_candidates.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV  : {csv_path}")

    # Rebuild JSON from candidates with proper list fields
    json_out = []
    for c in candidates:
        json_out.append({
            "rank":                 c["rank"],
            "filename":             c["filename"],
            "final_score":          c["final_score"],
            "semantic_similarity":  c["semantic_similarity"],
            "llm_scores": {
                "skills_match":         c["skills_match"],
                "experience_relevance": c["experience_relevance"],
                "education_fit":        c["education_fit"],
                "overall_impression":   c["overall_impression"],
            },
            "strengths": [s.strip() for s in c["strengths"].split(";") if s.strip()],
            "gaps":      [g.strip() for g in c["gaps"].split(";") if g.strip()],
            "summary":   c["summary"],
        })

    json_path = out_dir / "ranked_candidates.json"
    json_path.write_text(json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ JSON : {json_path}")

    # --- Print ranked summary ---------------------------------------------
    _print_summary(candidates)

    return json_out


def _print_summary(candidates: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("  RANKED SHORTLIST")
    print(f"{'='*60}")
    header = f"{'Rank':<5} {'Score':>6}  {'File':<35} {'Summary'}"
    print(header)
    print("-" * 90)
    for c in candidates:
        summary_short = c["summary"][:60] + "…" if len(c["summary"]) > 60 else c["summary"]
        print(f"  #{c['rank']:<3} {c['final_score']:>6.1f}  {c['filename']:<35} {summary_short}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Resume Screening Agent — ranks resumes against a job description."
    )
    parser.add_argument(
        "--jd",
        default=str(JD_FILE),
        help=f"Path to job description text file (default: {JD_FILE})",
    )
    parser.add_argument(
        "--resumes",
        default=str(RESUMES_DIR),
        help=f"Path to folder containing resume files (default: {RESUMES_DIR})",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR),
        help=f"Directory to save CSV/JSON output (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()

    run_screening(
        jd_path=args.jd,
        resumes_dir=args.resumes,
        output_dir=args.output,
    )
