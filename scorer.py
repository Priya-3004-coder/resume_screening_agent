"""
scorer.py — NLP similarity scoring between resumes and a job description.

Two-stage scoring:
  1. Semantic similarity  (sentence-transformers cosine similarity) — 40 % weight
  2. LLM structured evaluation (OpenAI or Groq) — 60 % weight
     Breaks down into: skills match, experience relevance, education fit, overall

The semantic score gives a fast, objective baseline.
The LLM score adds nuanced reasoning and outputs human-readable rationale.
"""

from __future__ import annotations
import json
import os
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Semantic similarity (sentence-transformers)
# ---------------------------------------------------------------------------

_embed_model = None  # lazy-loaded


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading embedding model (all-MiniLM-L6-v2)…")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def semantic_similarity(jd_text: str, resume_text: str) -> float:
    """
    Returns a cosine similarity score in [0, 1] between the JD and resume.
    Uses sentence-transformers 'all-MiniLM-L6-v2' (small, fast, good quality).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    model = _get_embed_model()
    embeddings = model.encode([jd_text, resume_text], convert_to_numpy=True)
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    # Clamp to [0, 1] — cosine can technically go slightly negative
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# LLM evaluation (OpenAI)
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = """
You are an expert technical recruiter. You will be given a Job Description (JD)
and a candidate's resume. Evaluate the resume against the JD and return ONLY a
valid JSON object — no markdown, no extra text — with exactly this structure:

{
  "skills_match":          <integer 0–25>,
  "experience_relevance":  <integer 0–25>,
  "education_fit":         <integer 0–15>,
  "overall_impression":    <integer 0–35>,
  "strengths":             [<string>, ...],
  "gaps":                  [<string>, ...],
  "summary":               "<2–3 sentence rationale>"
}

Scoring guide:
  skills_match         — How well the candidate's skills match the required/preferred skills.
  experience_relevance — How relevant their past roles and projects are to this position.
  education_fit        — Degree, certifications, and field alignment.
  overall_impression   — Holistic gut-check: communication, career trajectory, culture fit signals.

Be honest and calibrated. A perfect score is rare.
""".strip()


def llm_evaluate(
    jd_text: str,
    resume_text: str,
    client,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Ask the LLM to evaluate a single resume against the JD.
    Works with both OpenAI and Groq clients (both use the same SDK interface).
    Returns a dict with scores and reasoning.
    """
    user_content = (
        f"=== JOB DESCRIPTION ===\n{jd_text}\n\n"
        f"=== CANDIDATE RESUME ===\n{resume_text}"
    )

    # Groq models don't support response_format=json_object for all models,
    # so we add a JSON reminder to the user message as a fallback.
    is_groq_model = any(name in model for name in ["llama-", "llama3", "mixtral", "gemma", "qwen", "deepseek"])
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_content + ("\n\nRemember: respond with ONLY valid JSON, no other text." if is_groq_model else ""),
            },
        ],
        temperature=0.2,
    )
    # Only pass response_format for OpenAI-compatible models
    if not is_groq_model:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wrapped the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw output:\n{raw}")

    # Validate expected keys exist
    required = ["skills_match", "experience_relevance", "education_fit",
                "overall_impression", "strengths", "gaps", "summary"]
    for key in required:
        if key not in result:
            result[key] = 0 if "match" in key or "relevance" in key or "fit" in key or "impression" in key else []

    return result


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------

def compute_final_score(
    semantic_score: float,
    llm_result: dict[str, Any],
    semantic_weight: float = 0.40,
) -> float:
    """
    Combine semantic similarity and LLM sub-scores into a final score out of 100.

    Weights:
      40 % — semantic cosine similarity (scaled to 0–40)
      60 % — LLM evaluation (skills + experience + education + overall = 0–100, scaled to 0–60)
    """
    llm_raw = (
        llm_result.get("skills_match", 0)
        + llm_result.get("experience_relevance", 0)
        + llm_result.get("education_fit", 0)
        + llm_result.get("overall_impression", 0)
    )  # max = 100

    llm_component = llm_raw * (1 - semantic_weight)        # 0–60
    semantic_component = semantic_score * 100 * semantic_weight  # 0–40

    return round(llm_component + semantic_component, 2)
