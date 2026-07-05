# Resume Screening Agent
**Rooman AI Challenge — Junior AI Research Associate**

An end-to-end AI agent that ranks a batch of resumes against a job description and outputs a scored, ordered shortlist with reasoning.

---

## What It Does

```
Job Description + Folder of Resumes
          ↓
  [1] Parse PDFs / DOCX / TXT
          ↓
  [2] Semantic similarity (sentence-transformers)
          ↓
  [3] LLM structured evaluation (GPT-4o-mini)
          ↓
  [4] Weighted final score (0–100)
          ↓
Ranked CSV + JSON + Console summary
```

---

## Quick Start

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd resume_screener
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.10+ recommended. Uses `sentence-transformers`, `openai`, `PyPDF2`, `python-docx`, `pandas`.

### 3. Add your LLM API key

**Option A — OpenAI** (requires billing credit at [platform.openai.com/settings/billing](https://platform.openai.com/settings/billing)):
```bash
cp .env.example .env
# Add to .env:
# OPENAI_API_KEY=sk-...
```

**Option B — Groq (FREE)** — recommended if your OpenAI account has no credits:
1. Sign up at [console.groq.com](https://console.groq.com) (free, no credit card)
2. Go to **API Keys** → **Create API Key**
3. Add to `.env`:
```
GROQ_API_KEY=gsk_your-key-here
LLM_PROVIDER=groq
```
That's it. Groq's `llama-3.1-8b-instant` is fast, free, and works great for this task.

### 4. Run the agent

```bash
python agent.py
```

**Custom paths:**
```bash
python agent.py --jd my_jd.txt --resumes my_resumes/ --output my_results/
```

### 5. View results

```
results/
  ranked_candidates.csv   ← spreadsheet with all scores + reasoning
  ranked_candidates.json  ← full structured output with lists
```

---

## No API Key? Run the Offline Demo

Uses only semantic similarity (no OpenAI calls needed):

```bash
python demo_offline.py
```

This still produces a meaningful ranking — it just won't have LLM reasoning text.

---

## Project Structure

```
resume_screener/
├── agent.py              # Main pipeline — run this
├── parser.py             # PDF / DOCX / TXT text extraction
├── scorer.py             # Semantic similarity + LLM evaluation + scoring
├── demo_offline.py       # No-API-key demo using only embeddings
├── job_description.txt   # Sample JD (Junior AI/ML Engineer)
├── requirements.txt
├── .env.example
├── resumes/              # 10 sample resumes (varied profiles)
│   ├── 01_arjun_sharma.txt
│   ├── 02_priya_menon.txt
│   └── ... (10 total)
└── results/
    ├── sample_ranked_candidates.csv    ← pre-generated sample output
    └── sample_ranked_candidates.json
```

---

## Scoring Method

Every resume is scored out of **100** using two complementary methods:

### Stage 1 — Semantic Similarity (40% weight)
- Model: `all-MiniLM-L6-v2` (sentence-transformers)
- Computes cosine similarity between JD embedding and resume embedding
- Fast, objective, language-agnostic baseline
- Score: `cosine_similarity × 100 × 0.40` → contributes 0–40 points

### Stage 2 — LLM Structured Evaluation (60% weight)
- Model: `gpt-4o-mini` (configurable via `.env`)
- The LLM scores 4 dimensions with a calibrated rubric:

| Dimension | Max Points | What it measures |
|---|---|---|
| Skills Match | 25 | Required + preferred skill overlap |
| Experience Relevance | 25 | How past roles map to this position |
| Education Fit | 15 | Degree, field, certifications |
| Overall Impression | 35 | Trajectory, communication, culture |

- LLM total (0–100) × 0.60 → contributes 0–60 points
- Also returns: `strengths[]`, `gaps[]`, `summary` (human-readable reasoning)

### Final Formula
```
final_score = (semantic_cosine × 100 × 0.40) + (llm_total × 0.60)
```

The semantic score prevents the LLM from being fooled by fluent-but-irrelevant resumes. The LLM adds nuance that keyword matching misses.

---

## Sample Output

Pre-generated results for the included sample data are in `results/`:

| Rank | Candidate | Score | Notes |
|---|---|---|---|
| 1 | 04_sneha_iyer | 91.2 | M.Tech AI, LLM thesis, production RAG on GCP |
| 2 | 08_riya_kapoor | 88.7 | AI specialisation, Ola Electric LLM internship |
| 3 | 01_arjun_sharma | 84.1 | IIT Hyderabad, LangChain RAG project, AWS |
| 4 | 09_vivek_singh | 72.4 | Solid NLP, no LLM/RAG experience yet |
| 5 | 06_divya_nair | 67.8 | Computer vision strength, weak on NLP/LLM |
| 6 | 02_priya_menon | 58.3 | Data analyst profile, not ML engineer |
| 7 | 10_meera_joshi | 38.5 | Strong maths, very early in ML journey |
| 8 | 05_karan_patel | 32.1 | Early learner, no substantial projects yet |
| 9 | 03_rohan_verma | 22.7 | Strong backend dev, wrong role entirely |
| 10 | 07_amit_desai | 18.4 | Finance career-changer, no tech output yet |

---

## Adding Your Own Resumes

1. Drop `.pdf`, `.docx`, or `.txt` files into the `resumes/` folder
2. Edit `job_description.txt` with your actual JD
3. Run `python agent.py`

Supported formats: **PDF**, **DOCX**, **TXT**, **MD**

---

## Design Tradeoffs & Limitations

### What works well
- **Two-stage scoring is more robust than either method alone.** Semantic similarity catches vocabulary overlap; LLM evaluation catches career narrative and reasoning.
- **Structured JSON output from the LLM** (`response_format: json_object`) avoids parsing fragility.
- **Retry logic** handles transient API failures gracefully.
- **Offline demo** lets reviewers verify the pipeline without spending money.

### Known limitations
1. **PDF parsing quality varies.** `PyPDF2` struggles with multi-column, scanned, or image-based PDFs. For production, `pdfminer.six` or a document AI service would be better.
2. **LLM scores are not perfectly calibrated.** The model may rate candidates generously or inconsistently across runs. Mitigation: low temperature (0.2), explicit rubric in system prompt.
3. **Context window cap.** Very long resumes (>8k tokens) may be truncated. A production system would chunk and summarise.
4. **No true RAG over JD skills.** A future improvement would extract a structured skills list from the JD first, then compare against each resume's extracted skills — more precise than full-text embedding.
5. **English-only.** Resumes in other languages will get low scores regardless of quality.
6. **Cost.** Roughly $0.001–0.005 per resume at gpt-4o-mini rates. For 100+ resumes, use batching.

### What I'd add with more time
- PDF extraction upgrade (pdfminer or AWS Textract)
- Streamlit UI with drag-and-drop resume upload
- Structured skills extraction from JD before scoring
- Batch API calls for cost efficiency
- Confidence intervals on scores (run each resume twice, report variance)
- Support for multi-language resumes

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | If using OpenAI | — | Your OpenAI API key |
| `GROQ_API_KEY` | If using Groq | — | Your Groq API key (free at console.groq.com) |
| `LLM_PROVIDER` | No | auto-detect | Set to `groq` to use Groq; leave blank for OpenAI |
| `OPENAI_MODEL` | No | `gpt-4o-mini` / `llama-3.1-8b-instant` | Model name (auto-defaulted per provider) |

---

## Requirements

```
openai>=1.0.0
groq>=0.9.0
python-dotenv>=1.0.0
PyPDF2>=3.0.0
python-docx>=1.0.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```
