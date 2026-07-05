"""
parser.py — Extract raw text from PDF, DOCX, and plain-text resume files.
"""

import os
import pathlib


def parse_resume(file_path: str) -> str:
    """
    Read a resume file and return its full text content.

    Supported formats: .pdf, .docx, .txt, .md
    Raises ValueError for unsupported formats.
    """
    path = pathlib.Path(file_path)
    ext = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")

    if ext == ".txt" or ext == ".md":
        return _read_text(path)
    elif ext == ".pdf":
        return _read_pdf(path)
    elif ext == ".docx":
        return _read_docx(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}' for file: {file_path}")


def _read_text(path: pathlib.Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def _read_pdf(path: pathlib.Path) -> str:
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF parsing. Run: pip install PyPDF2")

    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts).strip()


def _read_docx(path: pathlib.Path) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX parsing. Run: pip install python-docx")

    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def load_all_resumes(folder_path: str) -> dict[str, str]:
    """
    Load all resumes from a folder.
    Returns a dict mapping filename -> extracted text.
    """
    folder = pathlib.Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {folder_path}")

    supported = {".txt", ".md", ".pdf", ".docx"}
    resumes = {}

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in supported:
            try:
                text = parse_resume(str(file))
                if text:
                    resumes[file.name] = text
                    print(f"  ✓ Loaded: {file.name} ({len(text)} chars)")
                else:
                    print(f"  ⚠ Empty content: {file.name} — skipped")
            except Exception as e:
                print(f"  ✗ Failed to parse {file.name}: {e}")

    return resumes
