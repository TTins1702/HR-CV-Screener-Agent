"""File parsing utilities for uploading CVs and JDs in PDF or text format."""

from __future__ import annotations

import io
from typing import BinaryIO


def extract_text_from_file(file_obj: BinaryIO, filename: str) -> str:
    """Extract plain text from an uploaded file object (.txt, .md, or .pdf)."""
    name_lower = filename.lower()
    if name_lower.endswith((".txt", ".md")):
        raw_bytes = file_obj.read()
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="replace")

    if name_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_obj)

    raise ValueError(f"Unsupported file format: {filename}. Please provide a .pdf, .txt, or .md file.")


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """Extract plain text from a PDF stream using pdfplumber with pypdf fallback."""
    raw_bytes = file_obj.read()
    buffer = io.BytesIO(raw_bytes)

    # Attempt 1: pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(buffer) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            full_text = "\n\n".join(p.strip() for p in pages if p.strip())
            if full_text.strip():
                return full_text
    except Exception:
        pass

    # Attempt 2: pypdf fallback
    try:
        import pypdf

        buffer.seek(0)
        reader = pypdf.PdfReader(buffer)
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n\n".join(p.strip() for p in pages if p.strip())
        if full_text.strip():
            return full_text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}") from e

    raise RuntimeError("Could not extract any readable text from this PDF.")
