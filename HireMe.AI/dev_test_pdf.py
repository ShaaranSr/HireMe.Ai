#!/usr/bin/env python3
"""
Local PDF extraction test for HireMe.AI

Reads a PDF from disk, simulates a Streamlit upload object, and prints
extracted text length and a preview snippet using app.extract_text_from_upload.
"""

import sys
from pathlib import Path
from typing import Optional

from app import extract_text_from_upload


class FakeUpload:
    def __init__(self, data: bytes, name: str, mime: Optional[str] = None):
        self._data = data
        self.name = name
        self.type = mime

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data


def main():
    pdf_path = Path("SHAARAN 2025.pdf") if len(sys.argv) < 2 else Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    data = pdf_path.read_bytes()
    upload = FakeUpload(data, name=pdf_path.name, mime="application/pdf")
    text = extract_text_from_upload(upload)
    print("PDF path:", pdf_path)
    print("Extracted length:", len(text))
    # Normalize whitespace for display
    snippet = " ".join(text.split())[:500]
    print("Snippet:", snippet)


if __name__ == "__main__":
    main()



