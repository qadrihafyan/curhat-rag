"""
Script ingest: pecah PDF jadi chunk teks, buat embedding, simpan ke Supabase (kb_chunks).

Cara pakai:
    1. Taruh file PDF di backend/data/ (contoh: burnout.pdf, self_compassion.pdf)
    2. Jalankan: python ingest.py
"""
import os
from pathlib import Path
from pypdf import PdfReader

from supabase_client import supabase
from services.gemini_service import embed_text

DATA_DIR = Path(__file__).parent / "data"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def ingest_file(filename: str):
    path = DATA_DIR / filename
    text = extract_pdf_text(path)
    chunks = chunk_text(text)

    print(f"📄 {filename}: {len(chunks)} chunk ditemukan")

    for idx, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        try:
            supabase.table("kb_chunks").insert({
                "source": filename,
                "content": chunk,
                "embedding": embedding,
            }).execute()
            print(f"  ✅ chunk {idx + 1}/{len(chunks)} tersimpan")
        except Exception as e:
            print(f"  ❌ gagal simpan chunk {idx}: {e}")


def main():
    if not DATA_DIR.exists():
        print(f"Folder data tidak ditemukan: {DATA_DIR}")
        return

    pdf_files = [f.name for f in DATA_DIR.glob("*.pdf")]
    if not pdf_files:
        print("Tidak ada file .pdf di backend/data/. Taruh PDF kamu di sana dulu.")
        return

    for filename in pdf_files:
        ingest_file(filename)

    print("🎉 Ingest selesai!")


if __name__ == "__main__":
    main()