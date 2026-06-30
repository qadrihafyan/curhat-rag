import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

chat_model = genai.GenerativeModel("gemini-flash-latest")
EMBED_MODEL = "models/text-embedding-004"


def detect_emotion(user_message: str) -> dict:
    """
    Minta Gemini mengklasifikasikan emosi user ke kategori + skor 1-5.
    Skor: 1 = sangat negatif/berat, 5 = sangat positif.
    """
    prompt = f"""
Kamu adalah classifier emosi. Baca pesan user di bawah, lalu balas HANYA dalam format JSON murni (tanpa markdown, tanpa penjelasan tambahan):

{{
  "emosi": "<satu kata, contoh: Lelah / Sedih / Cemas / Marah / Senang / Bingung / Kecewa>",
  "emoji": "<satu emoji yang merepresentasikan emosi itu>",
  "skor": <angka 1-5, 1=sangat berat/negatif, 5=sangat positif>
}}

Pesan user: "{user_message}"
""".strip()

    result = chat_model.generate_content(prompt)
    text = result.text.strip()

    try:
        cleaned = re.sub(r"```json|```", "", text).strip()
        parsed = json.loads(cleaned)
        return {
            "emosi": parsed.get("emosi", "Tidak diketahui"),
            "emoji": parsed.get("emoji", "🙂"),
            "skor": int(parsed.get("skor", 3)),
        }
    except Exception:
        return {"emosi": "Tidak diketahui", "emoji": "🙂", "skor": 3}


def embed_text(text: str) -> list[float]:
    """Embedding untuk RAG (query & ingest chunk), dipaksa 768 dimensi
    biar cocok dengan kolom vector(768) di Supabase."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        output_dimensionality=768,
    )
    return result["embedding"]


def generate_supportive_response(user_message: str, emosi: str, emoji: str, context_chunks: list[dict]) -> str:
    konteks = "\n".join(
        f"({i + 1}) [sumber: {c['source']}] {c['content']}"
        for i, c in enumerate(context_chunks)
    )

    prompt = f"""
Kamu adalah teman suportif di ruang curhat bernama "Ruang Cerita". Tugasmu mendengarkan dan memvalidasi
perasaan user, BUKAN menggurui atau memberi ceramah panjang. Gunakan bahasa hangat, natural, dan singkat
(maksimal 4-5 kalimat). Boleh sisipkan emoji secukupnya sesuai gaya emosi user.

GAYA EMOSI USER SAAT INI: {emosi} {emoji}

KONTEKS RELEVAN (dari materi self-help, gunakan kalau memang nyambung, jangan dipaksakan,
jangan sebut "menurut dokumen" — sampaikan secara natural seolah saran sendiri):
{konteks or "(tidak ada konteks relevan)"}

PESAN USER:
"{user_message}"

Balas sebagai teman suportif:
""".strip()

    result = chat_model.generate_content(prompt)
    return result.text.strip()