import os
from datetime import date
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supabase_client import supabase
from services.safety_gate import safety_gate
from services.gemini_service import detect_emotion, generate_supportive_response
from services.rag_service import retrieve_relevant_chunks

load_dotenv()

app = FastAPI(title="Ruang Cerita Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti ke domain frontend kamu pas production
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    userId: str | None = None


@app.get("/")
def root():
    return {"status": "ok", "message": "Ruang Cerita backend jalan 🚀"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong.")

    user_id = req.userId or "anon"

    # 1) SAFETY GATE
    safety = safety_gate(message)

    if not safety.is_safe:
        supabase.table("conversations").insert({
            "user_id": user_id,
            "pesan_user": message,
            "emosi": None,
            "emosi_emoji": None,
            "skor_emosi": None,
            "is_safe": False,
            "safety_reason": safety.reason,
            "retrieved_chunks": None,
            "jawaban_ai": safety.escalation_message,
        }).execute()

        return {
            "safe": False,
            "reason": safety.reason,
            "reply": safety.escalation_message,
        }

    # 2) EMOTION DETECTION
    emotion = detect_emotion(message)
    emosi, emoji, skor = emotion["emosi"], emotion["emoji"], emotion["skor"]

    # 3) RAG
    chunks = retrieve_relevant_chunks(message, match_count=3)

    # 4) GENERATE RESPONSE
    reply = generate_supportive_response(message, emosi, emoji, chunks)

    # 5) SIMPAN KE DB
    try:
        saved = supabase.table("conversations").insert({
            "user_id": user_id,
            "pesan_user": message,
            "emosi": emosi,
            "emosi_emoji": emoji,
            "skor_emosi": skor,
            "is_safe": True,
            "safety_reason": None,
            "retrieved_chunks": chunks,
            "jawaban_ai": reply,
        }).execute()
        conversation_id = saved.data[0]["id"] if saved.data else None
    except Exception as e:
        print("Gagal simpan ke DB:", e)
        conversation_id = None

    return {
        "safe": True,
        "emosi": emosi,
        "emoji": emoji,
        "skor": skor,
        "retrieved_chunks": [
            {"source": c["source"], "snippet": c["content"][:120]} for c in chunks
        ],
        "reply": reply,
        "conversation_id": conversation_id,
    }


@app.get("/api/history/{user_id}")
def history(user_id: str):
    res = (
        supabase.table("conversations")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    return res.data


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)