from supabase_client import supabase
from services.gemini_service import embed_text


def retrieve_relevant_chunks(query: str, match_count: int = 3) -> list[dict]:
    """
    Cari top-N chunk paling relevan dari knowledge base (kb_chunks)
    via RPC function match_kb_chunks (cosine similarity, pgvector).
    """
    query_embedding = embed_text(query)

    try:
        res = supabase.rpc(
            "match_kb_chunks",
            {"query_embedding": query_embedding, "match_count": match_count},
        ).execute()
        data = res.data or []
    except Exception as e:
        print("RAG retrieval error:", e)
        return []

    return [c for c in data if c.get("similarity", 0) > 0.5]