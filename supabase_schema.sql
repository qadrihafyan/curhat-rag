-- =========================================================
-- RUANG CERITA — Supabase Schema
-- Jalankan di Supabase SQL Editor (Project > SQL Editor > New query)
-- =========================================================

create extension if not exists vector;

create table if not exists kb_chunks (
  id uuid primary key default gen_random_uuid(),
  source text not null,
  content text not null,
  embedding vector(768),
  created_at timestamptz default now()
);

create index if not exists kb_chunks_embedding_idx
  on kb_chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

create table if not exists conversations (
  id uuid primary key default gen_random_uuid(),
  user_id text,
  tanggal date default current_date,
  pesan_user text not null,
  emosi text,
  emosi_emoji text,
  skor_emosi int,
  is_safe boolean default true,
  safety_reason text,
  retrieved_chunks jsonb,
  jawaban_ai text,
  created_at timestamptz default now()
);

create index if not exists conversations_user_idx on conversations (user_id, created_at desc);

create or replace function match_kb_chunks (
  query_embedding vector(768),
  match_count int default 3
)
returns table (
  id uuid,
  source text,
  content text,
  similarity float
)
language sql stable
as $$
  select
    kb_chunks.id,
    kb_chunks.source,
    kb_chunks.content,
    1 - (kb_chunks.embedding <=> query_embedding) as similarity
  from kb_chunks
  order by kb_chunks.embedding <=> query_embedding
  limit match_count;
$$;