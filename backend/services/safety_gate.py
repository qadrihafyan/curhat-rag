"""
Safety Gate
Cek pesan user SEBELUM masuk ke pipeline RAG + Gemini.
Kalau terdeteksi indikasi krisis, sistem TIDAK boleh hanya
mengandalkan jawaban generatif AI biasa — harus menampilkan
jalur eskalasi/sumber bantuan profesional.

Catatan: ini rule-based sederhana (cepat & murah).
Untuk produksi, sebaiknya dikombinasikan dengan classifier
yang lebih robust (bukan cuma keyword matching).
"""

from dataclasses import dataclass
from typing import Optional

CRISIS_KEYWORDS = [
    "bunuh diri",
    "mengakhiri hidup",
    "ga mau hidup",
    "gak mau hidup",
    "pengen mati",
    "ingin mati",
    "self harm",
    "menyakiti diri",
    "melukai diri",
    "gantung diri",
    "overdosis",
]

HARM_TO_OTHERS_KEYWORDS = [
    "membunuh",
    "menyakiti orang",
    "melukai dia",
]


@dataclass
class SafetyResult:
    is_safe: bool
    reason: Optional[str]
    escalation_message: Optional[str]


def safety_gate(message: str) -> SafetyResult:
    lower = message.lower()

    hits_crisis = any(kw in lower for kw in CRISIS_KEYWORDS)
    hits_harm_others = any(kw in lower for kw in HARM_TO_OTHERS_KEYWORDS)

    if hits_crisis:
        return SafetyResult(
            is_safe=False,
            reason="indikasi_krisis_diri_sendiri",
            escalation_message=(
                "Terima kasih sudah cerita, ini terdengar berat banget. Karena ini menyangkut "
                "keselamatanmu, aku ingin pastikan kamu terhubung dengan bantuan yang tepat. "
                "Kalau kamu di Indonesia, kamu bisa hubungi Layanan Sehat Jiwa Kemenkes 119 ext 8 "
                "(call/WA), atau bicara dengan orang terdekat yang kamu percaya sekarang juga. "
                "Kamu tidak harus melewati ini sendirian."
            ),
        )

    if hits_harm_others:
        return SafetyResult(
            is_safe=False,
            reason="indikasi_bahaya_ke_orang_lain",
            escalation_message=(
                "Aku dengar kamu sedang marah/terluka banget. Untuk hal yang menyangkut "
                "keselamatan orang lain, aku tidak bisa lanjut sebagai ruang curhat biasa — "
                "penting untuk bicara dengan profesional (psikolog/konselor) atau pihak "
                "berwenang kalau situasinya mendesak."
            ),
        )

    return SafetyResult(is_safe=True, reason=None, escalation_message=None)