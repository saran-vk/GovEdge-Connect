"""B1 - Speech dataset curation via edge-tts synthesis.

Synthesises short welfare-scheme utterances in en-IN / hi-IN / ta-IN with
exact ground-truth transcripts, mirroring the role real IndicVoices audio
would play. Real audio can be dropped into track_b/data/audio instead; the
benchmark only requires a matching <clip>.json transcript.

Writes:
  track_b/data/audio/<clip>.wav
  track_b/data/transcripts/<clip>.json  {"language": "...", "text": "..."}
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import edge_tts

from shared.config import load_settings, resolve
from shared.logger import get_logger

log = get_logger(__name__)

# One per language: {language: [utterance, ...]}
UTTERANCES: dict[str, list[str]] = {
    "en-IN": [
        "Is my family eligible for the PM Kisan scheme?",
        "How much money does the rural housing scheme give?",
        "Which documents are required for Ayushman Bharat?",
        "How can I check my application status online?",
        "What is the Kalaignar women assistance scheme?",
    ],
    "hi-IN": [
        "क्या मेरा परिवार पीएम किसान योजना के लिए पात्र है?",
        "आयुष्मान भारत के लिए कौन से दस्तावेज़ चाहिए?",
        "आवास योजना में कितना पैसा मिलेगा?",
        "अपने आवेदन की स्थिति कैसे जान सकते हैं?",
        "कलाईग्नर महिला सहायता योजना क्या है?",
    ],
    "ta-IN": [
        "என் குடும்பம் PM கிசான் திட்டத்திற்கு தகுதி உண்டா?",
        "ஆயுஷ்மான் பாரத் திட்டத்திற்கு என்ன ஆவணங்கள் தேவை?",
        "வீட்டுத் திட்டத்தில் எவ்வளவு பணம் கிடைக்கும்?",
        "என் விண்ணப்பத்தின் நிலையை எப்படி அறிவது?",
        "கலைஞர் மகளிர் உரிமைத் தொகை திட்டம் என்றால் என்ன?",
    ],
}


async def _synthesize(text: str, voice: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def synth_corpus() -> list[Path]:
    track_b = load_settings()["track_b"]
    cfg = track_b["speech"]
    audio_dir = resolve(track_b["audio_dir"])
    txn_dir = resolve(track_b["transcript_dir"])
    audio_dir.mkdir(parents=True, exist_ok=True)
    txn_dir.mkdir(parents=True, exist_ok=True)

    voice_map = cfg["voice_map"]
    per_lang = cfg["utterances_per_language"]

    async def _run() -> list[Path]:
        created: list[Path] = []
        for lang, utterances in UTTERANCES.items():
            voice = voice_map.get(lang)
            if voice is None:
                log.warning("no voice mapped for %s - skipping", lang)
                continue
            for i, text in enumerate(utterances[:per_lang]):
                clip_id = f"{lang.replace('-', '_')}_{i:02d}"
                wav = audio_dir / f"{clip_id}.wav"
                txn = txn_dir / f"{clip_id}.json"
                if wav.exists() and txn.exists():
                    log.info("exists - skip %s", clip_id)
                    created.append(wav)
                    continue
                await _synthesize(text, voice, wav)
                txn.write_text(
                    json.dumps({"clip_id": clip_id, "language": lang, "text": text},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                created.append(wav)
                log.info("synthesized %s (%.2f kB)", clip_id, wav.stat().st_size / 1024)
        return created

    return asyncio.run(_run())


if __name__ == "__main__":
    clips = synth_corpus()
    log.info("synth corpus ready: %d clips", len(clips))
