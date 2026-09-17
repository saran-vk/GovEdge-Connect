"""B3 - Baseline WER/CER benchmarking with jiwer.

For every <clip>.wav in the audio dir, transcribe and compare against the
ground-truth <clip>.json transcript. Emits:

  track_b/data/results/wer_baseline.json         (machine-readable)
  track_b/data/results/week1_wer_baseline.md     (report)

Per-language and overall WER/CER are written for the conclusions generator.
"""
from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from jiwer import cer, wer, transforms as tr

from shared.config import load_settings, resolve
from shared.logger import get_logger

from track_b.asr import ASR

log = get_logger(__name__)


class NFKC(tr.AbstractTransform):
    """Unicode NFKC normalization - folds Indic orthographic variants
    (devanagari/tamil conjuncts, zero-width joiner/non-joiner) so WER/CER are
    not inflated by representation, not content."""

    def process_string(self, s: str) -> str:
        return unicodedata.normalize("NFKC", s)

    def process_list(self, inp: list[str]) -> list[str]:
        return [self.process_string(s) for s in inp]


# jiwer 3.x does not normalize by default - apply explicit fair transforms so
# case/punctuation do not inflate WER/CER. Pipelines must end with a
# ReduceToListOfListOf* transform (see jiwer.Compose docs).
_NORMALIZE = [NFKC(), tr.RemoveMultipleSpaces(), tr.Strip(), tr.RemovePunctuation(),
              tr.ToLowerCase(), tr.ExpandCommonEnglishContractions()]
_WORD_STANDARD = tr.Compose([*_NORMALIZE, tr.ReduceToListOfListOfWords()])
_CHAR_STANDARD = tr.Compose([*_NORMALIZE, tr.ReduceToListOfListOfChars()])


def _wer(ref: str, hyp: str) -> float:
    return wer(ref, hyp, reference_transform=_WORD_STANDARD, hypothesis_transform=_WORD_STANDARD)


def _cer(ref: str, hyp: str) -> float:
    return cer(ref, hyp, reference_transform=_CHAR_STANDARD, hypothesis_transform=_CHAR_STANDARD)


def _load_transcripts(txn_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for txn in txn_dir.glob("*.json"):
        data = json.loads(txn.read_text(encoding="utf-8"))
        clip_id = data.get("clip_id", txn.stem)
        out[clip_id] = data
    return out


def run_benchmark(asr: ASR | None = None,
                  audio_dir: str | Path | None = None,
                  txn_dir: str | Path | None = None,
                  results_dir: str | Path | None = None) -> dict:
    cfg = load_settings()["track_b"]
    audio_dir = Path(audio_dir) if audio_dir else resolve(cfg["audio_dir"])
    txn_dir = Path(txn_dir) if txn_dir else resolve(cfg["transcript_dir"])
    results_dir = Path(results_dir) if results_dir else resolve(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    asr = asr or ASR()
    transcripts = _load_transcripts(txn_dir)

    clips = sorted(audio_dir.glob("*.wav"))
    if not clips:
        log.warning("no audio clips found in %s", audio_dir)
        return {}

    rows: list[dict] = []
    per_lang: dict[str, dict] = defaultdict(lambda: {"wer": [], "cer": [], "n": 0})

    for wav in clips:
        clip_id = wav.stem
        gt = transcripts.get(clip_id)
        if gt is None:
            log.warning("no transcript for %s - skipping", clip_id)
            continue
        # Pin the language from ground truth (kills auto-detect misfires on
        # short clips, which previously inflated WER and 'hallucinated' flags).
        pred = asr.transcribe(wav, language=gt["language"][:2].lower())
        ref, hyp = gt["text"], pred.transcript
        w = _wer(ref, hyp) if (ref and hyp) else 1.0
        c = _cer(ref, hyp) if (ref and hyp) else 1.0
        lang_mismatch = pred.detected_language[:2].lower() != gt["language"][:2].lower()
        hallucinated = w > 1.0 or lang_mismatch
        rows.append({
            "clip_id": clip_id,
            "language": gt["language"],
            "reference": ref,
            "hypothesis": hyp,
            "wer": round(w, 4),
            "cer": round(c, 4),
            "confidence": round(pred.confidence_score, 4),
            "detected_language": pred.detected_language,
            "hallucinated": hallucinated,
        })
        per_lang[gt["language"]]["wer"].append(w)
        per_lang[gt["language"]]["cer"].append(c)
        per_lang[gt["language"]]["n"] += 1
        log.info("%s | WER=%.2f CER=%.2f | %s", clip_id, w, c, hyp)

    lang_stats = {
        lang: {
            "wer": round(sum(v["wer"]) / v["n"], 4),
            "cer": round(sum(v["cer"]) / v["n"], 4),
            "n": v["n"],
        }
        for lang, v in per_lang.items()
    }

    all_w = [r["wer"] for r in rows]
    all_c = [r["cer"] for r in rows]
    clean_w = [r["wer"] for r in rows if not r["hallucinated"]]
    clean_c = [r["cer"] for r in rows if not r["hallucinated"]]
    result = {
        "asr_model": asr.model_size,
        "asr_device": asr.device,
        "asr_compute_type": asr.compute_type,
        "n_clips": len(rows),
        "n_hallucinated": sum(1 for r in rows if r["hallucinated"]),
        "overall_wer": round(sum(all_w) / len(all_w), 4) if all_w else None,
        "overall_cer": round(sum(all_c) / len(all_c), 4) if all_c else None,
        "overall_wer_ex_hallucinations": round(sum(clean_w) / len(clean_w), 4) if clean_w else None,
        "overall_cer_ex_hallucinations": round(sum(clean_c) / len(clean_c), 4) if clean_c else None,
        "per_language": lang_stats,
        "rows": rows,
    }

    if not rows:
        log.warning("no benchmarkable clips (missing transcripts?)")
        return result

    (results_dir / "wer_baseline.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "week1_wer_baseline.md").write_text(
        _render_md(result), encoding="utf-8")
    log.info("benchmark written: overall WER=%.2f CER=%.2f over %d clips",
             result["overall_wer"], result["overall_cer"], len(rows))
    return result


def _render_md(result: dict) -> str:
    lines = [
        "# Week 1 WER Baseline",
        "",
        f"- ASR: faster-whisper `{result['asr_model']}` on `{result['asr_device']}` "
        f"({result['asr_compute_type']})",
        f"- Clips benchmarked: {result['n_clips']} "
        f"({result['n_hallucinated']} flagged as hallucinated)",
        f"- **Overall WER: {result['overall_wer']:.2%}** | "
        f"**Overall CER: {result['overall_cer']:.2%}**",
        f"- Overall WER excluding hallucinations: "
        f"{result['overall_wer_ex_hallucinations']:.2%}",
        "",
        "| Language | N | WER | CER |",
        "| :-- | --: | --: | --: |",
    ]
    for lang, s in result["per_language"].items():
        lines.append(f"| {lang} | {s['n']} | {s['wer']:.2%} | {s['cer']:.2%} |")
    lines += ["", "## Per-clip detail", "", "| Clip | Lang | WER | CER | Confidence | Hypothesis |"]
    for r in result["rows"]:
        hyp = r["hypothesis"][:60].replace("|", "/")
        lines.append(
            f"| {r['clip_id']} | {r['language']} | {r['wer']:.2%} | {r['cer']:.2%} "
            f"| {r['confidence']:.2f} | {hyp} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run_benchmark()
