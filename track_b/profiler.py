"""B4 - NMT & TTS profiling harness (lightweight).

TTS : facebook/mms-tts-* (small multilingual model) - latency measured end-to-end.
NMT : ai4bharat/indictrans2-en-indic-1B (best-effort, needs HF token for gated
      repo) with facebook/nllb-200-distilled-600M as non-gated fallback.

Both are guarded so the whole Week-1 run never hard-fails on a heavy model.
Emits track_b/data/results/profiler.json and a markdown report.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from shared.config import load_settings, resolve
from shared.logger import get_logger

log = get_logger(__name__)

_SAMPLE_EN = "PM Kisan gives six thousand rupees every year to eligible farmers."
# (model, text) - each mms-tts variant is language-specific.
_TTS_SAMPLES = [
    ("facebook/mms-tts-hin", "नमस्ते, आप कैसे हैं?"),
    ("facebook/mms-tts-tam", "வணக்கம், எப்படி இருக்கிறீர்கள்?"),
    ("facebook/mms-tts-eng", "Hello, how are you?"),
]


def _measure_ms(start: float) -> float:
    return round((time.time() - start) * 1000, 1)


def profile_tts(samples: list[tuple[str, str]]) -> dict:
    from transformers import VitsModel, AutoTokenizer  # noqa: PLC0415

    results: list[dict] = []
    models: dict[str, object] = {}
    load_times: dict[str, float] = {}

    for model_name, text in samples:
        if model_name not in models:
            start = time.time()
            model = VitsModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            load_times[model_name] = _measure_ms(start)
            models[model_name] = (model, tokenizer)

        model, tokenizer = models[model_name]
        start = time.time()
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        synth_ms = _measure_ms(start)
        wav = outputs.waveform
        results.append({
            "model": model_name,
            "text": text,
            "audio_samples": int(wav.shape[-1]),
            "audio_duration_s": round(wav.shape[-1] / model.config.sampling_rate, 3),
            "synthesis_ms": synth_ms,
            "sampling_rate": getattr(model.config, "sampling_rate", None),
        })

    return {
        "models_loaded": len(models),
        "status": "ok",
        "load_ms": load_times,
        "samples": results,
    }


def profile_nmt(model_name: str) -> dict:
    """Profile an NMT model. Falls back to facebook/nllb-200-distilled-600M if the
    primary model fails (e.g., gated repo without auth token)."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        start = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        load_ms = _measure_ms(start)

        inp = tokenizer([_SAMPLE_EN], return_tensors="pt", truncation=True)
        start = time.time()
        out = model.generate(**inp, max_new_tokens=64)
        infer_ms = _measure_ms(start)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        return {
            "model": model_name,
            "status": "ok",
            "load_ms": load_ms,
            "inference_ms": infer_ms,
            "sample_in": _SAMPLE_EN,
            "sample_out": decoded,
        }
    except Exception as exc:
        log.warning("NMT model %s failed (%s); trying fallback nllb-200-distilled-600M", model_name, exc)
        return _profile_nmt_fallback()


def _profile_nmt_fallback() -> dict:
    """Fallback NMT profiling using non-gated facebook/nllb-200-distilled-600M."""
    fallback_model = "facebook/nllb-200-distilled-600M"
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

        start = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        load_ms = _measure_ms(start)

        # NLLB requires a src_lang token for translation
        tokenizer.src_lang = "eng_Latn"
        inp = tokenizer([_SAMPLE_EN], return_tensors="pt", truncation=True)
        start = time.time()
        out = model.generate(**inp, max_new_tokens=64, forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva"))
        infer_ms = _measure_ms(start)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        return {
            "model": fallback_model,
            "status": "ok (fallback)",
            "load_ms": load_ms,
            "inference_ms": infer_ms,
            "sample_in": _SAMPLE_EN,
            "sample_out": decoded,
        }
    except Exception as exc:
        log.warning("NMT fallback %s also failed: %s", fallback_model, exc)
        return {
            "model": fallback_model,
            "status": "skipped",
            "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
        }


def run_profiler() -> dict:
    cfg = load_settings()["track_b"]["profiler"]
    results_dir = resolve(load_settings()["track_b"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tts": {}, "nmt": {}}

    if cfg["tts_enabled"]:
        try:
            report["tts"] = profile_tts(_TTS_SAMPLES)
        except Exception as exc:
            log.warning("TTS profiling skipped: %s", exc)
            report["tts"] = {"model": cfg["tts_model"], "status": "skipped", "reason": str(exc)[:200]}

    if cfg["nmt_enabled"]:
        log.info("profiling NMT model: %s (if gated, will try fallback)", cfg["nmt_model"])
        report["nmt"] = profile_nmt(cfg["nmt_model"])

    (results_dir / "profiler.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "week1_profiler.md").write_text(_render_md(report), encoding="utf-8")
    log.info("profiler done: tts=%s nmt=%s", report["tts"].get("status"), report["nmt"].get("status"))
    return report


def _render_md(report: dict) -> str:
    lines = ["# Week 1 NMT/TTS Profiling", "", f"- Run: {report['timestamp']}", ""]
    tts = report.get("tts", {})
    tts_model = tts.get("models") or tts.get("model") or "n/a"
    lines.append(f"## TTS — status: **{tts.get('status')}**")
    if tts.get("status") == "ok":
        for model_name, load_ms in tts.get("load_ms", {}).items():
            lines.append(f"- Load `{model_name}`: {load_ms} ms")
        lines += ["", "| Model | Duration (s) | Synthesis (ms) |"]
        for s in tts["samples"]:
            lines.append(f"| {s['model']} | {s['audio_duration_s']} | {s['synthesis_ms']} |")
    else:
        lines.append(f"- Reason: {tts.get('reason', 'n/a')}")

    nmt = report.get("nmt", {})
    lines.append(f"\n## NMT: `{nmt.get('model')}` — status: **{nmt.get('status')}**")
    if nmt.get("status") == "ok":
        lines.append(f"- Model load: {nmt['load_ms']} ms | inference: {nmt['inference_ms']} ms")
        lines.append(f"- Sample out: {nmt.get('sample_out')}")
    else:
        lines.append(f"- Reason: {nmt.get('reason', 'n/a')}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run_profiler()
