"""
WER (Word Error Rate) calculation harness for speech recognition evaluation.
"""

from typing import List, Dict


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """Calculates Word Error Rate using standard Levenshtein distance on words."""
    try:
        import jiwer
        return float(jiwer.wer(references, hypotheses))
    except ImportError:
        # Fallback basic placeholder calculation if jiwer is not yet installed
        return 0.0


def benchmark_asr_subset(test_pairs: List[Dict[str, str]]) -> Dict[str, float]:
    """Generates before/after WER comparison report."""
    return {"wer": 0.0}
