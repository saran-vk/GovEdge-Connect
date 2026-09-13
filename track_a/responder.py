"""A5 - Template-based response generation from retrieved RAG chunks.

Takes the intent, entities, and retrieved chunks from ChromaDB and formats
a human-readable answer. This is the missing piece in the Week 1 pipeline:
ASR -> NLU -> RAG retrieval -> *response generation*.

Uses simple templates per intent; no LLM required. Designed for the Week 1
prototype where zero-hallucination is a hard constraint.
"""
from __future__ import annotations

from shared.logger import get_logger

log = get_logger(__name__)

_INTENT_TEMPLATES: dict[str, str] = {
    "check_eligibility": (
        "Based on the available information, here are the eligibility details "
        "for {scheme}:\n\n{content}\n\nIf you meet these criteria, you may be "
        "eligible to apply. Please contact your local office for confirmation."
    ),
    "scheme_benefits": (
        "Here is what you need to know about the benefits of {scheme}:\n\n{content}"
    ),
    "required_documents": (
        "To apply for {scheme}, you will typically need the following documents:\n\n"
        "{content}\n\nPlease carry original documents along with self-attested copies."
    ),
    "application_status": (
        "To check your application status for {scheme}:\n\n{content}\n\n"
        "You can also visit your nearest Common Service Centre (CSC) for assistance."
    ),
    "general_inquiry": (
        "Here is information about {scheme}:\n\n{content}\n\n"
        "For more details, you can visit the official website or call the helpline."
    ),
}

_DEFAULT_TEMPLATE = "Here is what I found about {scheme}:\n\n{content}"


def _detect_scheme_name(entities: dict[str, list[str]], chunks: list[dict]) -> str:
    """Extract a human-readable scheme name from entities or chunk sources."""
    scheme_names = entities.get("scheme_name", [])
    if scheme_names:
        return scheme_names[0].replace("-", " ").title()

    # Fallback: try to infer from chunk source metadata
    for chunk in chunks:
        source = chunk.get("source", "")
        if "pm_kisan" in source:
            return "PM-KISAN"
        if "/ayushman" in source or "pmjay" in source:
            return "Ayushman Bharat"
        if "pm_awas" in source or "pmay" in source:
            return "PM Awas Yojana"
        if "tn_" in source or "kalaignar" in source:
            return "Tamil Nadu Welfare Schemes"

    return "the requested scheme"


def _format_chunks(chunks: list[dict]) -> str:
    """Join retrieved chunks into a readable text block."""
    if not chunks:
        return "No specific information was found for your query."

    parts: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)

    return "\n\n".join(parts) if parts else "No specific information was found."


def generate_response(
    intent: str,
    entities: dict[str, list[str]],
    chunks: list[dict],
    query: str = "",
) -> str:
    """Generate a template-based response from intent, entities, and RAG chunks.

    Args:
        intent: Classified intent string (e.g. "check_eligibility").
        entities: Extracted entities (e.g. {"scheme_name": ["pm-kisan"]}).
        chunks: Retrieved ChromaDB chunks (list of dicts with "text" key).
        query: Original user query (for logging).

    Returns:
        A formatted human-readable response string.
    """
    scheme = _detect_scheme_name(entities, chunks)
    content = _format_chunks(chunks)
    template = _INTENT_TEMPLATES.get(intent, _DEFAULT_TEMPLATE)

    response = template.format(scheme=scheme, content=content)
    log.info("generated response for intent=%s scheme=%s (%d chunks)", intent, scheme, len(chunks))
    return response
