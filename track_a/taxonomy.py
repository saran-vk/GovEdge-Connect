"""A3 - Intent & entity taxonomy for the welfare-assistant NLU engine.

Intents follow the Week 1 plan's five query categories; entities follow the
annotated regional user-input fields. Includes annotation guidance used by the
few-shot classifier and by future fine-tuning data collection.
"""
from __future__ import annotations

INTENTS: dict[str, str] = {
    "check_eligibility": "Does the user qualify for a scheme (age, income, caste, occupation)?",
    "scheme_benefits": "What are the benefits/amounts/entitlements of a scheme?",
    "required_documents": "Which documents must be submitted to apply?",
    "application_status": "Where/how to check the status of an application?",
    "general_inquiry": "Anything else: scheme overview, helpline, general information.",
}

INTENT_ALIASES: dict[str, list[str]] = {
    "check_eligibility": [
        "eligible", "qualify", "am i eligible", "can i get", "entitled",
        "apply for", "who can", "क्या पात्र", "தகுதி", "yogya",
    ],
    "scheme_benefits": [
        "benefit", "how much", "amount", "money", "payout", "subsidy",
        "help get", "fayda", "லாபம்", "நன்மை", "लाभ",
    ],
    "required_documents": [
        "documents", "papers", "proof", "required", "need to apply",
        "aadhaar", "pan", "certificate", "दस्तावेज", "ஆவணம்",
    ],
    "application_status": [
        "status", "track", "how long", "update", "when will", "approval",
        "sthaathi", "நிலை", "स्टेटस", "kahan",
    ],
    "general_inquiry": [
        "what is", "about", "tell me", "information", "help",
        "jaankari", "தகவல்", "जानकारी", "sarkari",
    ],
}

ENTITIES: dict[str, str] = {
    "scheme_name": "PM-KISAN, Ayushman Bharat, PMAY, state welfare schemes",
    "age": "age in years (numeric)",
    "income_level": "annual family income / income band",
    "occupation": "farmer, labourer, weaver, self-employed, pensioner",
    "caste_category": "SC / ST / OBC / EWS / General",
    "district_state": "district and/or state mentioned in the query",
}

# regexes for rule-based entity extraction (English + transliterated Hindi/Tamil)
ENTITY_PATTERNS: dict[str, str] = {
    "age": r"\b(?:age|aged|years? old|साल|வயது)\s*:?\s*(\d{2})\b|(\d{2})\s*(?:years?|saal|varusham)\b",
    "income_level": r"\b(?:income|annual income)\s*:?\s*₹?\s?(\d[\d,]*)\b|(\d[\d,]*)\s*(?:rs|rupees)\b",
    "caste_category": r"\b(SC|ST|OBC|EWS|General|scheduled caste|scheduled tribe|अनुसूचित|बीसी|எஸ்சி|ஓபிசி)\b",
    "scheme_name": r"\b(PM-KISAN|PM Kisan|Pradhan Mantri Kisan|Ayushman Bharat|PM Awas|PMAY|PMJAY|rural housing)\b",
}

SCHEME_KEYWORDS: dict[str, list[str]] = {
    "pm-kisan": ["pm-kisan", "pm kisan", "pradhan mantri kisan", "kisan samman"],
    "ayushman-bharat": ["ayushman", "pmjay", "health insurance", "medical cover"],
    "pmay": ["pmay", "pm awas", "awas yojana", "rural housing", "housing"],
    "mgnrega": ["mgnrega", "mnrega", "employment guarantee", "job card", "narega", "रोजगार"],
    "ujjwala": ["ujjwala", "lpg", "cooking gas", "gas connection", "उज्ज्वला"],
    "jan-dhan": ["jan dhan", "pmjdy", "bank account", "zero balance", "jan धन"],
    "sukanya-samriddhi": ["sukanya", "samriddhi", "girl child", "beti bachao", "सुकन्या"],
}
