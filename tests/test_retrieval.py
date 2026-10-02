"""
RAG Retrieval Verification Test Suite (Section 7.1 Gate).
Tests 10 hand-crafted queries per scheme (PM-KISAN, MGNREGA, Ayushman Bharat = 30 queries)
and ensures top-3 recall >= 90%.
"""

import pytest
from services.rag.retriever import RAGRetriever
from services.rag.ingest import tokenize_stem

RETRIEVAL_TEST_CASES = [
    # PM-KISAN (10 queries)
    {"scheme": "pm_kisan", "query": "Who is eligible for PM-KISAN landholder scheme?", "expected_keywords": ["eligible", "landholding", "cultivable"]},
    {"scheme": "pm_kisan", "query": "Are income tax payers excluded from PM Kisan?", "expected_keywords": ["tax", "exclusion", "excluded"]},
    {"scheme": "pm_kisan", "query": "What documents are required for PM-KISAN registration?", "expected_keywords": ["aadhaar", "document", "khasra"]},
    {"scheme": "pm_kisan", "query": "Is Aadhaar card mandatory for PM Kisan?", "expected_keywords": ["aadhaar", "mandatory"]},
    {"scheme": "pm_kisan", "query": "How to register online on pmkisan.gov.in portal?", "expected_keywords": ["portal", "register", "farmer"]},
    {"scheme": "pm_kisan", "query": "Can I apply at Common Service Centre CSC for PM Kisan?", "expected_keywords": ["csc", "service", "centre"]},
    {"scheme": "pm_kisan", "query": "How much annual benefit is given in PM-KISAN?", "expected_keywords": ["6,000", "installment", "benefit"]},
    {"scheme": "pm_kisan", "query": "How many installments of 2000 rupees are paid?", "expected_keywords": ["2,000", "installment", "three"]},
    {"scheme": "pm_kisan", "query": "Can constitutional post holders get PM Kisan money?", "expected_keywords": ["constitutional", "exclusion", "ministers"]},
    {"scheme": "pm_kisan", "query": "What bank account details are needed for DBT transfer?", "expected_keywords": ["bank", "dbt", "account"]},

    # MGNREGA (10 queries)
    {"scheme": "mgnrega", "query": "Who can apply for 100 days guaranteed work under MGNREGA?", "expected_keywords": ["rural", "adult", "manual"]},
    {"scheme": "mgnrega", "query": "What is the minimum age to work in MGNREGA?", "expected_keywords": ["18", "age", "adult"]},
    {"scheme": "mgnrega", "query": "What documents are required to get an MGNREGA Job Card?", "expected_keywords": ["job card", "aadhaar", "photo"]},
    {"scheme": "mgnrega", "query": "Do I need proof of rural residence for Job Card?", "expected_keywords": ["residence", "panchayat", "ration"]},
    {"scheme": "mgnrega", "query": "How to submit job card application in Gram Panchayat?", "expected_keywords": ["gram panchayat", "application", "job card"]},
    {"scheme": "mgnrega", "query": "Within how many days must Gram Panchayat issue Job Card?", "expected_keywords": ["15 days", "free of cost", "issue"]},
    {"scheme": "mgnrega", "query": "How many days of wage employment is guaranteed per financial year?", "expected_keywords": ["100 days", "guarantee", "financial year"]},
    {"scheme": "mgnrega", "query": "Are equal wages paid to men and women under MGNREGA?", "expected_keywords": ["wage", "equal", "notified"]},
    {"scheme": "mgnrega", "query": "Is unemployment allowance paid if work is not given in 15 days?", "expected_keywords": ["unemployment", "allowance", "15 days"]},
    {"scheme": "mgnrega", "query": "What is the distance limit for MGNREGA worksite from village?", "expected_keywords": ["5 km", "radius", "travel"]},

    # Ayushman Bharat (10 queries)
    {"scheme": "ayushman_bharat", "query": "Who is eligible for 5 lakh health insurance under Ayushman Bharat?", "expected_keywords": ["secc", "eligible", "health", "deprivation"]},
    {"scheme": "ayushman_bharat", "query": "Are senior citizens aged 70 and above eligible for Ayushman cover?", "expected_keywords": ["70", "senior", "citizen"]},
    {"scheme": "ayushman_bharat", "query": "What documents are required for Ayushman Golden Card e-KYC?", "expected_keywords": ["aadhaar", "ration", "card"]},
    {"scheme": "ayushman_bharat", "query": "How to check eligibility online on beneficiary.nha.gov.in?", "expected_keywords": ["portal", "beneficiary", "aadhaar"]},
    {"scheme": "ayushman_bharat", "query": "What is the role of Ayushman Mitra at empanelled hospital?", "expected_keywords": ["ayushman mitra", "hospital", "desk"]},
    {"scheme": "ayushman_bharat", "query": "What is the annual health coverage amount per family?", "expected_keywords": ["5,00,000", "5 lakh", "cover"]},
    {"scheme": "ayushman_bharat", "query": "Is there any restriction on family size or age in PM-JAY?", "expected_keywords": ["family size", "restriction", "floater"]},
    {"scheme": "ayushman_bharat", "query": "Does Ayushman Bharat cover cashless secondary and tertiary hospitalization?", "expected_keywords": ["cashless", "secondary", "tertiary", "hospital"]},
    {"scheme": "ayushman_bharat", "query": "Are pre-existing diseases and medical conditions covered?", "expected_keywords": ["pre-existing", "day one", "covered"]},
    {"scheme": "ayushman_bharat", "query": "How to download Ayushman Card PDF after approval?", "expected_keywords": ["card", "download", "pvc", "ayushman"]},
]


@pytest.mark.asyncio
async def test_30_query_retrieval_suite():
    retriever = RAGRetriever()
    hits = 0
    total = len(RETRIEVAL_TEST_CASES)

    for case in RETRIEVAL_TEST_CASES:
        scheme = case["scheme"]
        query = case["query"]
        expected_keywords = case["expected_keywords"]

        chunks, citations, ms = await retriever.retrieve(query=query, scheme_name=scheme, top_k=3)
        assert len(chunks) > 0, f"No chunks returned for query: {query}"
        assert ms < 200.0, f"Retrieval latency too high: {ms}ms"

        combined_text = " ".join([c.text.lower() for c in chunks])
        matched = False
        for kw in expected_keywords:
            kw_stem = tokenize_stem(kw)
            if any(s in combined_text for s in kw_stem) or kw.lower() in combined_text:
                matched = True
                break

        if matched:
            hits += 1
        else:
            print(f"FAILED RETRIEVAL: {query} -> {chunks[0].text[:60]}")

    recall = hits / total
    print(f"\n[Retrieval Benchmark] Hits: {hits}/{total} ({recall * 100:.1f}%)")
    assert recall >= 0.90, f"Retrieval top-3 recall {recall:.2f} is below 90% target!"
