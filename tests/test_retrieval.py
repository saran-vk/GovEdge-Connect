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

    # Operational Additions: PM-KISAN (3 queries)
    {"scheme": "pm_kisan", "query": "How can an ineligible farmer refund PM-KISAN installment online?", "expected_keywords": ["refund", "ineligible", "voluntary"]},
    {"scheme": "pm_kisan", "query": "What should I do if PFMS shows CBE0061 UID never enabled for DBT?", "expected_keywords": ["cbe0061", "dbt", "mandate", "npci"]},
    {"scheme": "pm_kisan", "query": "Can PM Kisan beneficiaries get a Kisan Credit Card KCC loan at 4 percent?", "expected_keywords": ["kcc", "credit", "loan", "4%"]},

    # Operational Additions: MGNREGA (4 queries)
    {"scheme": "mgnrega", "query": "Are water harvesting and check dams permissible under Category A NRM works?", "expected_keywords": ["natural resource", "water conservation", "permissible"]},
    {"scheme": "mgnrega", "query": "How is worker attendance captured using NMMS mobile app?", "expected_keywords": ["nmms", "attendance", "geo-tagged", "photographs"]},
    {"scheme": "mgnrega", "query": "What facilities like drinking water, first-aid, and crèche are mandated at worksite?", "expected_keywords": ["drinking water", "first-aid", "crèche", "shade"]},
    {"scheme": "mgnrega", "query": "What is the wage delay compensation percentage per day after 15 days?", "expected_keywords": ["0.05%", "delay compensation", "unpaid wage"]},

    # Operational Additions: Ayushman Bharat (3 queries)
    {"scheme": "ayushman_bharat", "query": "Does PM-JAY cover cancer chemotherapy oncology and heart bypass surgery?", "expected_keywords": ["oncology", "chemotherapy", "cardiology", "surgical"]},
    {"scheme": "ayushman_bharat", "query": "Can I get cashless hospital treatment in another state outside my hometown?", "expected_keywords": ["portability", "interstate", "empanelled"]},
    {"scheme": "ayushman_bharat", "query": "What is the toll-free helpline number 14555 to report hospital fraud?", "expected_keywords": ["14555", "helpline", "toll-free", "anti-fraud"]},

    # New Scheme: PMAY-G (10 queries)
    {"scheme": "pmay_g", "query": "What is the unit financial assistance amount for building house in plain areas?", "expected_keywords": ["1,20,000", "plain", "assistance"]},
    {"scheme": "pmay_g", "query": "What is the assistance amount for hilly and northeastern states in PMAY-G?", "expected_keywords": ["1,30,000", "hilly", "north-eastern"]},
    {"scheme": "pmay_g", "query": "How many days of MGNREGA unskilled labour wages are provided for house construction?", "expected_keywords": ["90", "95", "person-days", "mgnrega"]},
    {"scheme": "pmay_g", "query": "How are construction inspection stages tracked using AwaasApp mobile photos?", "expected_keywords": ["awaasapp", "geo-tagged", "photographs", "inspection"]},
    {"scheme": "pmay_g", "query": "What are the exclusion criteria like motorized vehicles or KCC limit in PMAY-G?", "expected_keywords": ["exclusion", "motorized", "50,000", "vehicle"]},
    {"scheme": "pmay_g", "query": "How are the three installments disbursed across foundation and roof cast stages?", "expected_keywords": ["installment", "first", "second", "roof-cast"]},
    {"scheme": "pmay_g", "query": "What is the Rural Mason Training program for disaster resilient houses?", "expected_keywords": ["rural mason", "training", "disaster-resilient", "nsqf"]},
    {"scheme": "pmay_g", "query": "How do landless rural households get homestead land for PMAY house?", "expected_keywords": ["landless", "homestead", "residential land"]},
    {"scheme": "pmay_g", "query": "Does PM-JANMAN mission provide fast track houses for vulnerable tribal groups?", "expected_keywords": ["pm-janman", "pvtg", "tribal", "vulnerable"]},
    {"scheme": "pmay_g", "query": "How can beneficiaries track their FTO and inspection photos on pmayg.nic.in?", "expected_keywords": ["pmayg.nic.in", "fto", "registration", "tracking"]},
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
