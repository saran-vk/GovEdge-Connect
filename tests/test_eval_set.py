"""
Tests verifying integrity and coverage of the 50 seed evaluation queries.
"""

import json
from pathlib import Path
from gateway.schemas import Intent

EVAL_FILE = Path(__file__).parent.parent / "data" / "eval" / "50_seed_queries.json"


def test_eval_dataset_integrity():
    assert EVAL_FILE.exists(), f"Evaluation file not found: {EVAL_FILE}"

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Exactly 50 queries as promised
    assert len(data) == 50, f"Expected 50 queries, found {len(data)}"

    schemes = {"pm_kisan": 0, "mgnrega": 0, "ayushman_bharat": 0, None: 0}
    intents = {i.value: 0 for i in Intent}
    ids = set()

    for item in data:
        assert "id" in item
        assert "query_en" in item
        assert "target_scheme" in item
        assert "target_intent" in item
        assert "ground_truth_facts" in item
        assert "is_out_of_scope" in item

        # Unique IDs
        assert item["id"] not in ids, f"Duplicate ID found: {item['id']}"
        ids.add(item["id"])

        # Intent validity
        assert item["target_intent"] in intents
        intents[item["target_intent"]] += 1

        # Scheme validity
        assert item["target_scheme"] in schemes
        schemes[item["target_scheme"]] += 1

        # Ground truth facts
        assert len(item["ground_truth_facts"]) > 0

    # Verify coverage balance
    assert schemes["pm_kisan"] == 13
    assert schemes["mgnrega"] == 13
    assert schemes["ayushman_bharat"] == 13
    assert schemes[None] == 11
    assert intents[Intent.OUT_OF_SCOPE.value] == 11
