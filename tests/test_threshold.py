"""
Test how similarity threshold affects precision/recall.

This shows: "If I only keep results with similarity > X, how does precision change?"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import retrieve
from tests.test_data import TEST_QUERIES
from tests.eval_metrics import precision_at_k, recall_at_k


def test_thresholds():
    """
    For each test query, try different similarity thresholds and show how it affects results.
    """
    print("\n" + "="*80)
    print("SIMILARITY THRESHOLD ANALYSIS")
    print("="*80 + "\n")

    test_case = TEST_QUERIES[0]  # Use first test (Messi vs Ronaldo)
    query = test_case["query"]
    domain = test_case.get("domain")
    expected = test_case["expected_subjects"]

    print(f"Test: {test_case['description']}")
    print(f"Query: {query}\n")

    # Get raw results with very low threshold to see all options
    raw_results = retrieve(query, domain=domain, similarity_threshold=0.3)

    print("All results with similarity scores:")
    for i, result in enumerate(raw_results, 1):
        marker = "✓" if result["name"] in expected else "✗"
        print(f"  {i}. {marker} {result['name']:20s} (similarity: {result['similarity']:.3f})")
    print()

    # Test different thresholds
    thresholds = [0.60, 0.70, 0.75, 0.80, 0.85]

    print("="*80)
    print("THRESHOLD TESTING")
    print("="*80 + "\n")

    for threshold in thresholds:
        filtered = retrieve(query, domain=domain, similarity_threshold=threshold)
        filtered_names = [r["name"] for r in filtered]

        precision = precision_at_k(filtered_names, expected, k=len(filtered_names))
        recall = recall_at_k(filtered_names, expected, k=len(filtered_names))

        print(f"Threshold >= {threshold:.2f}:")
        print(f"  Results kept: {len(filtered)} (were {len(raw_results)})")
        if filtered:
            print(f"  Names: {', '.join(r['name'] for r in filtered)}")
            print(f"  Precision: {precision:.0%} | Recall: {recall:.0%}")
        else:
            print(f"  No results (too strict!)")
        print()


if __name__ == "__main__":
    test_thresholds()
