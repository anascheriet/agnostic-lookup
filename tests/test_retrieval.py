"""
Test retrieval quality of the RAG system.

This runs your test queries through the RAG, measures how well
it retrieves relevant documents, and reports metrics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import retrieve
from tests.test_data import TEST_QUERIES
from tests.eval_metrics import precision_at_k, recall_at_k, mrr, f1_score


def test_retrieval():
    """
    Run all test queries and print retrieval metrics.
    """
    print("\n" + "="*80)
    print("RETRIEVAL QUALITY EVALUATION")
    print("="*80 + "\n")

    results = []

    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        domain = test_case.get("domain")
        expected = test_case["expected_subjects"]

        print(f"[Test {i}] {test_case['description']}")
        print(f"  Query: {query}")
        print(f"  Domain: {domain}")
        print(f"  Expected subjects: {expected}\n")

        # Run retrieval (this is the core RAG step)
        matches = retrieve(query, domain=domain)
        retrieved_names = [m["name"] for m in matches]

        print(f"  Retrieved ({len(matches)} results above threshold):")
        for j, match in enumerate(matches, 1):
            marker = "✓" if match["name"] in expected else "✗"
            similarity = match.get("similarity", 0)
            print(f"    {j}. {marker} {match['name']} (similarity: {similarity:.3f})")

        # Calculate metrics (use actual result count as k)
        k = len(matches) if matches else 1
        p_at_k = precision_at_k(retrieved_names, expected, k=k)
        r_at_k = recall_at_k(retrieved_names, expected, k=k)
        m_rr = mrr(retrieved_names, expected)
        f1 = f1_score(p_at_k, r_at_k)

        print(f"\n  Metrics (based on {k} results):")
        print(f"    Precision: {p_at_k:.0%}")
        print(f"    Recall:    {r_at_k:.0%}")
        print(f"    MRR:       {m_rr:.2f}")
        print(f"    F1 Score:  {f1:.2f}")
        print()

        results.append({
            "test": test_case["description"],
            "precision": p_at_k,
            "recall": r_at_k,
            "mrr": m_rr,
            "f1": f1,
        })

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)

    print(f"Average Precision: {avg_precision:.0%}")
    print(f"Average Recall:    {avg_recall:.0%}")
    print(f"Average MRR:       {avg_mrr:.2f}")
    print(f"Average F1 Score:  {avg_f1:.2f}")
    print()

    # Interpretation
    print("Interpretation:")
    print(f"  Precision {avg_precision:.0%}: Of results returned, {avg_precision:.0%} were relevant (threshold-filtered)")
    print(f"  Recall {avg_recall:.0%}: Of all relevant docs, we found {avg_recall:.0%}")
    print()

    return results


if __name__ == "__main__":
    test_retrieval()
