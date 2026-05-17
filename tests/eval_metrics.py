"""
Retrieval evaluation metrics.
Measure how good your RAG's retrieval is.
"""


def precision_at_k(retrieved_names: list[str], expected_names: list[str], k: int = 4) -> float:
    """
    Precision@K: Of the top K results, how many were relevant?

    Args:
        retrieved_names: List of names returned by the vector DB
        expected_names: List of names that SHOULD have been retrieved
        k: Number of top results to consider (default 4)

    Returns:
        Fraction between 0 and 1. Higher is better.
        1.0 = all top K results were relevant
        0.0 = none of them were relevant

    Example:
        retrieved = ["Messi", "Ronaldo", "Random", "Another"]
        expected = ["Messi", "Ronaldo"]
        precision_at_k(retrieved, expected, k=4) = 2/4 = 0.5
    """
    retrieved_at_k = retrieved_names[:k]
    relevant_count = sum(1 for name in retrieved_at_k if name in expected_names)
    return relevant_count / k if k > 0 else 0.0


def recall_at_k(retrieved_names: list[str], expected_names: list[str], k: int = 4) -> float:
    """
    Recall@K: Of all relevant docs, how many did we find?

    Args:
        retrieved_names: List of names returned by the vector DB
        expected_names: List of names that SHOULD have been retrieved
        k: Number of top results to consider (default 4)

    Returns:
        Fraction between 0 and 1. Higher is better.
        1.0 = we found all relevant documents
        0.0 = we found none of them

    Example:
        retrieved = ["Messi", "Ronaldo", "Random", "Another"]
        expected = ["Messi", "Ronaldo"]
        recall_at_k(retrieved, expected, k=4) = 2/2 = 1.0
        (we found all 2 expected, even though there were other irrelevant results)
    """
    retrieved_at_k = retrieved_names[:k]
    relevant_count = sum(1 for name in retrieved_at_k if name in expected_names)
    return relevant_count / len(expected_names) if len(expected_names) > 0 else 0.0


def mrr(retrieved_names: list[str], expected_names: list[str]) -> float:
    """
    Mean Reciprocal Rank: How high up is the first relevant result?

    Args:
        retrieved_names: List of names returned by the vector DB
        expected_names: List of names that SHOULD have been retrieved

    Returns:
        Fraction between 0 and 1. Higher is better.
        1.0 = first result was relevant
        0.5 = second result was relevant
        0.0 = no relevant results found

    Example:
        retrieved = ["Random", "Ronaldo", "Messi"]
        expected = ["Messi", "Ronaldo"]
        mrr(retrieved, expected) = 1/2 = 0.5
        (first relevant result is at position 2)
    """
    for i, name in enumerate(retrieved_names):
        if name in expected_names:
            return 1.0 / (i + 1)
    return 0.0


def f1_score(precision: float, recall: float) -> float:
    """
    F1 score: Harmonic mean of precision and recall.
    Balances both metrics.

    Returns:
        Score between 0 and 1. Higher is better.
        0.0 if both precision and recall are 0
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
