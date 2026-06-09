"""
Test different chunk sizes and compare precision/recall.
Automates the full pipeline: clear → ingest → test → report.
"""

import os
import subprocess
import json
import shutil
from pathlib import Path

CHUNK_SIZES = [300, 400, 500, 1000]
CHROMA_PATH = "./chroma_db"

def clear_chroma():
    """Remove old ChromaDB to start fresh."""
    if Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH)
        print(f"✓ Cleared {CHROMA_PATH}")

def ingest_with_chunk_size(chunk_size):
    """Run ingestion with specific chunk size."""
    print(f"\n{'='*80}")
    print(f"INGESTING with CHUNK_SIZE={chunk_size}")
    print(f"{'='*80}\n")

    env = os.environ.copy()
    env["CHUNK_SIZE"] = str(chunk_size)

    result = subprocess.run(
        ["python3", "ingest.py"],
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"✗ Ingestion failed: {result.stderr}")
        return False

    print(result.stdout)
    return True

def run_tests():
    """Run test suite and parse results."""
    print(f"\n{'='*80}")
    print("RUNNING TESTS")
    print(f"{'='*80}\n")

    result = subprocess.run(
        ["python3", "tests/test_retrieval.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"✗ Tests failed: {result.stderr}")
        return None

    # Parse the output to extract precision/recall
    lines = result.stdout.split('\n')
    metrics = {}

    for i, line in enumerate(lines):
        if 'Average Precision' in line:
            precision = float(line.split(':')[1].strip())
            metrics['precision'] = precision
        elif 'Average Recall' in line:
            recall = float(line.split(':')[1].strip())
            metrics['recall'] = recall
        elif 'Average MRR' in line:
            mrr = float(line.split(':')[1].strip())
            metrics['mrr'] = mrr
        elif 'Average F1' in line:
            f1 = float(line.split(':')[1].strip())
            metrics['f1'] = f1

    print(result.stdout)
    return metrics

def main():
    print("\n" + "="*80)
    print("CHUNK SIZE OPTIMIZATION TEST")
    print("="*80)

    results = {}

    for chunk_size in CHUNK_SIZES:
        clear_chroma()

        if not ingest_with_chunk_size(chunk_size):
            results[chunk_size] = None
            continue

        metrics = run_tests()
        results[chunk_size] = metrics

    # Print comparison table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80 + "\n")

    print(f"{'Chunk Size':<15} {'Precision':<15} {'Recall':<15} {'MRR':<15} {'F1':<15}")
    print("-" * 75)

    best_precision = None
    best_chunk = None

    for chunk_size in CHUNK_SIZES:
        if results[chunk_size] is None:
            print(f"{chunk_size:<15} {'FAILED':<15}")
            continue

        metrics = results[chunk_size]
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        mrr = metrics.get('mrr', 0)
        f1 = metrics.get('f1', 0)

        precision_str = f"{precision:.0%}"
        recall_str = f"{recall:.0%}"
        mrr_str = f"{mrr:.2f}"
        f1_str = f"{f1:.2f}"

        print(f"{chunk_size:<15} {precision_str:<15} {recall_str:<15} {mrr_str:<15} {f1_str:<15}")

        if best_precision is None or precision > best_precision:
            best_precision = precision
            best_chunk = chunk_size

    print("-" * 75)
    if best_chunk:
        print(f"\n✓ WINNER: {best_chunk} chars (Precision: {best_precision:.0%})")

    print("\nNote: Higher precision is prioritized. Pick the chunk size with best precision.")

if __name__ == "__main__":
    main()
