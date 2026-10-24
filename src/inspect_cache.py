"""
Provenance: [FORK NEW]

Purpose:
Interactive visual CLI inspector for dataset question-answer pairs and precomputed
LLM ensemble responses from the offline cache. Enables qualitative error analysis
and verification of model agreement versus human curation targets.
"""

import argparse
import os
import sys
from collections import Counter
from datasets import load_dataset

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from src.offline_loader import OfflineDataLoader, OFFICIAL_PAPER_MODELS
from src.utils import get_parse_fn, check_prediction


def inspect_example(dataset_name: str, example_idx: int, show_all_models: bool = False):
    """Visualizes an individual question, its targets, and ensemble model answers."""
    loader = OfflineDataLoader()

    print(f"\n[Loading] Dataset: {dataset_name} | Split: test ...")
    dataset = load_dataset("madrylab/platinum-bench-paper-version", dataset_name, split="test")

    if example_idx < 0 or example_idx >= len(dataset):
        print(f"[Error] Index {example_idx} out of range (0 to {len(dataset)-1}).")
        return

    example = dataset[example_idx]
    strategy = example.get("platinum_parsing_strategy", "text")
    parse_fn = get_parse_fn(strategy)

    print("\n" + "=" * 90)
    print(f"                      BENCHMARK: {dataset_name.upper()} — EXAMPLE #{example_idx}                      ")
    print("=" * 90)

    # 1. Question Information
    question_text = example.get("question", "")
    print(f"\n[QUESTION / STATEMENT]")
    print("-" * 90)
    print(question_text.strip())

    # If DROP or Reading Comprehension with passage / context
    passage = example.get("passage") or example.get("context")
    if passage:
        print(f"\n[PASSAGE / CONTEXT] (first 300 chars):")
        print("-" * 90)
        print(passage.strip()[:300] + "..." if len(passage) > 300 else passage.strip())

    # 2. Target Labels & Metadata
    print(f"\n[LABELS & METADATA]")
    print("-" * 90)
    print(f"  * Platinum Target (Curated) : {example.get('platinum_target')}")
    print(f"  * Original Legacy Target    : {example.get('original_target')}")
    print(f"  * Cleaning Status           : {example.get('cleaning_status')}")
    print(f"  * Parsing Strategy          : {strategy}")

    # 3. Retrieve Ensemble Answers from Cache
    ensemble = loader.get_ensemble_responses(dataset_name, example, OFFICIAL_PAPER_MODELS)
    parsed_answers = {}
    correctness_map = {}

    for m in OFFICIAL_PAPER_MODELS:
        raw = ensemble["responses"].get(m)
        if raw is not None:
            try:
                parsed = parse_fn(raw)
            except Exception as e:
                parsed = f"PARSING_ERROR: {e}"
        else:
            parsed = "MISSING_IN_CACHE"

        parsed_answers[m] = parsed
        is_corr = check_prediction(str(parsed), example.get("platinum_target", []), "", dataset_name)
        correctness_map[m] = is_corr

    # 4. Vote Distribution Summary
    vote_counts = Counter(parsed_answers.values())
    print(f"\n[VOTE DISTRIBUTION ACROSS {len(OFFICIAL_PAPER_MODELS)} MODELS]")
    print("-" * 90)
    for ans, count in vote_counts.most_common():
        match_target = "Correct" if check_prediction(str(ans), example.get("platinum_target", []), "", dataset_name) else "Divergent"
        pct = (count / len(OFFICIAL_PAPER_MODELS)) * 100
        print(f"  [{count:2d}/24 votes ({pct:5.1f}%)] -> '{ans}' ({match_target})")

    # 5. Model Predictions Breakdown
    models_to_display = OFFICIAL_PAPER_MODELS if show_all_models else [
        "gpt-4o-2024-11-20",
        "claude-3-5-sonnet-20241022",
        "deepseek-r1",
        "gemini-2.0-flash-thinking",
        "o1-mini",
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "mistral-large",
        "Qwen/Qwen2.5-72B-Instruct",
    ]

    print(f"\n[SAMPLE MODEL REASONING & OUTPUTS] ({len(models_to_display)} models):")
    print("-" * 90)
    for m in models_to_display:
        raw = ensemble["responses"].get(m, "")
        parsed = parsed_answers.get(m)
        status_sym = "CORRECT" if correctness_map.get(m) else "INCORRECT"

        print(f"\n  [{status_sym}] Model: {m}")
        print(f"      Parsed Prediction: {parsed}")
        
        # Display preview of raw text
        cleaned_raw = raw.strip() if raw else "[NO RESPONSE]"
        if len(cleaned_raw) > 250:
            preview = cleaned_raw[:200].replace("\n", " ") + " ... " + cleaned_raw[-50:].replace("\n", " ")
        else:
            preview = cleaned_raw.replace("\n", " ")
        print(f"      Raw Text Preview : {preview}")

    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Cached Benchmark Q&A Pairs")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name (e.g., gsm8k, svamp, drop)")
    parser.add_argument("--idx", type=int, default=0, help="Example index in test split")
    parser.add_argument("--all-models", action="store_true", help="Display full list of all 24 models")
    args = parser.parse_args()

    inspect_example(args.dataset, args.idx, args.all_models)
