"""
Provenance: [FORK NEW]

Purpose:
Audits offline cache coverage, model availability, and parsing health across
benchmarks without initiating external API calls. Outputs a formatted console
table and exports a CSV audit report to outputs/tables/cache_inventory.csv.
"""

import argparse
import os
import sys
from typing import List, Dict, Any

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from src.offline_loader import OfflineDataLoader, OFFICIAL_PAPER_MODELS
from src.utils import get_parse_fn


# Core primary benchmarks for initial development
CORE_BENCHMARKS: List[str] = ["gsm8k", "svamp", "drop"]

# All 14 text benchmarks supported in the paper
ALL_TEXT_BENCHMARKS: List[str] = [
    "gsm8k",
    "svamp",
    "drop",
    "singleop",
    "singleq",
    "multiarith",
    "mmlu_math",
    "bbh_logical_deduction_three_objects",
    "bbh_object_counting",
    "bbh_navigate",
    "tab_fact",
    "hotpotqa",
    "squad",
    "winograd_wsc",
]


def audit_benchmark_cache(
    dataset_name: str,
    loader: OfflineDataLoader,
    model_list: List[str],
) -> Dict[str, Any]:
    """
    Performs full sanity audit on a single benchmark dataset.

    Args:
        dataset_name (str): Identifier of the benchmark.
        loader (OfflineDataLoader): Offline cache reader.
        model_list (List[str]): List of target model identifiers.

    Returns:
        Dict[str, Any]: Summary metrics for the benchmark.
    """
    print(f"\n[Audit] Inspecting dataset: '{dataset_name}'...")

    # Load dataset test split from Hugging Face
    dataset = load_dataset("madrylab/platinum-bench-paper-version", dataset_name, split="test")
    total_instances = len(dataset)

    # Separate rejected instances
    valid_instances = [ex for ex in dataset if ex.get("cleaning_status") != "rejected"]
    rejected_count = total_instances - len(valid_instances)
    valid_count = len(valid_instances)

    # Determine canonical parsing strategy
    strategy = valid_instances[0].get("platinum_parsing_strategy", "math")
    try:
        parse_fn = get_parse_fn(strategy)
    except Exception as e:
        print(f"  [Warning] Could not initialize parse_fn for strategy '{strategy}': {e}")
        parse_fn = None

    total_expected_lookups = valid_count * len(model_list)
    total_found_lookups = 0
    total_parsing_success = 0
    total_parsing_errors = 0

    full_cache_instances = 0
    incomplete_cache_instances = 0
    available_counts_per_instance: List[int] = []

    for ex in tqdm(valid_instances, desc=f"Scanning {dataset_name}", leave=False):
        ensemble_data = loader.get_ensemble_responses(
            dataset_name=dataset_name,
            example=ex,
            model_list=model_list,
        )

        k_i = ensemble_data["num_available"]
        available_counts_per_instance.append(k_i)
        total_found_lookups += k_i

        if ensemble_data["incomplete_cache"]:
            incomplete_cache_instances += 1
        else:
            full_cache_instances += 1

        # Check parsing health on available responses
        if parse_fn is not None:
            for model_name in ensemble_data["models_available"]:
                raw_resp = ensemble_data["responses"][model_name]
                try:
                    parsed = parse_fn(raw_resp)
                    if parsed is not None and str(parsed).strip() != "":
                        total_parsing_success += 1
                    else:
                        total_parsing_errors += 1
                except Exception:
                    total_parsing_errors += 1

    coverage_rate = (total_found_lookups / total_expected_lookups) * 100 if total_expected_lookups > 0 else 0.0
    mean_k = sum(available_counts_per_instance) / valid_count if valid_count > 0 else 0.0
    parsing_rate = (total_parsing_success / total_found_lookups) * 100 if total_found_lookups > 0 else 0.0

    return {
        "dataset": dataset_name,
        "strategy": strategy,
        "total_examples": total_instances,
        "valid_examples": valid_count,
        "rejected_examples": rejected_count,
        "target_models_K": len(model_list),
        "mean_available_K": round(mean_k, 2),
        "full_cache_examples": full_cache_instances,
        "incomplete_cache_examples": incomplete_cache_instances,
        "cache_coverage_pct": round(coverage_rate, 2),
        "parsing_success_pct": round(parsing_rate, 2),
        "parsing_errors_count": total_parsing_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Cache Sanity and Inventory Audit Tool (TAAC MSc AI)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["core"],
        help="List of benchmarks to audit, or 'core' for (gsm8k, svamp, drop), or 'all' for all 14 text benchmarks.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Path to local cache directory containing .pkl files.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./outputs/tables/cache_inventory.csv",
        help="Path to save summary inventory CSV.",
    )
    args = parser.parse_args()

    # Resolve dataset selection
    if "core" in args.datasets:
        selected_datasets = CORE_BENCHMARKS
    elif "all" in args.datasets:
        selected_datasets = ALL_TEXT_BENCHMARKS
    else:
        selected_datasets = args.datasets

    print("=" * 80)
    print("      OFFLINE CACHE SANITY & INVENTORY AUDITOR (Anti-API Mode)      ")
    print(f"Target Benchmarks: {', '.join(selected_datasets)}")
    print(f"Model Pool: {len(OFFICIAL_PAPER_MODELS)} Models from Vendrow et al. (2025)")
    print(f"Cache Directory: {os.path.abspath(args.cache_dir)}")
    print("=" * 80)

    loader = OfflineDataLoader(cache_dir=args.cache_dir)
    results: List[Dict[str, Any]] = []

    for dataset_name in selected_datasets:
        stats = audit_benchmark_cache(
            dataset_name=dataset_name,
            loader=loader,
            model_list=OFFICIAL_PAPER_MODELS,
        )
        results.append(stats)

    df_results = pd.DataFrame(results)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df_results.to_csv(args.output_csv, index=False)

    print("\n" + "=" * 80)
    print("                     SANITY & INVENTORY REPORT                     ")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)
    print(f"[Done] Report successfully exported to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
