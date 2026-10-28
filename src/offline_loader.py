"""
Provenance: [FORK NEW]

Purpose:
Strictly offline, network-free data loader for precomputed LLM response caches.
Reconstructs canonical cache keys (prompt, temperature, run_id, model_name) across
frontier models evaluated in Vendrow et al. (2025) and enforces hermetic isolation
by raising MISSING_IN_CACHE flags without making external network calls.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Any


# ------------------------------------------------------------------------------
# Official 24 Models evaluated in the base paper (Vendrow et al., 2025)
# ------------------------------------------------------------------------------
OFFICIAL_PAPER_MODELS: List[str] = [
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "claude-3-5-sonnet",
    "o1-mini",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gemini-1.5-pro",
    "mistral-large",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "gpt-4o-mini",
    "gemini-1.5-flash",
    "mistral-small",
    "o1-preview-2024-09-12",
    "meta-llama/Llama-3.3-70B-Instruct",
    "grok-2-1212",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku",
    "gemini-2.0-flash",
    "gemini-2.0-flash-thinking",
    "deepseek/deepseek-chat",
    "Qwen/Qwen2.5-72B-Instruct",
    "o1-2024-12-17-med",
    "o1-2024-12-17-high",
    "deepseek-r1",
    "o3-mini-2025-01-31-high",
]

# Reasoning models that use native Chain-of-Thought without explicit prompt instructions
REASONING_MODELS: List[str] = [
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-2024-12-17-high",
    "o1-2024-12-17-med",
    "o3-mini-2025-01-31-high",
    "deepseek-r1",
    "gemini-2.0-flash-thinking",
    "gemini-2.0-flash-thinking-01-21",
    "claude-3-7-sonnet-20250219-thinking",
]

# Model-specific temperature overrides from the base paper
MODEL_TEMPERATURE_OVERRIDES: Dict[str, Optional[float]] = {
    "o1-preview-2024-09-12": 1.0,
    "o1-mini": 1.0,
    "o1-2024-12-17-high": 1.0,
    "o1-2024-12-17-med": 1.0,
    "o3-mini-2025-01-31-high": 1.0,
    "claude-3-7-sonnet-20250219-thinking": None,
}


class OfflineDataLoader:
    """
    Manages access to precomputed LLM inferences stored in pickle files (.pkl).
    Completely isolated from external APIs.
    """

    def __init__(self, cache_dir: str = "./cache"):
        """
        Initializes the OfflineDataLoader.

        Args:
            cache_dir (str): Relative or absolute path to the directory containing .pkl cache files.
        """
        self.cache_dir = os.path.abspath(cache_dir)
        # In-memory dictionary holding loaded benchmark caches: {dataset_name: dict}
        self._loaded_caches: Dict[str, Dict[Tuple, str]] = {}

        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(
                f"[OfflineDataLoader] Cache directory not found at: '{self.cache_dir}'. "
                "Please run `bash scripts/download_paper_cache.sh` first."
            )

    @staticmethod
    def get_temperature(model_name: str) -> Optional[float]:
        """
        Retrieves the canonical inference temperature used for a given model in the paper.
        Default is 0.5; reasoning models typically use 1.0; certain thinking models use None.

        Args:
            model_name (str): Identifier of the model.

        Returns:
            Optional[float]: Numeric temperature or None.
        """
        if model_name in MODEL_TEMPERATURE_OVERRIDES:
            return MODEL_TEMPERATURE_OVERRIDES[model_name]
        return 0.5

    @staticmethod
    def get_prompt(example: Dict[str, Any], model_name: str) -> str:
        """
        Constructs the canonical prompt string for a model, mirroring the paper's exact logic.
        - Native reasoning models receive 'platinum_prompt_no_cot'.
        - Special workaround: o1 models have 'Then, provide' replaced with 'Provide' to prevent refusals.
        - Standard models receive 'platinum_prompt' (with Chain-of-Thought instructions).

        Args:
            example (dict): Hugging Face test record.
            model_name (str): Model identifier.

        Returns:
            str: Canonical prompt string.
        """
        # Specific workaround for o1-preview / o1-2024-12-17 refusal behavior
        if model_name.startswith("o1-preview") or model_name.startswith("o1-2024-12-17"):
            prompt_no_cot = example.get("platinum_prompt_no_cot", "")
            return prompt_no_cot.replace("Then, provide", "Provide")

        # Models with native reasoning capabilities
        if model_name in REASONING_MODELS:
            return example.get("platinum_prompt_no_cot", "")

        # Standard prompting with explicit Chain-of-Thought instruction
        return example.get("platinum_prompt", "")

    def build_cache_key(self, example: Dict[str, Any], model_name: str, run_id: int = 0) -> Tuple:
        """
        Reconstructs the deterministic 4-tuple cache key used in the base paper.

        Signature: (prompt: str, temperature: float, run_id: int, model_name: str)

        Args:
            example (dict): Test instance record.
            model_name (str): Model identifier.
            run_id (int): Run identifier (default 0 for all paper caches).

        Returns:
            tuple: The exact tuple key stored in the pickle dictionary.
        """
        prompt = self.get_prompt(example, model_name)
        temperature = self.get_temperature(model_name)
        return (prompt, temperature, run_id, model_name)

    def load_dataset_cache(self, dataset_name: str) -> Dict[Tuple, str]:
        """
        Loads the pickle cache for a given benchmark dataset into memory.

        Args:
            dataset_name (str): Benchmark name (e.g., 'gsm8k', 'svamp', 'drop').

        Returns:
            dict: Mapping of cache key tuple -> raw model response string.
        """
        if dataset_name in self._loaded_caches:
            return self._loaded_caches[dataset_name]

        cache_filename = f"reliability_benchmark_cache_{dataset_name}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"[OfflineDataLoader] Cache file not found for dataset '{dataset_name}' at: '{cache_path}'. "
                "Ensure `bash scripts/download_paper_cache.sh` was executed successfully."
            )

        # Strict read-only binary opening
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)

        self._loaded_caches[dataset_name] = cache_data
        return cache_data

    def get_model_response(
        self,
        dataset_name: str,
        example: Dict[str, Any],
        model_name: str,
        run_id: int = 0,
    ) -> Tuple[Optional[str], str, Tuple]:
        """
        Retrieves the raw cached response string for an (example, model) pair.
        Never calls external APIs on a cache miss.

        Args:
            dataset_name (str): Benchmark name.
            example (dict): Test instance record.
            model_name (str): Model identifier.
            run_id (int): Run ID (default 0).

        Returns:
            Tuple[Optional[str], str, Tuple]: (raw_response, status, cache_key)
            status is either 'FOUND' or 'MISSING_IN_CACHE'.
        """
        cache = self.load_dataset_cache(dataset_name)
        key = self.build_cache_key(example, model_name, run_id=run_id)

        if key in cache:
            return cache[key], "FOUND", key
        else:
            return None, "MISSING_IN_CACHE", key

    def get_ensemble_responses(
        self,
        dataset_name: str,
        example: Dict[str, Any],
        model_list: Optional[List[str]] = None,
        run_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Retrieves all available model responses for an ensemble on a given example.

        Args:
            dataset_name (str): Benchmark name.
            example (dict): Test instance record.
            model_list (list, optional): List of model names. Defaults to OFFICIAL_PAPER_MODELS.
            run_id (int): Run ID (default 0).

        Returns:
            dict containing:
                - 'responses': Dict[model_name, raw_response_string]
                - 'statuses': Dict[model_name, 'FOUND' | 'MISSING_IN_CACHE']
                - 'keys': Dict[model_name, tuple_key]
                - 'models_available': List of models with 'FOUND' status
                - 'models_missing': List of models with 'MISSING_IN_CACHE' status
                - 'num_available': int (K_i)
                - 'num_target': int (total models requested)
                - 'incomplete_cache': bool (True if num_available < num_target)
        """
        if model_list is None:
            model_list = OFFICIAL_PAPER_MODELS

        responses: Dict[str, Optional[str]] = {}
        statuses: Dict[str, str] = {}
        keys: Dict[str, Tuple] = {}
        models_available: List[str] = []
        models_missing: List[str] = []

        for model_name in model_list:
            raw_response, status, key = self.get_model_response(
                dataset_name=dataset_name,
                example=example,
                model_name=model_name,
                run_id=run_id,
            )
            responses[model_name] = raw_response
            statuses[model_name] = status
            keys[model_name] = key

            if status == "FOUND":
                models_available.append(model_name)
            else:
                models_missing.append(model_name)

        num_available = len(models_available)
        num_target = len(model_list)
        incomplete_cache = num_available < num_target

        return {
            "responses": responses,
            "statuses": statuses,
            "keys": keys,
            "models_available": models_available,
            "models_missing": models_missing,
            "num_available": num_available,
            "num_target": num_target,
            "incomplete_cache": incomplete_cache,
        }

    def get_cache_stats(self, dataset_name: str) -> Dict[str, Any]:
        """
        Returns basic summary statistics for a benchmark cache file.

        Args:
            dataset_name (str): Benchmark name.

        Returns:
            dict with entry count, distinct models, and file size in MB.
        """
        cache = self.load_dataset_cache(dataset_name)
        cache_path = os.path.join(self.cache_dir, f"reliability_benchmark_cache_{dataset_name}.pkl")
        file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)

        distinct_models = sorted(list(set(k[3] for k in cache.keys())))

        return {
            "dataset_name": dataset_name,
            "total_entries": len(cache),
            "num_distinct_models": len(distinct_models),
            "distinct_models": distinct_models,
            "file_size_mb": round(file_size_mb, 2),
        }
