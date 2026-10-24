"""
Provenance: [FORK NEW]

Purpose:
Unit test suite for OfflineDataLoader verifying deterministic key reconstruction,
temperature rules, cache lookups, and missing key safety flags.
"""

import unittest
import os
from src.offline_loader import (
    OfflineDataLoader,
    OFFICIAL_PAPER_MODELS,
    REASONING_MODELS,
)


class TestOfflineDataLoader(unittest.TestCase):
    """Tests the deterministic behavior and safety rules of OfflineDataLoader."""

    def setUp(self):
        """Set up loader pointing to local cache directory."""
        self.cache_dir = "./cache"
        self.loader = OfflineDataLoader(cache_dir=self.cache_dir)
        
        # Synthetic example mimicking Hugging Face dataset format
        self.sample_example = {
            "platinum_prompt": "Solve math question. Think step-by-step. Then, provide answer: XXX",
            "platinum_prompt_no_cot": "Solve math question. Then, provide answer: XXX",
            "cleaning_status": "original",
            "platinum_target": ["42"],
        }

    def test_invalid_cache_dir_raises_error(self):
        """Ensures that initializing with a non-existent path fails immediately."""
        with self.assertRaises(FileNotFoundError):
            OfflineDataLoader(cache_dir="./non_existent_cache_directory_12345")

    def test_temperature_mapping(self):
        """Verifies deterministic temperature assignment according to paper rules."""
        # Standard models: temperature = 0.5
        self.assertEqual(self.loader.get_temperature("gpt-4o-2024-11-20"), 0.5)
        self.assertEqual(self.loader.get_temperature("claude-3-5-sonnet"), 0.5)
        self.assertEqual(self.loader.get_temperature("gemini-1.5-pro"), 0.5)
        self.assertEqual(self.loader.get_temperature("meta-llama/Meta-Llama-3.1-70B-Instruct"), 0.5)

        # Reasoning models: temperature = 1.0
        self.assertEqual(self.loader.get_temperature("o1-mini"), 1.0)
        self.assertEqual(self.loader.get_temperature("o1-preview-2024-09-12"), 1.0)
        self.assertEqual(self.loader.get_temperature("o3-mini-2025-01-31-high"), 1.0)

        # Specific thinking mode override: temperature = None
        self.assertIsNone(self.loader.get_temperature("claude-3-7-sonnet-20250219-thinking"))

    def test_prompt_selection_logic(self):
        """Ensures proper prompt routing (CoT vs no-CoT) and o1-specific wording."""
        # Standard model receives prompt with Chain-of-Thought
        prompt_std = self.loader.get_prompt(self.sample_example, "gpt-4o-2024-11-20")
        self.assertIn("Think step-by-step", prompt_std)

        # Native reasoning model receives prompt without Chain-of-Thought
        prompt_deepseek_r1 = self.loader.get_prompt(self.sample_example, "deepseek-r1")
        self.assertNotIn("Think step-by-step", prompt_deepseek_r1)
        self.assertEqual(prompt_deepseek_r1, self.sample_example["platinum_prompt_no_cot"])

        # o1-preview replaces 'Then, provide' with 'Provide'
        prompt_o1 = self.loader.get_prompt(self.sample_example, "o1-preview-2024-09-12")
        self.assertIn("Provide answer: XXX", prompt_o1)
        self.assertNotIn("Then, provide", prompt_o1)

    def test_build_cache_key(self):
        """Validates canonical 4-tuple cache key structure."""
        key = self.loader.build_cache_key(self.sample_example, "gpt-4o-2024-11-20", run_id=0)
        self.assertIsInstance(key, tuple)
        self.assertEqual(len(key), 4)
        self.assertEqual(key[1], 0.5)  # temperature
        self.assertEqual(key[2], 0)    # run_id
        self.assertEqual(key[3], "gpt-4o-2024-11-20")

    def test_real_cache_loading_gsm8k(self):
        """Verifies loading real gsm8k pickle file from disk without network access."""
        cache = self.loader.load_dataset_cache("gsm8k")
        self.assertIsInstance(cache, dict)
        self.assertGreater(len(cache), 0)

        stats = self.loader.get_cache_stats("gsm8k")
        self.assertEqual(stats["dataset_name"], "gsm8k")
        self.assertGreater(stats["total_entries"], 5000)
        self.assertGreater(stats["file_size_mb"], 1.0)
        self.assertIn("gpt-4o-2024-11-20", stats["distinct_models"])

    def test_cache_miss_behavior(self):
        """Confirms that querying an unknown model safely returns 'MISSING_IN_CACHE'."""
        raw_response, status, key = self.loader.get_model_response(
            dataset_name="gsm8k",
            example=self.sample_example,
            model_name="non-existent-mock-model-9999",
        )
        self.assertIsNone(raw_response)
        self.assertEqual(status, "MISSING_IN_CACHE")
        self.assertEqual(key[3], "non-existent-mock-model-9999")

    def test_ensemble_query_structure(self):
        """Checks ensemble query return fields and completeness indicator."""
        # Using a single-model list for rapid testing
        result = self.loader.get_ensemble_responses(
            dataset_name="gsm8k",
            example=self.sample_example,
            model_list=["non-existent-model-a", "non-existent-model-b"],
        )
        self.assertEqual(result["num_target"], 2)
        self.assertEqual(result["num_available"], 0)
        self.assertTrue(result["incomplete_cache"])
        self.assertEqual(len(result["models_missing"]), 2)


if __name__ == "__main__":
    unittest.main()
