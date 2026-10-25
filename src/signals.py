"""
Provenance: [FORK NEW]

Purpose:
Extracts observable uncertainty signals from raw LLM ensemble generations in strict
blind mode (zero access to ground truth targets or human curation statuses).
Computes the 5 mathematical dimensions of consensus:
1. Majority Ratio M(x)
2. Normalized Shannon Entropy Confidence V(x)
3. Chain-of-Thought Semantic Similarity Sim(x) (via local TF-IDF)
4. Response Length Stability LS(x)
5. Verbalized Epistemic Hedging Score Hed(x)
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import get_parse_fn


# Curated lexicon of explicit epistemic uncertainty terms
HEDGING_LEXICON = {
    "perhaps",
    "maybe",
    "unclear",
    "ambiguous",
    "assume",
    "assuming",
    "assumption",
    "cannot determine",
    "can't determine",
    "it is possible",
    "it's possible",
    "possibly",
    "alternatively",
    "may be",
    "might be",
    "not certain",
    "uncertain",
    "probable",
    "probably",
    "could be",
    "seems like",
    "approximate",
    "approximately",
}

# Empirical reference bounds calibrated strictly on the Core Development Suite (GSM8K, SVAMP, DROP - N=850)
# Ensures Sim(x) and LS(x) occupy the full [0.0, 1.0] dynamic range alongside M(x) and V(x).
SIM_FLOOR = 0.15
SIM_CEILING = 0.60

LS_FLOOR = 0.10
LS_CEILING = 0.60



def extract_reasoning_body(raw_output: str) -> str:
    """
    Extracts the Chain-of-Thought reasoning explanation from a model output,
    stripping introductory wrappers and trailing answer declarations.

    Hierarchy:
    1. Look for explicit XML reasoning tags (<cot>...</cot> or <thinking>...</thinking>).
    2. Split at the last occurrence of 'Answer:' or '\\boxed{' and keep preceding text.
    3. Fallback to full cleaned raw text if splitting is not applicable.

    Args:
        raw_output (str): Full generation string from LLM.

    Returns:
        str: Extracted reasoning text.
    """
    if not raw_output or not isinstance(raw_output, str):
        return ""

    text = raw_output.strip()

    # 1. Structured thinking tags (DeepSeek-R1, Gemini, Claude Thinking)
    cot_match = re.search(r"<cot>(.*?)</cot>", text, flags=re.DOTALL | re.IGNORECASE)
    if cot_match:
        return cot_match.group(1).strip()

    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL | re.IGNORECASE)
    if thinking_match:
        return thinking_match.group(1).strip()

    # 2. Split before explicit answer declaration
    answer_patterns = [r"\n\s*answer\s*:\s*", r"\n\s*final answer\s*:\s*", r"\s*answer\s*:\s*"]
    for pat in answer_patterns:
        splits = re.split(pat, text, flags=re.IGNORECASE)
        if len(splits) > 1:
            # Everything before the last answer declaration is reasoning
            reasoning_part = "".join(splits[:-1]).strip()
            if len(reasoning_part) > 20:
                return reasoning_part

    # 3. Fallback: return full text stripped of leading/trailing whitespace
    return text


def compute_majority_ratio(predictions: List[str]) -> Tuple[float, str, int]:
    """
    Computes the Majority Ratio M(x) across ensemble predictions.

    Formula:
        M(x) = max_c N_c(x) / K

    Args:
        predictions (List[str]): List of parsed canonical predictions from K models.

    Returns:
        Tuple[float, str, int]: (M(x) in [1/K, 1.0], modal_prediction, modal_count).
    """
    k = len(predictions)
    if k == 0:
        return 0.0, "", 0

    counts = Counter(predictions)
    modal_pred, modal_count = counts.most_common(1)[0]
    majority_ratio = float(modal_count / k)
    return majority_ratio, modal_pred, modal_count


def compute_entropy_confidence(predictions: List[str]) -> Tuple[float, float]:
    """
    Computes Normalized Shannon Entropy Confidence V(x) across vote distribution.

    Formula:
        H(x) = - sum_{c} p(c) * log2(p(c))
        H_norm(x) = H(x) / log2(K)
        V(x) = 1.0 - H_norm(x)

    Args:
        predictions (List[str]): List of parsed predictions from K models.

    Returns:
        Tuple[float, float]: (V(x) in [0.0, 1.0], raw_entropy H(x) in bits).
    """
    k = len(predictions)
    if k <= 1:
        return 1.0, 0.0

    counts = Counter(predictions)
    entropy = 0.0
    for count in counts.values():
        p = count / k
        if p > 0.0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(k)
    norm_entropy = entropy / max_entropy if max_entropy > 0.0 else 0.0
    # Numerical clip to guarantee strict [0.0, 1.0] bounds
    norm_entropy = max(0.0, min(1.0, norm_entropy))
    entropy_conf = 1.0 - norm_entropy
    return entropy_conf, entropy


def compute_cot_similarity(
    reasoning_texts: List[str],
    min_text_len: int = 20,
    return_raw: bool = False,
) -> Any:
    """
    Computes the Semantic Coherence Sim(x) across model Chain-of-Thought traces
    using dynamically fitted local TF-IDF and pairwise cosine similarity.

    Formula:
        Raw Sim(x) = (2 / (N * (N - 1))) * sum_{i < j} cos(v_i, v_j)
        Scaled Sim(x) = clip((Raw Sim - SIM_FLOOR) / (SIM_CEILING - SIM_FLOOR), 0.0, 1.0)

    Calibrated on the Core Development Suite (GSM8K, SVAMP, DROP - N=850) where
    raw cross-model cosine similarity spans [0.15, 0.60]. This normalization puts
    Sim(x) on an equal [0.0, 1.0] footing with M(x) and V(x).

    Args:
        reasoning_texts (List[str]): Extracted reasoning traces from models.
        min_text_len (int): Minimum length to consider a reasoning trace informative.
        return_raw (bool): If True, returns tuple (scaled_sim, raw_sim).

    Returns:
        float or Tuple[float, float]: Scaled Sim(x) in [0.0, 1.0] (and optionally raw_sim).
    """
    # Filter valid informative reasoning texts
    valid_texts = [t for t in reasoning_texts if isinstance(t, str) and len(t.strip()) >= min_text_len]
    n_valid = len(valid_texts)

    # Edge cases: fewer than 2 valid reasoning texts
    if n_valid < 2:
        default_val = 1.0 if n_valid == 1 else 0.5
        return (default_val, default_val) if return_raw else default_val

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r"(?u)\b\w+\b",
        )
        tfidf_matrix = vectorizer.fit_transform(valid_texts)

        # Compute K x K cosine similarity matrix
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Extract unique pairwise off-diagonal elements (i < j)
        upper_tri_indices = np.triu_indices(n_valid, k=1)
        pairwise_sims = sim_matrix[upper_tri_indices]

        if len(pairwise_sims) == 0:
            raw_sim = 1.0
        else:
            raw_sim = float(np.mean(pairwise_sims))
            raw_sim = max(0.0, min(1.0, raw_sim))

        scaled_sim = max(0.0, min(1.0, (raw_sim - SIM_FLOOR) / (SIM_CEILING - SIM_FLOOR)))
        return (scaled_sim, raw_sim) if return_raw else scaled_sim
    except Exception:
        fallback = 0.5
        return (fallback, fallback) if return_raw else fallback


def compute_length_stability(
    reasoning_texts: List[str],
    min_text_len: int = 1,
    return_raw: bool = False,
) -> Any:
    """
    Computes Response Length Stability LS(x) from character length dispersion.

    Formula:
        CV_L(x) = std(lengths) / (mean(lengths) + eps)
        Raw LS(x) = exp(-CV_L(x)) in (0.0, 1.0]
        Scaled LS(x) = clip((Raw LS - LS_FLOOR) / (LS_CEILING - LS_FLOOR), 0.0, 1.0)

    Calibrated on the Core Development Suite (GSM8K, SVAMP, DROP - N=850) where
    raw length stability spans [0.10, 0.60] due to inherent cross-model verbosity divergence.

    Args:
        reasoning_texts (List[str]): List of model reasoning strings.
        min_text_len (int): Minimum length to consider.
        return_raw (bool): If True, returns tuple (scaled_ls, raw_ls).

    Returns:
        float or Tuple[float, float]: Scaled LS(x) in (0.0, 1.0] (and optionally raw_ls).
    """
    lengths = [len(t.strip()) for t in reasoning_texts if isinstance(t, str) and len(t.strip()) >= min_text_len]
    if len(lengths) < 2:
        return (1.0, 1.0) if return_raw else 1.0

    mean_l = float(np.mean(lengths))
    std_l = float(np.std(lengths))
    eps = 1e-6

    cv = std_l / (mean_l + eps)
    raw_ls = float(np.exp(-cv))
    raw_ls = max(0.0, min(1.0, raw_ls))

    scaled_ls = max(0.0, min(1.0, (raw_ls - LS_FLOOR) / (LS_CEILING - LS_FLOOR)))
    return (scaled_ls, raw_ls) if return_raw else scaled_ls


def compute_hedging_score(reasoning_texts: List[str], theta: float = 2.0) -> float:
    """
    Computes Verbalized Epistemic Hedging Score Hed(x).
    Higher score means FEWER hedging terms (greater confidence).

    Formula:
        Hed(x) = 1.0 - min(1.0, (total_hedging_count / K) / theta)

    Args:
        reasoning_texts (List[str]): Model reasoning strings.
        theta (float): Normalization scaling constant (hedging terms per model to reach 0 confidence).

    Returns:
        float: Hed(x) in [0.0, 1.0].
    """
    k = len(reasoning_texts)
    if k == 0:
        return 1.0

    total_hedges = 0
    for text in reasoning_texts:
        if not text or not isinstance(text, str):
            continue
        text_lower = text.lower()
        for term in HEDGING_LEXICON:
            # Word-boundary matching for single words, direct substring for multi-word phrases
            if " " in term:
                total_hedges += text_lower.count(term)
            else:
                matches = re.findall(rf"\b{re.escape(term)}\b", text_lower)
                total_hedges += len(matches)

    avg_hedges = total_hedges / k
    hed_score = 1.0 - min(1.0, avg_hedges / theta)
    return max(0.0, min(1.0, float(hed_score)))


def extract_signals_for_ensemble(
    dataset_name: str,
    example: Dict[str, Any],
    ensemble_responses: Dict[str, Optional[str]],
    parsing_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level extractor computing all 5 uncertainty signals in strict blind mode.

    HERMETIC ISOLATION GUARANTEE:
    This function NEVER reads or accesses `platinum_target` or `cleaning_status`.
    All signals depend solely on observable model outputs and the problem text.

    Args:
        dataset_name (str): Identifier of the benchmark (e.g. 'gsm8k', 'svamp', 'drop').
        example (Dict[str, Any]): Test instance record containing 'question' and parsing metadata.
        ensemble_responses (Dict[str, Optional[str]]): Map of {model_name: raw_output_text}.
        parsing_strategy (Optional[str]): Parsing strategy ('math', 'text', etc.).
                                         If None, inferred from example metadata.

    Returns:
        Dict[str, Any]: Comprehensive signal bundle containing:
            - 'majority_ratio': float M(x) in [1/K, 1.0]
            - 'entropy_confidence': float V(x) in [0.0, 1.0]
            - 'cot_similarity': calibrated float Sim(x) in [0.0, 1.0]
            - 'raw_cot_similarity': unscaled TF-IDF cosine similarity
            - 'length_stability': calibrated float LS(x) in [0.0, 1.0]
            - 'raw_length_stability': unscaled exponential length stability
            - 'hedging_score': float Hed(x) in [0.0, 1.0]
            - 'signal_vector': np.ndarray of shape (5,) with [M, V, Sim, LS, Hed]
            - 'y_maj': modal prediction
            - 'modal_count': count of modal prediction
            - 'num_models': total available models K
            - 'parsed_predictions': dict of {model_name: parsed_answer}
            - 'reasoning_traces': dict of {model_name: reasoning_text}
            - 'parse_error_count': number of parsing errors encountered
            - 'has_missing_cache': boolean indicating if any model was missing
    """
    strategy = parsing_strategy or example.get("platinum_parsing_strategy", "text")
    parse_fn = get_parse_fn(strategy)

    parsed_preds: Dict[str, str] = {}
    reasoning_traces: Dict[str, str] = {}
    parse_errors = 0
    missing_cache = False

    for model_name, raw_output in ensemble_responses.items():
        if raw_output is None or raw_output == "MISSING_IN_CACHE":
            missing_cache = True
            parsed_preds[model_name] = "MISSING_IN_CACHE"
            reasoning_traces[model_name] = ""
            continue

        try:
            parsed = parse_fn(raw_output)
            parsed_preds[model_name] = str(parsed)
        except Exception:
            parse_errors += 1
            parsed_preds[model_name] = "PARSING_ERROR"

        reasoning_traces[model_name] = extract_reasoning_body(raw_output)

    # Active predictions for consensus calculations
    predictions_list = list(parsed_preds.values())
    reasoning_list = list(reasoning_traces.values())

    # 1. Majority Ratio M(x)
    maj_ratio, y_maj, modal_count = compute_majority_ratio(predictions_list)

    # 2. Entropy Confidence V(x)
    entropy_conf, raw_entropy = compute_entropy_confidence(predictions_list)

    # 3. CoT Semantic Similarity Sim(x) (calibrated and raw)
    cot_sim, raw_cot_sim = compute_cot_similarity(reasoning_list, return_raw=True)

    # 4. Length Stability LS(x) (calibrated and raw)
    len_stability, raw_len_stability = compute_length_stability(reasoning_list, return_raw=True)

    # 5. Epistemic Hedging Score Hed(x)
    hedging = compute_hedging_score(reasoning_list)

    signal_vec = np.array([maj_ratio, entropy_conf, cot_sim, len_stability, hedging], dtype=float)

    return {
        "majority_ratio": maj_ratio,
        "entropy_confidence": entropy_conf,
        "raw_entropy": raw_entropy,
        "cot_similarity": cot_sim,
        "raw_cot_similarity": raw_cot_sim,
        "length_stability": len_stability,
        "raw_length_stability": raw_len_stability,
        "hedging_score": hedging,
        "signal_vector": signal_vec,
        "y_maj": y_maj,
        "modal_count": modal_count,
        "num_models": len(predictions_list),
        "parsed_predictions": parsed_preds,
        "reasoning_traces": reasoning_traces,
        "parse_error_count": parse_errors,
        "has_missing_cache": missing_cache,
    }
