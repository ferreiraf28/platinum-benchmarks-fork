"""
Provenance: [FORK NEW]

Purpose:
Implements the Selective Auditor Engine, baseline scoring rules, and operational decision
policies for LLM benchmark post-hoc quality assurance and triage.
Converts observable consensus signals into continuous confidence scores S(x) and outputs
binary operational triage decisions:
- ACCEPT_PROVISIONAL: Consensus is sufficiently high and uncompromised; retain without immediate review.
- DEFER_TO_HUMAN: Uncertainty, disagreement, cache incompleteness, or parsing errors detected; queue for human inspection.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression


# Operational decision constants
ACCEPT_PROVISIONAL = "ACCEPT_PROVISIONAL"
DEFER_TO_HUMAN = "DEFER_TO_HUMAN"

# Available scoring modes
MODE_MAJORITY = "majority"
MODE_ENTROPY = "entropy"
MODE_LINEAR_COMPOSITE = "linear_composite"
MODE_CALIBRATED_COMPOSITE = "calibrated_composite"

# Safety Gate modes
GATE_MODE_MAJORITY = "majority"  # Default: triggers only if modal consensus disagrees with legacy label
GATE_MODE_STRICT_UNANIMOUS = "strict_unanimous"  # Paper baseline: triggers if any single model disagrees

# Default weights for the transparent linear composite baseline
# S_comb_baseline = 0.40 * M + 0.35 * V + 0.15 * Sim_scaled + 0.10 * LS_scaled
DEFAULT_LINEAR_WEIGHTS = {
    "majority_ratio": 0.40,
    "entropy_confidence": 0.35,
    "cot_similarity": 0.15,
    "length_stability": 0.10,
}


@dataclass
class AuditorDecision:
    """
    Structured outcome of the selective auditor decision policy on a test instance.

    Attributes:
        decision (str): ACCEPT_PROVISIONAL or DEFER_TO_HUMAN.
        score (float): Continuous confidence score S(x) in [0.0, 1.0].
        tau (float): Decision threshold applied.
        mode (str): Scoring mode used (majority, entropy, linear_composite, etc.).
        modal_prediction (str): Canonical consensus prediction y_maj.
        modal_count (int): Number of models voting for y_maj.
        num_models (int): Total number of evaluated models K.
        is_cache_complete (bool): True if all required model outputs were present.
        has_parse_error (bool): True if any model output failed parsing.
        legacy_disagreement (Optional[bool]): True if any model disagreed with legacy target.
        reason (str): Explicit explanation for the operational decision.
        signal_breakdown (Dict[str, float]): Snapshot of observable signals used.
    """
    decision: str
    score: float
    tau: float
    mode: str
    modal_prediction: str
    modal_count: int
    num_models: int
    is_cache_complete: bool
    has_parse_error: bool
    legacy_disagreement: Optional[bool] = None
    reason: str = ""
    signal_breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def is_accepted(self) -> bool:
        """Convenience property returning True if accepted provisionally."""
        return self.decision == ACCEPT_PROVISIONAL


class SelectiveAuditor:
    """
    Core selective auditor engine computing confidence scores and operational triage decisions.
    
    Guarantees strict blind execution: computes S(x) and decisions without accessing
    ground-truth platinum labels or curation metadata.
    """

    def __init__(
        self,
        linear_weights: Optional[Dict[str, float]] = None,
        calibrated_weights: Optional[np.ndarray] = None,
        calibrated_intercept: float = 0.0,
    ):
        """
        Initializes the auditor with configurable linear and calibrated model parameters.

        Args:
            linear_weights (Optional[Dict[str, float]]): Weights for S_comb_baseline.
            calibrated_weights (Optional[np.ndarray]): Vector of shape (5,) for S_comb_calib.
            calibrated_intercept (float): Scalar intercept for S_comb_calib.
        """
        self.linear_weights = linear_weights or DEFAULT_LINEAR_WEIGHTS.copy()
        self.calibrated_weights = calibrated_weights
        self.calibrated_intercept = calibrated_intercept
        self._logistic_model: Optional[LogisticRegression] = None

    def score(self, signals: Dict[str, Any], mode: str = MODE_LINEAR_COMPOSITE) -> float:
        """
        Computes the continuous confidence score S(x) in [0.0, 1.0] for an instance.

        Supported Modes:
            - 'majority' / 'maj': S_maj(x) = M(x)
            - 'entropy' / 'ent': S_ent(x) = V(x)
            - 'linear_composite' / 'linear': S_comb_baseline(x) = 0.40*M + 0.35*V + 0.15*Sim + 0.10*LS
            - 'calibrated_composite' / 'calibrated': S_comb_calib(x) = sigmoid(w^T s(x) + b)

        Args:
            signals (Dict[str, Any]): Dictionary of signals extracted via src/signals.py.
            mode (str): Name of the scoring formulation.

        Returns:
            float: S(x) bounded in [0.0, 1.0].
        """
        mode_clean = mode.lower().strip()

        # 1. Majority Baseline: S_maj(x) = M(x)
        if mode_clean in ("majority", "maj", MODE_MAJORITY):
            m = float(signals.get("majority_ratio", 0.0))
            return max(0.0, min(1.0, m))

        # 2. Entropy Baseline: S_ent(x) = V(x)
        if mode_clean in ("entropy", "ent", MODE_ENTROPY):
            v = float(signals.get("entropy_confidence", 0.0))
            return max(0.0, min(1.0, v))

        # 3. Transparent Linear Composite Baseline
        if mode_clean in ("linear_composite", "linear", "comb_baseline", MODE_LINEAR_COMPOSITE):
            m = float(signals.get("majority_ratio", 0.0))
            v = float(signals.get("entropy_confidence", 0.0))
            sim = float(signals.get("cot_similarity", 0.0))
            ls = float(signals.get("length_stability", 0.0))

            w_m = self.linear_weights.get("majority_ratio", 0.40)
            w_v = self.linear_weights.get("entropy_confidence", 0.35)
            w_sim = self.linear_weights.get("cot_similarity", 0.15)
            w_ls = self.linear_weights.get("length_stability", 0.10)

            score_val = (w_m * m) + (w_v * v) + (w_sim * sim) + (w_ls * ls)
            return max(0.0, min(1.0, float(score_val)))

        # 4. Calibrated Logistic Composite
        if mode_clean in ("calibrated_composite", "calibrated", "calib", MODE_CALIBRATED_COMPOSITE):
            # Extract 5-dimensional feature vector [M, V, Sim, LS, Hed]
            if "signal_vector" in signals:
                feat = np.asarray(signals["signal_vector"], dtype=float)
            else:
                feat = np.array([
                    float(signals.get("majority_ratio", 0.0)),
                    float(signals.get("entropy_confidence", 0.0)),
                    float(signals.get("cot_similarity", 0.0)),
                    float(signals.get("length_stability", 0.0)),
                    float(signals.get("hedging_score", 1.0)),
                ], dtype=float)

            if self._logistic_model is not None:
                # Use fitted sklearn LogisticRegression model
                prob = float(self._logistic_model.predict_proba(feat.reshape(1, -1))[0, 1])
                return max(0.0, min(1.0, prob))

            if self.calibrated_weights is not None:
                z = float(np.dot(self.calibrated_weights, feat) + self.calibrated_intercept)
                prob = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
                return max(0.0, min(1.0, prob))

            raise ValueError(
                "Calibrated scoring requested but no calibrated weights are fitted or provided. "
                "Call fit_calibrated() or initialize with calibrated_weights."
            )

        raise ValueError(f"Unknown auditor scoring mode: '{mode}'. Must be one of: majority, entropy, linear_composite, calibrated_composite.")

    def fit_calibrated(
        self,
        signal_records: List[Dict[str, Any]],
        is_correct: Union[List[bool], np.ndarray],
        c_reg: float = 1.0,
    ) -> "SelectiveAuditor":
        """
        Fits logistic regression parameters on development instances to parameterize S_comb_calib.

        Args:
            signal_records (List[Dict[str, Any]]): List of signal dictionaries extracted from Dev set.
            is_correct (Union[List[bool], np.ndarray]): Binary labels indicating if modal prediction was correct.
            c_reg (float): Inverse regularization strength for LogisticRegression.

        Returns:
            SelectiveAuditor: self, with updated weights and fitted model.
        """
        x_rows = []
        for sig in signal_records:
            if "signal_vector" in sig:
                x_rows.append(np.asarray(sig["signal_vector"], dtype=float))
            else:
                x_rows.append([
                    float(sig.get("majority_ratio", 0.0)),
                    float(sig.get("entropy_confidence", 0.0)),
                    float(sig.get("cot_similarity", 0.0)),
                    float(sig.get("length_stability", 0.0)),
                    float(sig.get("hedging_score", 1.0)),
                ])

        X = np.asarray(x_rows, dtype=float)
        y = np.asarray(is_correct, dtype=int)

        clf = LogisticRegression(C=c_reg, solver="liblinear", max_iter=1000)
        clf.fit(X, y)

        self._logistic_model = clf
        self.calibrated_weights = clf.coef_[0]
        self.calibrated_intercept = float(clf.intercept_[0])
        return self

    def decide(
        self,
        signals: Dict[str, Any],
        tau: float = 0.85,
        mode: str = MODE_LINEAR_COMPOSITE,
        required_models: Optional[int] = None,
    ) -> AuditorDecision:
        """
        Applies the operational decision policy D_tau(x) to an instance:
        - If cache is incomplete (K_i < K_target) -> DEFER_TO_HUMAN (mandatory).
        - If parsing errors occurred -> DEFER_TO_HUMAN (mandatory).
        - If S(x) >= tau -> ACCEPT_PROVISIONAL.
        - Else S(x) < tau -> DEFER_TO_HUMAN.

        Args:
            signals (Dict[str, Any]): Bundle of signals extracted in blind mode.
            tau (float): Confidence threshold in [0.0, 1.0].
            mode (str): Scoring formulation to apply.
            required_models (Optional[int]): Minimum required models (e.g. 24). If None, defaults to signals['num_models'].

        Returns:
            AuditorDecision: Comprehensive decision object containing verdict and rationale.
        """
        modal_pred = str(signals.get("y_maj", ""))
        modal_count = int(signals.get("modal_count", 0))
        num_models = int(signals.get("num_models", len(signals.get("parsed_predictions", {}))))
        parse_errors = int(signals.get("parse_error_count", 0))
        missing_cache = bool(signals.get("has_missing_cache", False))

        # Check model completeness against target quorum
        target_quorum = required_models if required_models is not None else num_models
        is_cache_complete = (not missing_cache) and (num_models >= target_quorum)
        has_parse_error = parse_errors > 0

        # Compute continuous confidence score S(x)
        score_val = self.score(signals, mode=mode)

        signal_snapshot = {
            "majority_ratio": float(signals.get("majority_ratio", 0.0)),
            "entropy_confidence": float(signals.get("entropy_confidence", 0.0)),
            "cot_similarity": float(signals.get("cot_similarity", 0.0)),
            "length_stability": float(signals.get("length_stability", 0.0)),
            "hedging_score": float(signals.get("hedging_score", 1.0)),
        }

        # 1. Mandatory Deferral: Incomplete Cache Quorum
        if not is_cache_complete:
            return AuditorDecision(
                decision=DEFER_TO_HUMAN,
                score=score_val,
                tau=tau,
                mode=mode,
                modal_prediction=modal_pred,
                modal_count=modal_count,
                num_models=num_models,
                is_cache_complete=False,
                has_parse_error=has_parse_error,
                reason=f"DEFERRED_MANDATORY: Cache quorum incomplete (K={num_models}/{target_quorum})",
                signal_breakdown=signal_snapshot,
            )

        # 2. Mandatory Deferral: Generation Parsing Error
        if has_parse_error:
            return AuditorDecision(
                decision=DEFER_TO_HUMAN,
                score=score_val,
                tau=tau,
                mode=mode,
                modal_prediction=modal_pred,
                modal_count=modal_count,
                num_models=num_models,
                is_cache_complete=True,
                has_parse_error=True,
                reason=f"DEFERRED_MANDATORY: Detected {parse_errors} parsing errors among model generations",
                signal_breakdown=signal_snapshot,
            )

        # 3. Threshold Decision Policy S(x) >= tau
        if score_val >= tau:
            return AuditorDecision(
                decision=ACCEPT_PROVISIONAL,
                score=score_val,
                tau=tau,
                mode=mode,
                modal_prediction=modal_pred,
                modal_count=modal_count,
                num_models=num_models,
                is_cache_complete=True,
                has_parse_error=False,
                reason=f"ACCEPTED_PROVISIONAL: Score {score_val:.4f} >= threshold {tau:.4f}",
                signal_breakdown=signal_snapshot,
            )

        return AuditorDecision(
            decision=DEFER_TO_HUMAN,
            score=score_val,
            tau=tau,
            mode=mode,
            modal_prediction=modal_pred,
            modal_count=modal_count,
            num_models=num_models,
            is_cache_complete=True,
            has_parse_error=False,
            reason=f"DEFERRED_SCORE: Score {score_val:.4f} < threshold {tau:.4f}",
            signal_breakdown=signal_snapshot,
        )


class SafetyGateAuditor:
    """
    Conservative post-hoc triage auditor incorporating an external Legacy Label Safety Gate.

    Industrial Curation Context:
    Curators often inherit an existing benchmark label (y_legacy).
    
    Gate Modes:
    1. 'majority' (Default / Recommended):
       Triggers mandatory human review if and only if the ensemble's MODAL consensus prediction (y_maj)
       disagrees with y_legacy. If the majority agrees with y_legacy, minority model slips (e.g. 1 out
       of 24 models with a formatting typo) do not block auto-acceptance and are evaluated by standard
       confidence thresholding S(x) >= tau.
    2. 'strict_unanimous' (Paper Legacy Baseline):
       Triggers mandatory human review if ANY single model in the ensemble disagrees with y_legacy
       (replicating the paper's original filter E(x) = exists m: y_m != y_legacy).

    Mathematical Isolation Guarantee:
    The Safety Gate acts solely as an external control-flow filter; it never corrupts or alters
    the underlying continuous confidence score S(x).
    """

    def __init__(self, base_auditor: Optional[SelectiveAuditor] = None):
        """
        Initializes the Safety Gate Auditor wrapping a SelectiveAuditor.

        Args:
            base_auditor (Optional[SelectiveAuditor]): Underlying blind auditor engine.
        """
        self.base_auditor = base_auditor or SelectiveAuditor()

    def score(self, signals: Dict[str, Any], mode: str = MODE_LINEAR_COMPOSITE) -> float:
        """Delegates scoring directly to the base auditor."""
        return self.base_auditor.score(signals, mode=mode)

    def decide(
        self,
        signals: Dict[str, Any],
        legacy_target: Optional[Union[str, List[str]]] = None,
        tau: float = 0.85,
        mode: str = MODE_LINEAR_COMPOSITE,
        gate_mode: str = GATE_MODE_MAJORITY,
        required_models: Optional[int] = None,
    ) -> AuditorDecision:
        """
        Applies safety-gated decision policy:
        1. If legacy_target is provided:
           - In 'majority' mode: triggers DEFER_TO_HUMAN if modal prediction y_maj disagrees with y_legacy.
           - In 'strict_unanimous' mode: triggers DEFER_TO_HUMAN if ANY model disagrees with y_legacy.
        2. If safety gate passes (or legacy_target is None):
           - Delegates to SelectiveAuditor.decide(signals, tau, mode, required_models).

        Args:
            signals (Dict[str, Any]): Bundle of signals.
            legacy_target (Optional[Union[str, List[str]]]): Inherited reference label(s).
            tau (float): Acceptance threshold.
            mode (str): Scoring mode.
            gate_mode (str): 'majority' (default) or 'strict_unanimous'.
            required_models (Optional[int]): Model quorum requirement.

        Returns:
            AuditorDecision: Decision with legacy_disagreement flag populated.
        """
        parsed_predictions = signals.get("parsed_predictions", {})
        score_val = self.base_auditor.score(signals, mode=mode)
        modal_pred = str(signals.get("y_maj", ""))
        modal_count = int(signals.get("modal_count", 0))
        num_models = int(signals.get("num_models", len(parsed_predictions)))
        parse_errors = int(signals.get("parse_error_count", 0))
        missing_cache = bool(signals.get("has_missing_cache", False))
        target_quorum = required_models if required_models is not None else num_models
        is_cache_complete = (not missing_cache) and (num_models >= target_quorum)
        has_parse_error = parse_errors > 0

        signal_snapshot = {
            "majority_ratio": float(signals.get("majority_ratio", 0.0)),
            "entropy_confidence": float(signals.get("entropy_confidence", 0.0)),
            "cot_similarity": float(signals.get("cot_similarity", 0.0)),
            "length_stability": float(signals.get("length_stability", 0.0)),
            "hedging_score": float(signals.get("hedging_score", 1.0)),
        }

        # Check Legacy Safety Gate
        if legacy_target is not None:
            # Handle list of acceptable legacy target strings or single string
            legacy_set = {str(t).strip().lower() for t in legacy_target} if isinstance(legacy_target, list) else {str(legacy_target).strip().lower()}
            gate_clean = str(gate_mode).lower().strip()

            if gate_clean in ("majority", "maj", GATE_MODE_MAJORITY):
                # Modal consensus check: does winning prediction agree with legacy label?
                if modal_pred.strip().lower() not in legacy_set:
                    return AuditorDecision(
                        decision=DEFER_TO_HUMAN,
                        score=score_val,
                        tau=tau,
                        mode=mode,
                        modal_prediction=modal_pred,
                        modal_count=modal_count,
                        num_models=num_models,
                        is_cache_complete=is_cache_complete,
                        has_parse_error=has_parse_error,
                        legacy_disagreement=True,
                        reason=f"DEFERRED_MANDATORY: Safety gate triggered (modal consensus '{modal_pred}' disagrees with legacy label)",
                        signal_breakdown=signal_snapshot,
                    )

            elif gate_clean in ("strict_unanimous", "unanimous", GATE_MODE_STRICT_UNANIMOUS):
                # Replicates the paper's original filter: any single model disagreement triggers review
                disagreed_models = [
                    m_name for m_name, pred_val in parsed_predictions.items()
                    if str(pred_val).strip().lower() not in legacy_set
                ]
                if len(disagreed_models) > 0:
                    return AuditorDecision(
                        decision=DEFER_TO_HUMAN,
                        score=score_val,
                        tau=tau,
                        mode=mode,
                        modal_prediction=modal_pred,
                        modal_count=modal_count,
                        num_models=num_models,
                        is_cache_complete=is_cache_complete,
                        has_parse_error=has_parse_error,
                        legacy_disagreement=True,
                        reason=f"DEFERRED_MANDATORY: Safety gate triggered ({len(disagreed_models)} models disagreed with legacy label)",
                        signal_breakdown=signal_snapshot,
                    )
            else:
                raise ValueError(f"Unknown gate_mode: '{gate_mode}'. Must be 'majority' or 'strict_unanimous'.")

        # Safety gate passed (or no legacy target): delegate to base auditor
        base_decision = self.base_auditor.decide(
            signals=signals,
            tau=tau,
            mode=mode,
            required_models=required_models,
        )
        base_decision.legacy_disagreement = False if legacy_target is not None else None
        return base_decision
