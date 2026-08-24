"""
ML Drift Detection Engine
==========================
Implements distribution shift detection for both features (data drift)
and model predictions (concept drift / prediction drift).

Algorithms used:
  - Population Stability Index (PSI) — industry standard for feature drift
  - Kolmogorov-Smirnov (KS) test — non-parametric distribution comparison
  - Jensen-Shannon Divergence — symmetric KL divergence
  - Chi-Squared test — categorical feature drift
  - Page-Hinkley test — online sequential change detection

Industry patterns from: Evidently AI, WhyLabs, Arize, Fiddler.
"""

import math
import time
import logging
import datetime
import statistics
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger("drift_detection")


# ---------------------------------------------------------------------------
# Drift Severity & Alert Levels
# ---------------------------------------------------------------------------

class DriftSeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"          # 0.1 < PSI <= 0.2
    CRITICAL = "critical"         # PSI > 0.2 → retrain immediately
    UNKNOWN = "unknown"


@dataclass
class DriftAlert:
    feature_name: str
    drift_type: str               # "data_drift" | "concept_drift" | "prediction_drift"
    severity: DriftSeverity
    test_statistic: float
    threshold: float
    detected_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    recommendation: str = ""
    should_retrain: bool = False


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------

class StatisticalTests:

    @staticmethod
    def psi(
        reference: List[float],
        current: List[float],
        buckets: int = 10,
    ) -> float:
        """
        Population Stability Index.
        PSI < 0.1  : no significant change
        PSI 0.1-0.2: moderate shift, monitor
        PSI > 0.2  : major shift, retrain
        """
        if not reference or not current:
            return 0.0

        eps = 1e-6
        ref_min, ref_max = min(reference), max(reference)
        if ref_max == ref_min:
            return 0.0

        bucket_width = (ref_max - ref_min) / buckets
        ref_counts = [0] * buckets
        cur_counts = [0] * buckets

        for v in reference:
            idx = min(int((v - ref_min) / bucket_width), buckets - 1)
            ref_counts[idx] += 1

        for v in current:
            v_clipped = max(ref_min, min(ref_max, v))
            idx = min(int((v_clipped - ref_min) / bucket_width), buckets - 1)
            cur_counts[idx] += 1

        n_ref, n_cur = len(reference), len(current)
        psi = 0.0
        for r, c in zip(ref_counts, cur_counts):
            ref_pct = max(r / n_ref, eps)
            cur_pct = max(c / n_cur, eps)
            psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)

        return round(psi, 6)

    @staticmethod
    def ks_statistic(
        reference: List[float],
        current: List[float],
    ) -> float:
        """
        Kolmogorov-Smirnov D statistic.
        D > 0.1 at large N indicates distributional shift.
        """
        if not reference or not current:
            return 0.0

        ref_sorted = sorted(reference)
        cur_sorted = sorted(current)
        all_vals = sorted(set(ref_sorted + cur_sorted))

        n_ref, n_cur = len(ref_sorted), len(cur_sorted)
        max_diff = 0.0
        ref_idx = cur_idx = 0

        for v in all_vals:
            while ref_idx < n_ref and ref_sorted[ref_idx] <= v:
                ref_idx += 1
            while cur_idx < n_cur and cur_sorted[cur_idx] <= v:
                cur_idx += 1
            cdf_ref = ref_idx / n_ref
            cdf_cur = cur_idx / n_cur
            max_diff = max(max_diff, abs(cdf_ref - cdf_cur))

        return round(max_diff, 6)

    @staticmethod
    def jensen_shannon_divergence(
        reference: List[float],
        current: List[float],
        buckets: int = 20,
    ) -> float:
        """
        Jensen-Shannon Divergence — symmetric, bounded [0, 1].
        JSD > 0.1 signals meaningful distributional change.
        """
        eps = 1e-10
        if not reference or not current:
            return 0.0

        all_vals = reference + current
        v_min, v_max = min(all_vals), max(all_vals)
        if v_max == v_min:
            return 0.0

        width = (v_max - v_min) / buckets

        def to_hist(vals):
            counts = [0] * buckets
            for v in vals:
                idx = min(int((v - v_min) / width), buckets - 1)
                counts[idx] += 1
            n = sum(counts)
            return [c / n + eps for c in counts]

        p = to_hist(reference)
        q = to_hist(current)
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

        def kl(a, b):
            return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b))

        jsd = (kl(p, m) + kl(q, m)) / 2
        return round(min(1.0, max(0.0, jsd)), 6)

    @staticmethod
    def page_hinkley(
        values: List[float],
        delta: float = 0.005,
        lambda_threshold: float = 50.0,
    ) -> Tuple[bool, float]:
        """
        Page-Hinkley change detection test (online, sequential).
        Returns (change_detected, test_statistic).
        """
        if len(values) < 10:
            return False, 0.0

        cumsum = 0.0
        min_cumsum = float("inf")
        mean_estimate = statistics.mean(values[:max(1, len(values) // 2)])

        for x in values:
            cumsum += x - mean_estimate - delta
            min_cumsum = min(min_cumsum, cumsum)
            ph = cumsum - min_cumsum
            if ph > lambda_threshold:
                return True, round(ph, 4)

        return False, 0.0


# ---------------------------------------------------------------------------
# Drift Monitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Monitors feature distributions and model prediction distributions
    against a reference baseline (typically last week of production traffic).
    """

    # Reference distributions (populated at startup from training data stats)
    _reference_stats: Dict[str, List[float]] = {
        "bm25_score":      [0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.8, 0.6, 2.3,
                            1.1, 0.4, 1.7, 0.7, 2.0, 0.2, 1.4, 0.8, 1.9, 1.0],
        "tfidf_cosine":    [0.1, 0.4, 0.2, 0.6, 0.3, 0.5, 0.15, 0.45, 0.25, 0.55,
                            0.12, 0.38, 0.22, 0.52, 0.35, 0.18, 0.42, 0.28, 0.48, 0.32],
        "ctr_7d":          [0.05, 0.12, 0.08, 0.15, 0.07, 0.10, 0.06, 0.13, 0.09, 0.11,
                            0.04, 0.14, 0.07, 0.10, 0.08, 0.12, 0.05, 0.09, 0.11, 0.06],
        "freshness_score": [0.7, 0.9, 0.8, 0.6, 0.85, 0.75, 0.95, 0.65, 0.80, 0.88,
                            0.72, 0.92, 0.78, 0.68, 0.82, 0.77, 0.87, 0.73, 0.93, 0.83],
        "popularity_score":[70, 90, 80, 95, 75, 85, 92, 78, 88, 82, 79, 91, 76, 86,
                            83, 89, 77, 93, 81, 87],
        "model_score":     [0.3, 0.5, 0.6, 0.4, 0.7, 0.55, 0.45, 0.65, 0.35, 0.75,
                            0.5, 0.6, 0.4, 0.7, 0.55, 0.45, 0.65, 0.35, 0.75, 0.5],
    }

    _alerts_history: List[DriftAlert] = []
    _current_window: Dict[str, List[float]] = {}
    _window_size: int = 100

    @classmethod
    def record_prediction(cls, feature_name: str, value: float) -> None:
        """Record a live prediction value for online drift monitoring."""
        if feature_name not in cls._current_window:
            cls._current_window[feature_name] = []
        cls._current_window[feature_name].append(value)
        # Keep rolling window
        if len(cls._current_window[feature_name]) > cls._window_size:
            cls._current_window[feature_name] = cls._current_window[feature_name][-cls._window_size:]

    @classmethod
    def detect_feature_drift(cls, feature_name: str, current_values: Optional[List[float]] = None) -> DriftAlert:
        """
        Run PSI + KS + JSD tests on a feature distribution.
        Returns a DriftAlert with severity classification.
        """
        reference = cls._reference_stats.get(feature_name, [])
        current = current_values or cls._current_window.get(feature_name, [])

        if not reference or len(current) < 5:
            return DriftAlert(
                feature_name=feature_name,
                drift_type="data_drift",
                severity=DriftSeverity.UNKNOWN,
                test_statistic=0.0,
                threshold=0.1,
                recommendation="Insufficient data for drift detection — collect more samples.",
            )

        psi = StatisticalTests.psi(reference, current)
        ks = StatisticalTests.ks_statistic(reference, current)
        jsd = StatisticalTests.jensen_shannon_divergence(reference, current)

        # PSI-based severity (industry standard)
        if psi < 0.1:
            severity = DriftSeverity.NONE
            recommendation = "Feature distribution is stable."
        elif psi < 0.2:
            severity = DriftSeverity.WARNING
            recommendation = f"Moderate drift detected (PSI={psi:.3f}). Investigate source data changes."
        else:
            severity = DriftSeverity.CRITICAL
            recommendation = f"Critical drift detected (PSI={psi:.3f}). Trigger model retraining immediately."

        should_retrain = psi >= 0.2 or ks > 0.15

        alert = DriftAlert(
            feature_name=feature_name,
            drift_type="data_drift",
            severity=severity,
            test_statistic=psi,
            threshold=0.1,
            recommendation=recommendation,
            should_retrain=should_retrain,
        )

        cls._alerts_history.append(alert)
        if severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL):
            logger.warning(f"[DriftMonitor] {feature_name}: PSI={psi:.4f} KS={ks:.4f} JSD={jsd:.4f} → {severity.value}")

        return alert

    @classmethod
    def run_full_drift_report(cls) -> Dict[str, Any]:
        """Run drift detection on all monitored features and return report."""
        import random

        report = {
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "window_size": cls._window_size,
            "features": {},
            "overall_status": "stable",
            "retraining_recommended": False,
        }

        features_to_check = list(cls._reference_stats.keys())
        any_critical = False

        for feature_name in features_to_check:
            ref = cls._reference_stats[feature_name]
            # Simulate slight drift for demo — in production this is real traffic data
            current = [v * (1 + random.gauss(0, 0.05)) for v in ref]
            alert = cls.detect_feature_drift(feature_name, current)

            psi = StatisticalTests.psi(ref, current)
            ks = StatisticalTests.ks_statistic(ref, current)
            jsd = StatisticalTests.jensen_shannon_divergence(ref, current)
            ph_change, ph_stat = StatisticalTests.page_hinkley(current)

            report["features"][feature_name] = {
                "severity": alert.severity.value,
                "psi": psi,
                "ks_statistic": ks,
                "js_divergence": jsd,
                "page_hinkley_change": ph_change,
                "page_hinkley_stat": ph_stat,
                "recommendation": alert.recommendation,
                "should_retrain": alert.should_retrain,
            }

            if alert.severity == DriftSeverity.CRITICAL:
                any_critical = True

        if any_critical:
            report["overall_status"] = "critical"
            report["retraining_recommended"] = True
        elif any(v["severity"] == "warning" for v in report["features"].values()):
            report["overall_status"] = "warning"

        return report

    @classmethod
    def get_alerts_history(cls, limit: int = 50) -> List[Dict]:
        return [
            {
                "feature": a.feature_name,
                "drift_type": a.drift_type,
                "severity": a.severity.value,
                "psi": a.test_statistic,
                "detected_at": a.detected_at,
                "should_retrain": a.should_retrain,
                "recommendation": a.recommendation,
            }
            for a in reversed(cls._alerts_history[-limit:])
        ]
