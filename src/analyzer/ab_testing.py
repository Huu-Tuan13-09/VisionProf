"""A/B Testing and statistical significance comparison engine."""

import logging
import math
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ABComparisonEngine:
    """Computes statistical significance, effect sizes, and speedup for A/B runs."""

    @staticmethod
    def compute_speedup(latencies_a: List[float], latencies_b: List[float]) -> float:
        """Calculate overall speedup (Mean_A / Mean_B) based on total model iterations."""
        if not latencies_a or not latencies_b:
            return 1.0
        mean_a = sum(latencies_a) / len(latencies_a)
        mean_b = sum(latencies_b) / len(latencies_b)
        if mean_b <= 0.0:
            return 1.0
        return round(mean_a / mean_b, 4)

    @staticmethod
    def compute_cliffs_delta(sample_a: List[float], sample_b: List[float]) -> float:
        """Calculate Cliff's Delta non-parametric effect size."""
        n_a, n_b = len(sample_a), len(sample_b)
        if n_a == 0 or n_b == 0:
            return 0.0

        greater = 0
        lesser = 0
        for x in sample_a:
            for y in sample_b:
                if x > y:
                    greater += 1
                elif x < y:
                    lesser += 1

        delta = (greater - lesser) / float(n_a * n_b)
        return round(delta, 4)

    @classmethod
    def compare_runs(
        cls,
        latencies_a: List[float],
        latencies_b: List[float],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Perform Mann-Whitney U test, Cliff's delta, and speedup summary."""
        speedup = cls.compute_speedup(latencies_a, latencies_b)
        cliffs_delta = cls.compute_cliffs_delta(latencies_a, latencies_b)

        p_value = 1.0
        statistic = 0.0
        if HAS_SCIPY and len(latencies_a) >= 5 and len(latencies_b) >= 5:
            res = stats.mannwhitneyu(latencies_a, latencies_b, alternative="two-sided")
            p_value = float(res.pvalue)
            statistic = float(res.statistic)

        is_significant = p_value < alpha
        return {
            "speedup": speedup,
            "cliffs_delta": cliffs_delta,
            "mann_whitney_u": statistic,
            "p_value": p_value,
            "statistically_significant": is_significant,
            "sample_count_a": len(latencies_a),
            "sample_count_b": len(latencies_b),
        }
