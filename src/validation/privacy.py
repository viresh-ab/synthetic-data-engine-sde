import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class PrivacyMetric:
    name: str
    value: float
    passed: bool
    risk_level: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        symbols = {"low": "✓", "medium": "⚠", "high": "⚠⚠", "critical": "✗"}
        return f"{symbols.get(self.risk_level, '?')} {self.name}: {self.value:.3f} ({self.risk_level} risk)"


@dataclass
class PrivacyReport:
    overall_risk: str
    passed: bool
    metrics: List[PrivacyMetric] = field(default_factory=list)
    k_anonymity_score: Optional[int] = None
    reid_risk_score: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def add_metric(self, m: PrivacyMetric):
        self.metrics.append(m)

    def get_critical_risks(self):
        return [m for m in self.metrics if m.risk_level == "critical"]

    def get_high_risks(self):
        return [m for m in self.metrics if m.risk_level == "high"]


# =========================================================
# K-ANONYMITY
# =========================================================

class KAnonymity:

    def __init__(self, k: int = 5):
        self.k = k

    def check(self, data: pd.DataFrame, quasi_identifiers: List[str]):

        valid_cols = [c for c in quasi_identifiers if c in data.columns]
        if not valid_cols:
            return False, 0, {"error": "no valid quasi identifiers"}

        grouped = data.groupby(valid_cols).size()

        min_group = int(grouped.min())
        below_k = int((grouped < self.k).sum())

        details = {
            "min_group_size": min_group,
            "required_k": self.k,
            "groups_below_k": below_k,
            "total_groups": int(len(grouped)),
        }

        return min_group >= self.k, min_group, details

    def suggest_quasi_identifiers(self, data: pd.DataFrame, max_cardinality: int = 100):

        suggestions = []
        n = len(data)

        for c in data.columns:
            u = data[c].nunique()
            if 2 < u < max_cardinality:
                r = u / n
                if 0.01 < r < 0.5:
                    suggestions.append(c)

        return suggestions


# =========================================================
# RE-IDENTIFICATION RISK
# =========================================================

class ReIdentificationRisk:

    def assess(self, data: pd.DataFrame, quasi_identifiers: Optional[List[str]]):

        if not quasi_identifiers:
            quasi_identifiers = list(data.columns)

        valid_cols = [c for c in quasi_identifiers if c in data.columns]
        if not valid_cols:
            return 0.0, "low", {}

        grouped = data.groupby(valid_cols).size()

        uniqueness_ratio = float((grouped == 1).sum() / len(data))

        entropies = []
        for c in valid_cols:
            vc = data[c].value_counts(normalize=True)
            ent = -np.sum(vc * np.log2(vc + 1e-12))
            max_ent = np.log2(len(vc)) if len(vc) > 1 else 1
            entropies.append(ent / max_ent)

        distinctiveness = float(np.mean(entropies)) if entropies else 0.0

        combos = 1
        for c in valid_cols:
            combos *= data[c].nunique()
        combo_risk = min(1.0, combos / len(data))

        risk = uniqueness_ratio * 0.4 + distinctiveness * 0.3 + combo_risk * 0.3

        if risk < 0.25:
            level = "low"
        elif risk < 0.5:
            level = "medium"
        elif risk < 0.75:
            level = "high"
        else:
            level = "critical"

        return float(risk), level, {
            "uniqueness_score": uniqueness_ratio,
            "distinctiveness_score": distinctiveness,
            "combinatorial_risk": combo_risk,
        }


# =========================================================
# UNIQUENESS ANALYZER (SAFE GROUPBY FIX)
# =========================================================

class UniquenessAnalyzer:

    def __init__(self, rare_threshold: int = 3):
        self.rare_threshold = rare_threshold

    def analyze(self, data: pd.DataFrame, columns: Optional[List[str]] = None):

        if columns is None:
            columns = list(data.columns)

        valid_cols = [c for c in columns if c in data.columns]

        # ✅ critical fix — never groupby([])
        group_cols = valid_cols if valid_cols else list(data.columns)

        grouped = data.groupby(group_cols).size()

        unique_count = int((grouped == 1).sum())
        rare_count = int((grouped <= self.rare_threshold).sum())

        results = {
            "total_records": len(data),
            "total_unique_combinations": int(len(grouped)),
            "unique_records": unique_count,
            "rare_records": rare_count,
            "unique_percentage": float(unique_count / len(grouped) * 100) if len(grouped) else 0.0,
            "rare_percentage": float(rare_count / len(grouped) * 100) if len(grouped) else 0.0,
            "columns_analyzed": valid_cols,
        }

        col_stats = {}
        for c in valid_cols:
            col_stats[c] = {
                "unique_count": int(data[c].nunique()),
                "unique_ratio": float(data[c].nunique() / len(data)),
            }

        results["column_uniqueness"] = col_stats

        return results

    def find_unique_records(self, data: pd.DataFrame, columns: List[str]):

        valid_cols = [c for c in columns if c in data.columns]
        group_cols = valid_cols if valid_cols else list(data.columns)

        grouped = data.groupby(group_cols).size()
        unique_keys = grouped[grouped == 1].index

        mask = data.set_index(group_cols).index.isin(unique_keys)
        return data[mask]


# =========================================================
# PRIVACY VALIDATOR
# =========================================================

class PrivacyValidator:

    def __init__(self, config=None):

        if config and hasattr(config, "validation"):
            self.k = config.validation.k_anonymity
        else:
            self.k = 5

        self.k_checker = KAnonymity(self.k)
        self.reid = ReIdentificationRisk()
        self.unique = UniquenessAnalyzer()

    def validate(self, synthetic: pd.DataFrame,
                 reference: Optional[pd.DataFrame] = None,
                 quasi_identifiers: Optional[List[str]] = None):

        report = PrivacyReport("unknown", False)

        if quasi_identifiers is None:
            quasi_identifiers = self.k_checker.suggest_quasi_identifiers(
                synthetic)

        # ---------- K ANON ----------
        if quasi_identifiers:
            ok, actual_k, details = self.k_checker.check(
                synthetic, quasi_identifiers)
            report.k_anonymity_score = actual_k

            level = "low" if ok else "high"
            report.add_metric(PrivacyMetric(
                "k_anonymity",
                actual_k / self.k if self.k else 0,
                ok,
                level,
                details
            ))

        # ---------- REID ----------
        score, level, details = self.reid.assess(synthetic, quasi_identifiers)
        report.reid_risk_score = score

        report.add_metric(PrivacyMetric(
            "reidentification_risk",
            1 - score,
            score < 0.5,
            level,
            details
        ))

        # ---------- UNIQUENESS ----------
        uniq = self.unique.analyze(synthetic, quasi_identifiers)
        unique_ratio = uniq["unique_percentage"] / 100

        uniq_level = (
            "low" if unique_ratio < 0.1 else
            "medium" if unique_ratio < 0.3 else
            "high" if unique_ratio < 0.5 else
            "critical"
        )

        report.add_metric(PrivacyMetric(
            "uniqueness",
            1 - unique_ratio,
            unique_ratio < 0.3,
            uniq_level,
            uniq
        ))

        # ---------- LEAKAGE ----------
        if reference is not None:
            leak_score, leak_level = self._check_data_leakage(
                reference, synthetic)
            report.add_metric(PrivacyMetric(
                "data_leakage",
                leak_score,
                leak_score > 0.95,
                leak_level,
                {"leakage_ratio": 1 - leak_score}
            ))

        # ---------- OVERALL RISK ----------
        risks = [m.risk_level for m in report.metrics]
        report.overall_risk = (
            "critical" if "critical" in risks else
            "high" if "high" in risks else
            "medium" if "medium" in risks else
            "low"
        )

        report.passed = report.overall_risk in ("low", "medium")

        # ---------- RECOMMENDATIONS ----------
        report.recommendations = self._generate_recommendations(report)

        # ---------- SUMMARY ----------
        report.summary = {
            "total_metrics": len(report.metrics),
            "critical_risks": len(report.get_critical_risks()),
            "high_risks": len(report.get_high_risks()),
            "records_analyzed": len(synthetic),
            "quasi_identifiers": quasi_identifiers,
        }

        return report

    # =====================================================

    def _check_data_leakage(self, reference, synthetic):

        ref_set = set(map(tuple, reference.values))
        syn_set = set(map(tuple, synthetic.values))

        matches = len(ref_set & syn_set)
        ratio = matches / len(synthetic)
        score = 1 - ratio

        if ratio == 0:
            level = "low"
        elif ratio < 0.05:
            level = "medium"
        elif ratio < 0.1:
            level = "high"
        else:
            level = "critical"

        return score, level

    # =====================================================

    def _generate_recommendations(self, report: PrivacyReport):

        recs = []

        if report.k_anonymity_score and report.k_anonymity_score < self.k:
            recs.append(
                "Increase k-anonymity by generalizing quasi-identifiers.")

        if report.reid_risk_score and report.reid_risk_score > 0.5:
            recs.append(
                "Reduce re-identification risk via noise or aggregation.")

        if report.get_critical_risks():
            recs.append(
                "Critical privacy risks detected — do not release data.")

        if report.get_high_risks() and not report.get_critical_risks():
            recs.append(
                "High privacy risks detected — mitigation recommended.")

        if not recs:
            recs.append("Privacy validation passed.")

        return recs
