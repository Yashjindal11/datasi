"""Reference vs current comparison (train/test, last month vs this month, ...)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from datasi.configuration import Config
from datasi.core.dataset import Dataset
from datasi.core.pipeline import build_config, dataset_info
from datasi.detectors.base import BaseDetector, pct, tiered
from datasi.findings import Category, Finding, Severity
from datasi.loaders import load
from datasi.reports.model import DetectorRun, Report
from datasi.schema import ColumnType, Schema, infer_schema
from datasi.statistics import distribution as dist

DOMAIN_ROWS = 20_000


class DriftDetector(BaseDetector):
    name = "drift"
    category = Category.DISTRIBUTION
    description = (
        "Compares a current dataset with a reference: schema, missingness, numeric "
        "distributions (KS D, Wasserstein/IQR, PSI), categorical distributions (JSD, PSI, "
        "unseen categories), value ranges, and an overall domain classifier."
    )
    assumptions = (
        "Severity uses effect sizes (KS D, PSI, JSD) with configurable thresholds, never "
        "p-values alone: with large samples every tiny difference is 'significant'."
    )
    limitations = (
        "Univariate metrics miss shifts in joint distributions; the domain classifier "
        "partly covers this. PSI depends on binning (10 reference-quantile bins)."
    )

    def compare(
        self,
        ref: Dataset,
        cur: Dataset,
        ref_schema: Schema,
        cur_schema: Schema,
        config: Config,
        sections: dict[str, Any],
    ) -> list[Finding]:
        t = config.thresholds
        out: list[Finding] = []
        rcols, ccols = set(ref.columns), set(cur.columns)
        missing_cols = sorted(rcols - ccols)
        extra_cols = sorted(ccols - rcols)
        if missing_cols:
            out.append(
                self.finding(
                    "drift.columns_missing",
                    Severity.HIGH,
                    f"{len(missing_cols)} reference column(s) absent from current data",
                    f"Columns {missing_cols} exist in the reference but not in the current dataset.",
                    columns=missing_cols,
                    category=Category.SCHEMA,
                    rule="Any column removed between datasets -> high.",
                    suggestion="Check for renamed or dropped fields upstream.",
                )
            )
        if extra_cols:
            out.append(
                self.finding(
                    "drift.columns_added",
                    Severity.WARNING,
                    f"{len(extra_cols)} new column(s) in current data",
                    f"Columns {extra_cols} are not in the reference dataset.",
                    columns=extra_cols,
                    category=Category.SCHEMA,
                    rule="Any added column -> warning.",
                )
            )
        rows: list[dict[str, Any]] = []
        for col in [c for c in ref.columns if c in ccols]:
            rt, ct = ref_schema[col], cur_schema[col]
            row: dict[str, Any] = {
                "column": col,
                "type_reference": rt.column_type.value,
                "type_current": ct.column_type.value,
                "missing_reference": rt.missing_ratio,
                "missing_current": ct.missing_ratio,
            }
            sev = Severity.INFO
            if rt.column_type != ct.column_type and ColumnType.UNKNOWN not in (
                rt.column_type,
                ct.column_type,
            ):
                out.append(
                    self.finding(
                        "drift.type_changed",
                        Severity.HIGH,
                        f"Type of '{col}' changed",
                        f"'{col}' is {rt.column_type.value} in the reference but {ct.column_type.value} in the current data.",
                        columns=[col],
                        category=Category.SCHEMA,
                        evidence={
                            "reference": rt.column_type.value,
                            "current": ct.column_type.value,
                            "dtype_reference": rt.dtype,
                            "dtype_current": ct.dtype,
                        },
                        rule="Inferred semantic type differs -> high.",
                        suggestion="Check the upstream schema and parsing.",
                    )
                )
            gap = ct.missing_ratio - rt.missing_ratio
            if abs(gap) >= t.temporal_shift_gap:
                out.append(
                    self.finding(
                        "drift.missingness",
                        Severity.WARNING,
                        f"Missingness of '{col}' changed",
                        f"'{col}' is missing in {pct(ct.missing_ratio)} of current rows vs {pct(rt.missing_ratio)} of reference rows.",
                        columns=[col],
                        category=Category.MISSINGNESS,
                        evidence={"reference": rt.missing_ratio, "current": ct.missing_ratio},
                        rule=f"Absolute change in missing rate >= temporal_shift_gap ({t.temporal_shift_gap}).",
                        suggestion="Check whether collection of this field changed.",
                    )
                )
            if rt.column_type in (
                ColumnType.IDENTIFIER,
                ColumnType.TEXT,
                ColumnType.CONSTANT,
                ColumnType.UNKNOWN,
            ):
                row["kind"] = "skipped"
                rows.append(row)
                continue
            if rt.column_type == ColumnType.NUMERIC and ct.column_type == ColumnType.NUMERIC:
                from datasi.statistics.strings import to_number

                def num(ds: Dataset, c: str) -> pd.Series:
                    s = ds.frame[c]
                    return s.astype("float64") if pd.api.types.is_numeric_dtype(s) else to_number(s)

                m = dist.compare_numeric(num(ref, col), num(cur, col))
                row.update(
                    {
                        k: m.get(k)
                        for k in (
                            "kind",
                            "ks_statistic",
                            "psi",
                            "wasserstein_iqr",
                            "out_of_range_share",
                            "mean_reference",
                            "mean_current",
                        )
                    }
                )
                d, psi = m.get("ks_statistic"), m.get("psi")
                if d is not None:
                    sev = max(
                        tiered(
                            d,
                            [(t.ks_high, Severity.HIGH), (t.ks_warning, Severity.WARNING)],
                            Severity.INFO,
                        ),
                        tiered(
                            psi or 0,
                            [(t.psi_high, Severity.HIGH), (t.psi_warning, Severity.WARNING)],
                            Severity.INFO,
                        ),
                        key=lambda s: s.rank,
                    )
                    if sev != Severity.INFO:
                        out.append(
                            self.finding(
                                "drift.numeric",
                                sev,
                                f"Distribution shift in '{col}'",
                                f"KS D = {d:.3f}, PSI = {fmt(psi)}, Wasserstein = {m['wasserstein_iqr']:.2f} reference IQRs; "
                                f"median {m['median_reference']:.4g} -> {m['median_current']:.4g}.",
                                columns=[col],
                                evidence=m,
                                rule=(
                                    f"Worse of: KS D >= ks_high ({t.ks_high}) or PSI >= psi_high ({t.psi_high}) -> high; "
                                    f"KS D >= ks_warning ({t.ks_warning}) or PSI >= psi_warning ({t.psi_warning}) -> warning."
                                ),
                                interpretation="The populations differ for this feature; a model trained on the reference may degrade.",
                                suggestion="Check sampling/time windows of both datasets; consider reweighting or retraining.",
                                confidence=0.8
                                if min(m["n_reference"], m["n_current"]) >= 200
                                else 0.5,
                            )
                        )
                    oor = m.get("out_of_range_share") or 0.0
                    if oor >= 0.01:
                        out.append(
                            self.finding(
                                "drift.out_of_range",
                                Severity.WARNING,
                                f"Current values of '{col}' outside the reference range",
                                f"{pct(oor)} of current values fall outside the reference range "
                                f"[{m['range_reference'][0]:.4g}, {m['range_reference'][1]:.4g}].",
                                columns=[col],
                                evidence={
                                    "share": oor,
                                    "range_reference": m["range_reference"],
                                    "range_current": m["range_current"],
                                },
                                rule=">= 1% of current values outside the reference min/max -> warning.",
                                interpretation="Models extrapolate poorly outside the training range.",
                            )
                        )
            elif rt.column_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
                m = dist.compare_categorical(ref.frame[col], cur.frame[col])
                row.update(
                    {
                        k: m.get(k)
                        for k in ("kind", "jsd", "psi", "unseen_categories", "unseen_share")
                    }
                )
                jsd = m.get("jsd")
                if jsd is not None:
                    sev = max(
                        tiered(
                            jsd,
                            [(t.jsd_high, Severity.HIGH), (t.jsd_warning, Severity.WARNING)],
                            Severity.INFO,
                        ),
                        tiered(
                            m.get("psi") or 0,
                            [(t.psi_high, Severity.HIGH), (t.psi_warning, Severity.WARNING)],
                            Severity.INFO,
                        ),
                        key=lambda s: s.rank,
                    )
                    if sev != Severity.INFO:
                        top = m["largest_changes"][0]
                        out.append(
                            self.finding(
                                "drift.categorical",
                                sev,
                                f"Category mix shifted in '{col}'",
                                f"JSD = {jsd:.3f}, PSI = {fmt(m.get('psi'))}; largest change: '{top['label']}' "
                                f"{pct(top['reference'])} -> {pct(top['current'])}.",
                                columns=[col],
                                evidence=m,
                                rule=(
                                    f"Worse of: JSD >= jsd_high ({t.jsd_high}) or PSI >= psi_high -> high; "
                                    f"JSD >= jsd_warning ({t.jsd_warning}) or PSI >= psi_warning -> warning."
                                ),
                                interpretation="The category proportions differ between the datasets.",
                                suggestion="Check whether both datasets were sampled the same way.",
                                confidence=0.8,
                            )
                        )
                    if (m.get("unseen_share") or 0) >= 0.001:
                        out.append(
                            self.finding(
                                "drift.unseen_categories",
                                Severity.WARNING,
                                f"Unseen categories in '{col}'",
                                f"{m['unseen_categories']} label(s) covering {pct(m['unseen_share'])} of current rows never occur in the reference.",
                                columns=[col],
                                evidence={
                                    "unseen_categories": m["unseen_categories"],
                                    "unseen_share": m["unseen_share"],
                                },
                                rule="Unseen labels covering >= 0.1% of current rows -> warning.",
                                interpretation="Encoders fitted on the reference will not know these labels.",
                                suggestion="Decide how unseen labels should be handled before deployment.",
                            )
                        )
            row["severity"] = sev.value
            rows.append(row)
        auc = domain_classifier_auc(ref, cur, ref_schema, config.performance.random_state)
        sections["drift"] = {"columns": rows, "domain_classifier_auc": auc}
        if auc is not None and auc >= 0.6:
            out.append(
                self.finding(
                    "drift.domain_classifier",
                    Severity.HIGH if auc >= 0.8 else Severity.WARNING,
                    "A classifier can tell the datasets apart",
                    f"A gradient-boosted classifier separates reference from current rows with cross-validated AUC = {auc:.3f} (0.5 = indistinguishable).",
                    evidence={"auc": auc},
                    rule="Domain-classifier AUC >= 0.8 -> high; >= 0.6 -> warning. Identifier and text columns are excluded.",
                    interpretation="The joint distribution differs, possibly in ways no single column shows.",
                    suggestion="Inspect the per-column metrics; consider time-based splits or reweighting.",
                    confidence=0.7,
                )
            )
        return out


def fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def domain_classifier_auc(ref: Dataset, cur: Dataset, schema: Schema, seed: int) -> float | None:
    """Cross-validated AUC of a model predicting 'is current' (adversarial validation)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from threadpoolctl import threadpool_limits

    cols = [
        c
        for c in schema.columns
        if c in cur.frame
        and schema[c].column_type
        in (ColumnType.NUMERIC, ColumnType.CATEGORICAL, ColumnType.BOOLEAN, ColumnType.DATETIME)
    ]
    if not cols:
        return None
    a = ref.sample(DOMAIN_ROWS, seed)[cols]
    b = cur.sample(DOMAIN_ROWS, seed)[cols]
    if len(a) < 50 or len(b) < 50:
        return None
    both = pd.concat([a, b], ignore_index=True)
    feats = {}
    cat_mask = []
    for c in cols:
        s = both[c]
        kind = schema[c].column_type
        if kind == ColumnType.DATETIME:
            continue  # time ranges differ by construction in temporal splits
        if kind == ColumnType.NUMERIC and pd.api.types.is_numeric_dtype(s):
            feats[c] = s.astype("float64")
            cat_mask.append(False)
        else:
            codes, _ = pd.factorize(s.astype("string"), sort=True)
            feats[c] = pd.Series(
                np.where(codes < 0, np.nan, np.minimum(codes, 250)), dtype="float64"
            )
            cat_mask.append(True)
    if not feats:
        return None
    x = pd.DataFrame(feats).to_numpy()
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    model = HistGradientBoostingClassifier(
        max_iter=100, categorical_features=np.array(cat_mask), random_state=seed
    )
    # Many OpenMP threads on small data is ~15x slower (oversubscription on macOS).
    with threadpool_limits(limits=2):
        scores = cross_val_score(model, x, y, cv=3, scoring="roc_auc")
    return float(np.mean(scores))


def inspect_train_test(
    reference: Any,
    current: Any,
    *,
    config: Config | dict[str, Any] | str | None = None,
    target: str | None = None,
    reference_name: str | None = None,
    current_name: str | None = None,
) -> Report:
    """Compare two datasets (e.g. train vs test) and report distribution shifts."""
    started = time.perf_counter()
    cfg = build_config(config, target=target)
    ref = load(
        reference,
        name=reference_name or (None if not isinstance(reference, pd.DataFrame) else "reference"),
    )
    cur = load(
        current, name=current_name or (None if not isinstance(current, pd.DataFrame) else "current")
    )
    kw = {
        "parse_ratio": cfg.thresholds.type_parse_ratio,
        "unique_threshold": cfg.thresholds.identifier_unique_ratio,
        "id_columns": cfg.id_columns,
    }
    rs, cs = infer_schema(ref.frame, **kw), infer_schema(cur.frame, **kw)  # type: ignore[arg-type]
    sections: dict[str, Any] = {}
    det = DriftDetector()
    t0 = time.perf_counter()
    findings = det.compare(ref, cur, rs, cs, cfg, sections)
    findings.sort(key=lambda f: -f.severity.rank)
    if cfg.target and cfg.target in ref.frame and cfg.target in cur.frame:
        sections["target_drift"] = dist.compare_categorical(
            ref.frame[cfg.target], cur.frame[cfg.target]
        )
    return Report(
        kind="comparison",
        dataset=dataset_info(cur, cfg),
        reference=dataset_info(ref, cfg),
        target=cfg.target,
        config=cfg.model_dump(mode="json"),
        schema=cs.to_dict(),
        findings=findings,
        sections=sections,
        detectors=[
            DetectorRun(
                name="drift",
                status="ok",
                seconds=round(time.perf_counter() - t0, 4),
                findings=len(findings),
                info=det.info(),
            )
        ],
        runtime_seconds=round(time.perf_counter() - started, 4),
    )


compare = inspect_train_test
