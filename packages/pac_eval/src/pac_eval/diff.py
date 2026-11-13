from __future__ import annotations

from pac_eval.models import Scorecard

_METRIC_NAMES = [
    ("Recall@8", "mean_recall_at_8"),
    ("Recall@16", "mean_recall_at_16"),
    ("MRR", "mean_mrr"),
    ("Citation Precision", "mean_citation_precision"),
    ("Fact Coverage", "mean_fact_coverage"),
    ("Hedging Score", "mean_hedging_score"),
]

_PER_QUESTION_METRICS = [
    ("recall_at_8", "retrieval"),
    ("recall_at_16", "retrieval"),
    ("mrr", "retrieval"),
    ("citation_precision", "answer"),
    ("fact_coverage", "answer"),
    ("hedging_score", "answer"),
]


def diff_scorecards(
    scorecard_a: Scorecard,
    scorecard_b: Scorecard,
    threshold: float = 0.01,
) -> str:
    lines: list[str] = []
    lines.append(f"Eval Diff: {scorecard_a.run_id} vs {scorecard_b.run_id}")
    lines.append("=" * len(lines[0]))
    lines.append("")

    sa = scorecard_a.summary
    sb = scorecard_b.summary

    lines.append("Summary:")
    lines.append(f"  {'':24s} {'Run A':>10s} {'Run B':>10s} {'Delta':>10s}")
    regressions_found = False

    for label, attr in _METRIC_NAMES:
        val_a = getattr(sa, attr)
        val_b = getattr(sb, attr)
        delta = val_b - val_a
        flag = ""
        if delta < -threshold:
            flag = "  [REGRESSION]"
            regressions_found = True
        lines.append(f"  {label:24s} {val_a:10.3f} {val_b:10.3f} {delta:+10.3f}{flag}")

    lines.append("")
    lines.append(
        f"  Questions: {sa.questions_evaluated} -> {sb.questions_evaluated}"
        f"  Errors: {sa.questions_with_errors} -> {sb.questions_with_errors}"
    )

    results_a = {r.entry_id: r for r in scorecard_a.results}
    results_b = {r.entry_id: r for r in scorecard_b.results}
    common_ids = sorted(set(results_a) & set(results_b))

    per_q_regressions: list[str] = []
    for eid in common_ids:
        ra = results_a[eid]
        rb = results_b[eid]
        for metric, source in _PER_QUESTION_METRICS:
            if source == "retrieval":
                va = getattr(ra.retrieval, metric)
                vb = getattr(rb.retrieval, metric)
            else:
                va = getattr(ra.answer, metric) if ra.answer else None
                vb = getattr(rb.answer, metric) if rb.answer else None

            if va is None or vb is None:
                continue
            delta = vb - va
            if delta < -threshold:
                per_q_regressions.append(f"  {eid}: {metric}  {va:.3f} -> {vb:.3f}")

    if per_q_regressions:
        lines.append("")
        lines.append("Per-Question Regressions:")
        lines.extend(per_q_regressions)

    only_a = sorted(set(results_a) - set(results_b))
    only_b = sorted(set(results_b) - set(results_a))
    if only_a or only_b:
        lines.append("")
        if only_a:
            lines.append(f"  Only in Run A: {', '.join(only_a)}")
        if only_b:
            lines.append(f"  Only in Run B: {', '.join(only_b)}")

    if not regressions_found and not per_q_regressions:
        lines.append("")
        lines.append("No regressions detected.")

    return "\n".join(lines)
