from __future__ import annotations

from pac_eval.metrics_answer import compute_citation_precision, compute_hedging_score


class TestCitationPrecision:
    def test_all_valid(self) -> None:
        text = "The agency violated APRA (25-FC-118, p. 3) and (24-FC-42, p. 1)."
        precision, total, valid = compute_citation_precision(
            text, frozenset({"25-FC-118", "24-FC-42"})
        )
        assert precision == 1.0
        assert total == 2
        assert valid == 2

    def test_mixed_valid_invalid(self) -> None:
        text = "See (25-FC-118, p. 3) and (99-FC-999, p. 1)."
        precision, total, valid = compute_citation_precision(text, frozenset({"25-FC-118"}))
        assert precision == 0.5
        assert total == 2
        assert valid == 1

    def test_no_citations(self) -> None:
        text = "The agency should provide records."
        precision, total, valid = compute_citation_precision(text, frozenset({"25-FC-118"}))
        assert precision == 1.0
        assert total == 0
        assert valid == 0

    def test_all_invalid(self) -> None:
        text = "See (99-FC-999, p. 1) and (88-FC-888, p. 2)."
        precision, total, valid = compute_citation_precision(text, frozenset({"25-FC-118"}))
        assert precision == 0.0
        assert total == 2
        assert valid == 0

    def test_informal_opinion_citation(self) -> None:
        text = "See (24-INF-05, p. 2)."
        precision, total, _valid = compute_citation_precision(text, frozenset({"24-INF-05"}))
        assert precision == 1.0
        assert total == 1


class TestHedgingScore:
    def test_good_answer_with_disclaimer(self) -> None:
        text = (
            "The PAC found that the agency violated APRA. "
            "Note: PAC opinions are advisory and persuasive only, "
            "not binding on agencies or courts."
        )
        hedging, overclaim, score = compute_hedging_score(text)
        assert hedging is True
        assert overclaim is False
        assert score == 1.0

    def test_no_disclaimer(self) -> None:
        text = "The agency should provide records promptly."
        hedging, overclaim, score = compute_hedging_score(text)
        assert hedging is False
        assert overclaim is False
        assert score == 0.5

    def test_overclaim_without_qualifier(self) -> None:
        text = "The law requires agencies to respond. You must comply."
        hedging, overclaim, score = compute_hedging_score(text)
        assert hedging is False
        assert overclaim is True
        assert score == 0.0

    def test_overclaim_with_pac_qualifier_not_flagged(self) -> None:
        text = "The PAC opined that the law requires a timely response."
        _hedging, overclaim, _score = compute_hedging_score(text)
        assert overclaim is False

    def test_disclaimer_plus_overclaim(self) -> None:
        text = "You must provide records. PAC opinions are advisory and persuasive only."
        hedging, overclaim, score = compute_hedging_score(text)
        assert hedging is True
        assert overclaim is True
        assert score == 0.5

    def test_case_insensitive_disclaimer(self) -> None:
        text = "Advisory AND Persuasive Only."
        hedging, _, _ = compute_hedging_score(text)
        assert hedging is True
