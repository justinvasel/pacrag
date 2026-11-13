from __future__ import annotations

from pathlib import Path

import pytest
from pac_eval.gold_set import load_gold_set


class TestLoadGoldSet:
    def test_loads_valid_yaml(self, tmp_gold_set_path: Path) -> None:
        gs = load_gold_set(tmp_gold_set_path)
        assert gs.version == "1.0"
        assert len(gs.entries) == 3
        assert gs.entries[0].id == "q001"
        assert gs.entries[0].expected_opinions == ["24-FC-42", "23-FC-15"]

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="gold set not found"):
            load_gold_set(tmp_path / "nonexistent.yaml")

    def test_rejects_non_mapping(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_gold_set(p)

    def test_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        content = """\
version: "1.0"
entries:
  - id: "q001"
    question: "Q1?"
    expected_opinions: ["25-FC-118"]
    ideal_answer: "A1."
    key_facts: ["F1"]
  - id: "q001"
    question: "Q2?"
    expected_opinions: ["24-FC-42"]
    ideal_answer: "A2."
    key_facts: ["F2"]
"""
        p = tmp_path / "dupes.yaml"
        p.write_text(content)
        with pytest.raises(Exception, match="duplicate entry ids"):
            load_gold_set(p)

    def test_rejects_missing_fields(self, tmp_path: Path) -> None:
        content = """\
version: "1.0"
entries:
  - id: "q001"
    question: "Q1?"
"""
        p = tmp_path / "incomplete.yaml"
        p.write_text(content)
        with pytest.raises((ValueError, Exception)):
            load_gold_set(p)
