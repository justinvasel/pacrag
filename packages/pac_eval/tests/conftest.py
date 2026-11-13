from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture()
def tmp_gold_set_path(tmp_path: Path) -> Path:
    content = """\
version: "1.0"
entries:
  - id: "q001"
    question: "Can a city refuse electronic copies of public records?"
    statute_area: "APRA"
    expected_opinions:
      - "24-FC-42"
      - "23-FC-15"
    ideal_answer: "Under APRA, agencies must provide copies in the requested format."
    key_facts:
      - "APRA requires copies in the format requested"
      - "IC 5-14-3-3(b) governs format of production"
  - id: "q002"
    question: "Does a school board violate ODL by discussing contracts in executive session?"
    statute_area: "ODL"
    expected_opinions:
      - "22-FC-88"
    ideal_answer: "The ODL permits executive sessions for collective bargaining strategy."
    key_facts:
      - "IC 5-14-1.5-6.1 enumerates permitted purposes"
      - "Final action must occur in a public meeting"
  - id: "q003"
    question: "What are the response time requirements under APRA?"
    statute_area: "APRA"
    expected_opinions:
      - "25-FC-001"
    ideal_answer: >
      Agencies must respond within a reasonable time,
      generally interpreted as 24 hours.
    key_facts:
      - "Response within a reasonable time is required"
"""
    p = tmp_path / "gold_set.yaml"
    p.write_text(content)
    return p
