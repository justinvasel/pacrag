from __future__ import annotations

from pac_pipeline.extract_cli import build_parser


class TestBuildParser:
    def test_default_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.opinion_id is None
        assert args.reextract is False
        assert args.limit is None
        assert args.data_dir is None
        assert args.db_path is None
        assert args.verbose is False

    def test_opinion_id_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--opinion-id", "25-FC-118"])
        assert args.opinion_id == "25-FC-118"

    def test_reextract_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--reextract"])
        assert args.reextract is True

    def test_limit_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--limit", "50"])
        assert args.limit == 50

    def test_verbose_short_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-v"])
        assert args.verbose is True
