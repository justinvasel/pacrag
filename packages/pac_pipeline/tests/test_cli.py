from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pac_pipeline.cli import build_parser, main


class TestBuildParser:
    def test_opinion_id_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--opinion-id", "25-FC-118"])
        assert args.opinion_id == "25-FC-118"

    def test_year_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--year", "2025"])
        assert args.year == 2025

    def test_reparse_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--reparse"])
        assert args.reparse is True

    def test_data_dir_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data-dir", "/tmp/data"])
        assert args.data_dir == Path("/tmp/data")

    def test_db_path_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--db-path", "/tmp/test.duckdb"])
        assert args.db_path == Path("/tmp/test.duckdb")

    def test_lit_binary_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--lit-binary", "/usr/local/bin/lit"])
        assert args.lit_binary == "/usr/local/bin/lit"

    def test_verbose_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-v"])
        assert args.verbose is True

    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.opinion_id is None
        assert args.year is None
        assert args.reparse is False
        assert args.data_dir is None
        assert args.db_path is None
        assert args.lit_binary is None
        assert args.verbose is False


class TestMain:
    def test_calls_parse_all_with_settings_defaults(self, tmp_path: Path) -> None:
        with (
            patch("pac_pipeline.cli.parse_all", return_value=(0, 0)) as mock_parse,
            patch("pac_pipeline.cli.settings") as mock_settings,
            patch("sys.argv", ["pac-parse"]),
        ):
            mock_settings.data_dir = tmp_path / "data"
            mock_settings.duckdb_path = tmp_path / "test.duckdb"
            mock_settings.lit_binary = "lit-from-settings"
            main()
            mock_parse.assert_called_once()
            call_kwargs = mock_parse.call_args[1]
            assert call_kwargs["data_dir"] == tmp_path / "data"
            assert call_kwargs["db_path"] == tmp_path / "test.duckdb"
            assert call_kwargs["lit_binary"] == "lit-from-settings"
            assert call_kwargs["year"] is None
            assert call_kwargs["opinion_id"] is None
            assert call_kwargs["reparse"] is False

    def test_cli_flags_override_settings(self, tmp_path: Path) -> None:
        with (
            patch("pac_pipeline.cli.parse_all", return_value=(0, 0)) as mock_parse,
            patch("pac_pipeline.cli.settings") as mock_settings,
            patch(
                "sys.argv",
                [
                    "pac-parse",
                    "--data-dir",
                    str(tmp_path / "cli_data"),
                    "--year",
                    "2024",
                ],
            ),
        ):
            mock_settings.data_dir = tmp_path / "settings_data"
            main()
            call_kwargs = mock_parse.call_args[1]
            assert call_kwargs["data_dir"] == tmp_path / "cli_data"
            assert call_kwargs["year"] == 2024

    def test_opinion_id_filter(self, tmp_path: Path) -> None:
        with (
            patch("pac_pipeline.cli.parse_all", return_value=(1, 0)) as mock_parse,
            patch.dict("os.environ", {}),
            patch("sys.argv", ["pac-parse", "--opinion-id", "25-FC-118"]),
        ):
            main()
            call_kwargs = mock_parse.call_args[1]
            assert call_kwargs["opinion_id"] == "25-FC-118"

    def test_reparse_flag(self, tmp_path: Path) -> None:
        with (
            patch("pac_pipeline.cli.parse_all", return_value=(0, 0)) as mock_parse,
            patch.dict("os.environ", {}),
            patch("sys.argv", ["pac-parse", "--reparse"]),
        ):
            main()
            call_kwargs = mock_parse.call_args[1]
            assert call_kwargs["reparse"] is True
