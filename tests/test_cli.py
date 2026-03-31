# Copyright (C) 2025 Luigi Pertoldi <gipert@pm.me>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from legendsimflow.cli import (
    _partition,
    snakemake_nersc_batch_cli,
    snakemake_nersc_cli,
)

# ---------------------------------------------------------------------------
# _partition
# ---------------------------------------------------------------------------


def test_partition_even():
    result = _partition(list(range(6)), 3)
    assert result == [[0, 1], [2, 3], [4, 5]]


def test_partition_uneven():
    result = _partition(list(range(7)), 3)
    # 7 // 3 = 2 remainder 1 → first chunk gets 3 items
    assert [len(c) for c in result] == [3, 2, 2]
    assert sum(len(c) for c in result) == 7
    # All original items present exactly once
    assert sorted(x for chunk in result for x in chunk) == list(range(7))


def test_partition_single_chunk():
    items = [1, 2, 3]
    result = _partition(items, 1)
    assert result == [items]


def test_partition_empty():
    assert _partition([], 3) == [[], [], []]


# ---------------------------------------------------------------------------
# snakemake_nersc_cli - argument parsing
# ---------------------------------------------------------------------------


def _parse_nersc(argv):
    """Call parse_known_args via the same parser logic as snakemake_nersc_cli."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("-N", "--nodes", type=int, required=True)
    parser.add_argument("--without-srun", action="store_true")
    return parser.parse_known_args(argv)


def test_nersc_cli_nodes_required():
    args, _ = _parse_nersc(["-N", "4"])
    assert args.nodes == 4


def test_nersc_cli_no_submit_default_false():
    args, _ = _parse_nersc(["-N", "2"])
    assert args.no_submit is False


def test_nersc_cli_no_submit_flag():
    args, _ = _parse_nersc(["-N", "2", "--no-submit"])
    assert args.no_submit is True


def test_nersc_cli_without_srun_default_false():
    args, _ = _parse_nersc(["-N", "2"])
    assert args.without_srun is False


def test_nersc_cli_without_srun_flag():
    args, _ = _parse_nersc(["-N", "2", "--without-srun"])
    assert args.without_srun is True


def test_nersc_cli_extra_args_forwarded():
    _, extra = _parse_nersc(["-N", "2", "--dryrun", "--cores", "4"])
    assert "--dryrun" in extra
    assert "--cores" in extra


# ---------------------------------------------------------------------------
# snakemake_nersc_cli - runtime behaviour
# ---------------------------------------------------------------------------


def test_nersc_cli_raises_for_single_node(tmp_path):
    (tmp_path / "simflow-config.yaml").write_text("{}")
    with (
        patch.object(sys, "argv", ["snakemake-nersc", "-N", "1"]),
        patch("os.getcwd", return_value=str(tmp_path)),
        patch("legendsimflow.cli.Path", wraps=Path) as mock_path,
    ):
        # Redirect Path("./simflow-config.yaml") to tmp_path version
        orig_path = Path

        def patched_path(*args, **kwargs):
            if args == ("./simflow-config.yaml",):
                return tmp_path / "simflow-config.yaml"
            return orig_path(*args, **kwargs)

        mock_path.side_effect = patched_path
        with pytest.raises(ValueError, match="at least 2 nodes"):
            snakemake_nersc_cli()


def test_nersc_cli_raises_missing_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["snakemake-nersc", "-N", "2"])
    with pytest.raises(RuntimeError, match=r"simflow-config\.yaml"):
        snakemake_nersc_cli()


# ---------------------------------------------------------------------------
# snakemake_nersc_batch_cli - argument parsing
# ---------------------------------------------------------------------------


def _parse_batch(argv):
    """Replicate the parser from snakemake_nersc_batch_cli."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("-t", "--time", required=True)
    parser.add_argument("-N", "--nodes", default="1")
    parser.add_argument("-c", "--cpus-per-task", default="256")
    parser.add_argument("-J", "--job-name")
    parser.add_argument("--mail-user")
    return parser.parse_known_args(argv)


def test_batch_cli_time_required():
    args, _ = _parse_batch(["-t", "02:00:00"])
    assert args.time == "02:00:00"


def test_batch_cli_nodes_default():
    args, _ = _parse_batch(["-t", "01:00:00"])
    assert args.nodes == "1"


def test_batch_cli_nodes_custom():
    args, _ = _parse_batch(["-t", "01:00:00", "-N", "4"])
    assert args.nodes == "4"


def test_batch_cli_cpus_per_task_default():
    args, _ = _parse_batch(["-t", "01:00:00"])
    assert args.cpus_per_task == "256"


def test_batch_cli_cpus_per_task_custom():
    args, _ = _parse_batch(["-t", "01:00:00", "-c", "128"])
    assert args.cpus_per_task == "128"


def test_batch_cli_job_name_default_none():
    args, _ = _parse_batch(["-t", "01:00:00"])
    assert args.job_name is None


def test_batch_cli_job_name_custom():
    args, _ = _parse_batch(["-t", "01:00:00", "-J", "my-sim"])
    assert args.job_name == "my-sim"


def test_batch_cli_mail_user_default_none():
    args, _ = _parse_batch(["-t", "01:00:00"])
    assert args.mail_user is None


def test_batch_cli_mail_user_custom():
    args, _ = _parse_batch(["-t", "01:00:00", "--mail-user", "user@example.com"])
    assert args.mail_user == "user@example.com"


def test_batch_cli_no_submit_default_false():
    args, _ = _parse_batch(["-t", "01:00:00"])
    assert args.no_submit is False


def test_batch_cli_extra_args_forwarded():
    _, extra = _parse_batch(["-t", "01:00:00", "--dryrun"])
    assert "--dryrun" in extra


# ---------------------------------------------------------------------------
# snakemake_nersc_batch_cli - runtime behaviour
# ---------------------------------------------------------------------------


def test_batch_cli_raises_missing_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["snakemake-nersc-batch", "-t", "02:00:00"])
    with pytest.raises(RuntimeError, match=r"simflow-config\.yaml"):
        snakemake_nersc_batch_cli()


def test_batch_cli_no_submit_single_node(tmp_path, monkeypatch, capsys):
    (tmp_path / "simflow-config.yaml").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["snakemake-nersc-batch", "-t", "02:00:00", "--no-submit"]
    )
    snakemake_nersc_batch_cli()
    captured = capsys.readouterr()
    assert "sbatch" in captured.out
    assert "02:00:00" in captured.out
    # single node → uses plain snakemake, not snakemake-nersc
    assert "snakemake-nersc" not in captured.out


def test_batch_cli_no_submit_multi_node(tmp_path, monkeypatch, capsys):
    (tmp_path / "simflow-config.yaml").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["snakemake-nersc-batch", "-t", "02:00:00", "-N", "4", "--no-submit"],
    )
    snakemake_nersc_batch_cli()
    captured = capsys.readouterr()
    assert "sbatch" in captured.out
    assert "snakemake-nersc" in captured.out


def test_batch_cli_no_submit_with_job_name(tmp_path, monkeypatch, capsys):
    (tmp_path / "simflow-config.yaml").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "snakemake-nersc-batch",
            "-t",
            "02:00:00",
            "-J",
            "my-job",
            "--no-submit",
        ],
    )
    snakemake_nersc_batch_cli()
    captured = capsys.readouterr()
    assert "my-job" in captured.out


def test_batch_cli_no_submit_with_mail(tmp_path, monkeypatch, capsys):
    (tmp_path / "simflow-config.yaml").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "snakemake-nersc-batch",
            "-t",
            "02:00:00",
            "--mail-user",
            "user@example.com",
            "--no-submit",
        ],
    )
    snakemake_nersc_batch_cli()
    captured = capsys.readouterr()
    assert "user@example.com" in captured.out
    assert "--mail-type" in captured.out
