"""Tests for the figure plumbing in pokercfr/analysis/plots.py."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from pokercfr.analysis import plots

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_save_writes_png_and_pdf(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [3, 1, 2])
    paths = plots.save(fig, "demo", out_dir=tmp_path)
    assert [p.name for p in paths] == ["demo.png", "demo.pdf"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert paths[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert paths[1].read_bytes()[:5] == b"%PDF-"


def test_save_is_byte_identical_across_reruns(tmp_path):
    def draw(out):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [3, 1, 2])
        return plots.save(fig, "demo", out_dir=out)

    a = draw(tmp_path / "a")
    b = draw(tmp_path / "b")
    for x, y in zip(a, b, strict=True):
        assert x.read_bytes() == y.read_bytes(), f"{x.name} differs between reruns"


@pytest.mark.parametrize("bad", ["", "fig.png", "sub/fig", "..\\fig"])
def test_save_rejects_non_stem_names(tmp_path, bad):
    fig, _ = plt.subplots()
    with pytest.raises(ValueError):
        plots.save(fig, bad, out_dir=tmp_path)
    plt.close(fig)


def test_load_results_reads_committed_runs_with_filenames():
    records = plots.load_results()
    assert records, "results/ has no run files"
    names = [r["_file"] for r in records]
    assert names == sorted(names)
    assert all((RESULTS / n).is_file() for n in names)


def test_load_results_rejects_a_broken_record(tmp_path):
    good = sorted(RESULTS.glob("*.json"))[0]
    shutil.copy(good, tmp_path / good.name)
    bad = json.loads(good.read_text())
    del bad["curve"]
    (tmp_path / "broken.json").write_text(json.dumps(bad))
    with pytest.raises(plots.ResultError, match="broken.json"):
        plots.load_results(tmp_path)


def test_main_skips_unwritten_figures_and_exits_zero(capsys):
    assert plots.main() == 0
    out = capsys.readouterr().out
    assert out.count("\n") == len(plots.FIGURES)
    assert all(line.split()[0] in {"OK", "SKIP"} for line in out.splitlines())
