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


def _fake_run(game, variant, seed, values, players=2, budget=600.0):
    curve = [{"iterations": 100 * 2**i, "train_seconds": 30.0 * 2**i, "exploitability": v,
              "nashconv_mean": v} for i, v in enumerate(values)]
    return {"game": {"name": game, "players": players}, "spec": {"variant": variant},
            "time_budget": budget, "curve": curve, "_file": f"{game}__{variant}__s{seed}.json"}


def test_seed_band_median_and_range():
    runs = [_fake_run("g", "cfr", s, vals) for s, vals in
            enumerate([[1.0, 0.5], [3.0, 0.1], [2.0, 0.3]])]
    x, y, lo, hi = plots.seed_band(runs, "iterations", "exploitability")
    assert list(x) == [100, 200]
    assert list(y) == [2.0, 0.3] and list(lo) == [1.0, 0.1] and list(hi) == [3.0, 0.5]


def test_seed_band_refuses_mismatched_schedules():
    runs = [_fake_run("g", "cfr", 0, [1.0, 0.5]), _fake_run("g", "cfr", 1, [1.0])]
    with pytest.raises(ValueError, match="checkpoint count"):
        plots.seed_band(runs, "iterations", "exploitability")


def test_fig_convergence_lists_exactly_the_runs_it_drew(tmp_path):
    runs = [_fake_run("leduc-2p-cap1", v, s, [0.1, 0.05, 0.02])
            for v in ("cfr", "cfr+") for s in range(3)]
    runs.append(_fake_run("leduc-2p-cap1", "cfr", 9, [0.1, 0.05, 0.02], budget=3600.0))
    paths = plots.fig_convergence(runs, games=("leduc-2p-cap1",), out_dir=tmp_path)
    sources = (tmp_path / "fig_convergence.sources.txt").read_text().split()
    assert len(sources) == 6 and not any("s9" in s for s in sources)
    assert [p.suffix for p in paths] == [".png", ".pdf", ".txt"]


def test_fig_convergence_skips_cleanly_without_data(tmp_path):
    with pytest.raises(NotImplementedError):
        plots.fig_convergence([], out_dir=tmp_path)


def test_fig_convergence_on_committed_results(tmp_path):
    paths = plots.fig_convergence(out_dir=tmp_path)
    assert all(p.stat().st_size > 0 for p in paths)
    for line in (tmp_path / "fig_convergence.sources.txt").read_text().split():
        assert (RESULTS.parent / line).is_file()
