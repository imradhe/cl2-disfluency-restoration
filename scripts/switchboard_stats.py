#!/usr/bin/env python3
"""Run the statistics notebook from the command line and export tables/figures.

The notebook is the single source of the analysis and tagset rules. No kernel,
training code, or model download is needed when the corpus files are local.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "results/switchboard_stats",
                        help="Directory for CSV tables and PNG plots.")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Keep both corpus lookup and Matplotlib's cache inside this workspace.
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    notebook = root / "notebooks/switchboard_stats.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    table_count = figure_count = 0
    cell_number = 0

    def display(*objects):
        nonlocal table_count
        for obj in objects:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                table_count += 1
                filename = f"table_{table_count:02d}_cell_{cell_number:02d}.csv"
                obj.to_csv(out / filename)
                print(f"\n{filename}\n{obj.to_string()}")
            elif obj is not None:
                print(obj)

    def show(*_args, **_kwargs):
        nonlocal figure_count
        for number in plt.get_fignums():
            figure_count += 1
            plt.figure(number).savefig(out / f"figure_{figure_count:02d}.png", bbox_inches="tight")
        plt.close("all")

    plt.show = show
    namespace = {"__name__": "__stats__", "display": display}
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        for cell_number, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            filename = f"{notebook}:cell-{cell_number}"
            tree = ast.parse(source, filename=filename)
            # Match a notebook's display of its final expression.
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = tree.body[-1]
                tree.body[-1] = ast.copy_location(
                    ast.Expr(ast.Call(ast.Name("display", ast.Load()), [last.value], [])), last)
                ast.fix_missing_locations(tree)
            print(f"\n--- Cell {cell_number} ---", flush=True)
            exec(compile(tree, filename, "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
        plt.close("all")
    # Stable filenames for the key results; retain every mispronunciation pair,
    # including the rows beyond the notebook's 20-row preview.
    for name in ("coverage", "cov_sum", "cat_tab", "tsk_audit", "click_contexts",
                 "mispron_summary", "mispron_examples", "headline"):
        namespace[name].to_csv(out / f"{name}.csv")
    print(f"\nSaved {table_count} tables and {figure_count} figures to {out}")


if __name__ == "__main__":
    main()
