# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a self-contained educational Jupyter notebook repository (not a software package/app) for the STK 795 postgraduate course. It currently contains a single notebook:

- `EWMA_Control_Chart_Tutorial.ipynb` — a from-scratch, beginner-oriented walkthrough of the Exponentially Weighted Moving Average (EWMA) control chart: statistic definition, mean/variance derivation, exact vs. asymptotic control limits, simulated in-control/shifted process examples, and Average Run Length (ARL) performance analysis via Monte Carlo simulation.

There is no application code, build system, or package structure, and no test suite. Treat each notebook as the unit of work.

## Environment

No `requirements.txt`/`environment.yml` exists. A `.venv` (Python 3.14, created with `python -m venv`) lives at the repo root — use its interpreter rather than a global/user install. The venv normally has `numpy`, `pandas`, `matplotlib`, `ipykernel` but is missing `nbformat`/`nbclient`, which the execution workflow below requires; install them before executing notebooks:

```bash
./.venv/Scripts/python.exe -m pip install numpy pandas matplotlib nbformat nbclient ipykernel
./.venv/Scripts/python.exe -m ipykernel install --user --name python3 --display-name "Python 3"
```

## Running / executing notebooks

`nbconvert` is not installed in the venv, so the `jupyter` CLI's `jupyter execute` subcommand is unavailable. Execute notebooks headlessly via `nbclient`'s Python API instead:

```python
import nbformat
from nbclient import NotebookClient

path = "EWMA_Control_Chart_Tutorial.ipynb"
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()
nbformat.write(nb, path)  # saves outputs back into the notebook
```

Run this with `./.venv/Scripts/python.exe`. After executing, scan `nb.cells` for `output_type == "error"` to confirm every cell ran cleanly — there is no other CI/test mechanism in this repo.

## Authoring conventions established in this notebook

When adding to or creating notebooks in this repo, follow the pattern set by `EWMA_Control_Chart_Tutorial.ipynb`:

- **Build notebooks programmatically**, not by hand-editing `.ipynb` JSON. Write a Python script that constructs cells with `nbformat.v4.new_markdown_cell(...)` / `new_code_cell(...)`, appends them to a `cells` list, and writes the notebook with `nbf.write(nb, path)`. This keeps large LaTeX-heavy notebooks reviewable as plain Python source.
- **Markdown cell sources use raw strings** (`r"""..."""`) so LaTeX backslashes (`\lambda`, `\sigma`, etc.) survive unescaped. **Code cell sources must NOT use `\"\"\"` for docstrings inside these raw strings** — the backslash is preserved literally in the generated cell and produces a `SyntaxError` at execution time. Use single-quoted `'''...'''` docstrings inside code cells instead.
- **Math notation**: inline LaTeX with `$...$`, display equations with `$$...$$`, rendered in markdown cells directly (no LaTeX preamble/packages needed — standard Jupyter MathJax rendering).
- **Reproducibility**: seed a single shared `np.random.default_rng(seed=...)` (`RNG`) once in the setup cell and reuse it throughout, rather than reseeding per cell.
- **Plot style**: a shared `plt.rcParams.update({...})` block and a small named color palette (`COL_DATA`, `COL_CL`, `COL_LIMITS`, `COL_OOC`) are set once in the setup cell and reused across all figures for visual consistency.
- **Course notation**: this course's control-chart material (see sibling `STK 795/Programming material/*.sas` scripts) uses "Case K" (known in-control parameters, μ0/σ0 given rather than estimated). When extending this notebook or adding related ones (e.g. Shewhart, CUSUM, MEWMA), match that notation and, where relevant, the course's tuned parameter values (e.g. λ=0.05, L=2.4859, n=5, targeting ARL0≈370) so results are comparable across notebooks.
- **ARL estimation**: no closed-form ARL exists for EWMA; it is estimated by Monte Carlo (simulate many runs, record the run length to first out-of-control signal, average). Use the notebook's `simulate_run_length` / `estimate_arl` helper pattern (asymptotic/steady-state control limits, not exact time-varying ones, per standard ARL-study convention) as the template for similar performance studies.
