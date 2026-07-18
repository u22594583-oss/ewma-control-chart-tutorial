# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a self-contained educational Jupyter notebook repository (not a software package/app) for the STK 795 postgraduate course. Each `*_Tutorial.ipynb` at the repo root is a from-scratch, beginner-oriented tutorial notebook; a notebook's opening markdown cell states its scope. Two need internet when (re)run: `Dog_Breed_CNN_Tutorial.ipynb` (Wikimedia Commons image download — see "Data scraping gotchas" below first) and `Stock_Price_Forecast_Tutorial.ipynb` (Yahoo Finance via `yfinance`).

There is no application code, build system, or package structure, and no test suite. Treat each notebook as the unit of work.

## Environment

No `requirements.txt`/`environment.yml` exists. A `.venv` (Python 3.14, created with `python -m venv`) lives at the repo root — use its interpreter rather than a global/user install. Install with:

```bash
./.venv/Scripts/python.exe -m pip install numpy pandas matplotlib scikit-learn nbformat nbclient ipykernel torch torchvision requests tqdm yfinance plotly
./.venv/Scripts/python.exe -m ipykernel install --user --name python3 --display-name "Python 3"
```

- `nbformat`/`nbclient` are needed for the headless execution workflow below and are not installed by default.
- `torch`/`torchvision` (CPU wheels) are only needed for `Dog_Breed_CNN_Tutorial.ipynb`. **Do not install `tensorflow`/Keras into this venv** — TensorFlow has no published wheel for Python 3.14 (confirmed 2026-07); PyTorch was chosen specifically because it publishes `cp314` Windows wheels and lets this repo keep a single venv. (The sibling `STK 795/CNN SVM` project uses TensorFlow/Keras in its own separate Python 3.12 venv — that constraint doesn't apply here.)
- `yfinance`/`plotly` are only needed for `Stock_Price_Forecast_Tutorial.ipynb` (data download + the interactive fan chart).
- `data/` (the CNN notebook's downloaded image dataset) and `.venv/` are gitignored — never commit either.

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

Run this with `./.venv/Scripts/python.exe`. After executing, scan `nb.cells` for `output_type == "error"` to confirm every cell ran cleanly — there is no other CI/test mechanism in this repo. `Dog_Breed_CNN_Tutorial.ipynb`'s data-collection cell is slow (see below) — pass a generous `timeout` (e.g. 2400+ seconds) rather than the default when executing it.

## Data scraping gotchas (`Dog_Breed_CNN_Tutorial.ipynb`)

- Images are downloaded from `upload.wikimedia.org` via the Commons MediaWiki API. That host **actively rate-limits** (`HTTP 429` + `Retry-After`) once request volume crosses some threshold — observed thresholds varied a lot within one session (anywhere from ~1 req/sec sustained being fine, to a 429 with `Retry-After: 600` after a short burst), so treat any specific delay value as empirical, not a guarantee. The notebook's `fetch_image_bytes` already retries with backoff respecting `Retry-After` (capped) — don't remove that.
- The download function is **idempotent/resumable by design**: `download_breed_images` checks how many images already exist on disk for a breed and resumes from there (`saved = n_existing`), skipping the breed entirely if it already has enough. If a run times out or fails partway through, just re-run it — it will not restart from zero.
- If downloads are running unexpectedly slowly, it's likely the rate limiter reacting to *recent* request volume (possibly from earlier test runs in the same session, not just the current one) rather than anything wrong with the code. Reducing `N_PER_BREED` or simply waiting before retrying are both reasonable responses — don't assume the scraping logic itself is broken.
- Check for stray Python/kernel processes left over from a prior interrupted `nbclient` execution (`Get-Process python`) before starting a new one — a previous run's kernel can keep making requests in the background and contribute to rate-limit contention.

## Authoring conventions established in this notebook

When adding to or creating notebooks in this repo, follow the pattern set by `EWMA_Control_Chart_Tutorial.ipynb`:

- **Build notebooks programmatically**, not by hand-editing `.ipynb` JSON. Write a Python script that constructs cells with `nbformat.v4.new_markdown_cell(...)` / `new_code_cell(...)`, appends them to a `cells` list, and writes the notebook with `nbf.write(nb, path)`. This keeps large LaTeX-heavy notebooks reviewable as plain Python source.
- **Markdown cell sources use raw strings** (`r"""..."""`) so LaTeX backslashes (`\lambda`, `\sigma`, etc.) survive unescaped. **Code cell sources must NOT use `\"\"\"` for docstrings inside these raw strings** — the backslash is preserved literally in the generated cell and produces a `SyntaxError` at execution time. Use single-quoted `'''...'''` docstrings inside code cells instead.
- **Math notation**: inline LaTeX with `$...$`, display equations with `$$...$$`, rendered in markdown cells directly (no LaTeX preamble/packages needed — standard Jupyter MathJax rendering).
- **Reproducibility**: seed a single shared `np.random.default_rng(seed=...)` (`RNG`) once in the setup cell and reuse it throughout, rather than reseeding per cell.
- **Plot style**: a shared `plt.rcParams.update({...})` block and a small named color palette are set once in the setup cell and reused across all figures for visual consistency. The EWMA notebook's palette (`COL_DATA`, `COL_CL`, `COL_LIMITS`, `COL_OOC`) is specific to control charts; the SVM and CNN notebooks instead draw fixed per-category colors from the `dataviz` skill's validated categorical palette (`references/palette.md`) — e.g. the CNN notebook's `BREED_COLOR` dict maps each breed to one fixed hex color, assigned once and never reassigned/cycled, so a color always means the same class everywhere in the notebook.
- **Course notation**: this course's control-chart material (see sibling `STK 795/Programming material/*.sas` scripts) uses "Case K" (known in-control parameters, μ0/σ0 given rather than estimated). When extending this notebook or adding related ones (e.g. Shewhart, CUSUM, MEWMA), match that notation and, where relevant, the course's tuned parameter values (e.g. λ=0.05, L=2.4859, n=5, targeting ARL0≈370) so results are comparable across notebooks.
- **ARL estimation**: no closed-form ARL exists for EWMA; it is estimated by Monte Carlo (simulate many runs, record the run length to first out-of-control signal, average). Use the notebook's `simulate_run_length` / `estimate_arl` helper pattern (asymptotic/steady-state control limits, not exact time-varying ones, per standard ARL-study convention) as the template for similar performance studies.
- **Interactive charts** (`Stock_Price_Forecast_Tutorial.ipynb`'s slider fan chart) use Plotly with `pio.renderers.default = "notebook"`, which embeds the full `plotly.js` bundle in the first figure's output. This keeps the chart interactive (drag/hover/zoom) even without a live kernel or internet connection when the notebook is reopened later, at the cost of a multi-MB notebook file — that file-size increase is expected, not a sign something went wrong. Slider steps use Plotly `restyle` (updating one trace's data directly) rather than `frames`, which is simpler and avoids animation-frame bookkeeping for a "browse N precomputed scenarios" interaction.
