# STK 795 Tutorial Notebooks

A collection of self-contained, beginner-oriented Jupyter notebooks built for the
STK 795 postgraduate course. Each notebook teaches one method **from scratch** — full
math with derivations, intuitive analogies, and figures (several interactive) for every
step of the pipeline — and every random draw is seeded, so re-running a notebook top to
bottom reproduces its results exactly.

## The notebooks

### Statistics foundations

| Notebook | What it teaches |
|---|---|
| `Gaussian_White_Noise_Tutorial.ipynb` | Gaussian white noise from scratch: the normal density and the 68–95–99.7 rule, what "white" means (zero autocorrelation, no memory), a Central Limit Theorem demo, signal-to-noise ratio, and the σ/√n averaging law — with interactive sliders adding noise to four deterministic signals. |

### Statistical process control

| Notebook | What it teaches |
|---|---|
| `EWMA_Control_Chart_Tutorial.ipynb` | The Exponentially Weighted Moving Average control chart: the statistic, mean/variance derivation, exact vs. asymptotic control limits, and Average Run Length (ARL) analysis via Monte Carlo simulation. |
| `MEWMA_Control_Chart_Tutorial.ipynb` | The multivariate EWMA chart: monitoring several correlated variables at once with the $T^2$ statistic. |
| `MEWMA_Interactive_Explorer.html` | A standalone interactive explorer for the MEWMA chart — open directly in a browser, no Python needed. |

### Machine learning

| Notebook | What it teaches |
|---|---|
| `SVM_and_OneClassSVM_Tutorial.ipynb` | Support Vector Machines (margins, the dual problem, the kernel trick) and the One-Class SVM for novelty detection without labels. |
| `Dog_Breed_CNN_Tutorial.ipynb` | A Convolutional Neural Network image classifier, built and trained with PyTorch on a small self-scraped dog-breed image dataset. |
| `Conv1D_Autoencoder_Tutorial.ipynb` | A 1D convolutional autoencoder for anomaly detection on sequential data: convolution/pooling/transposed-convolution math, backprop worked by hand, Adam, and reconstruction-error scoring with a control-chart-style threshold. |
| `Stock_Price_Forecast_Tutorial.ipynb` | A Monte Carlo stock price forecast: Geometric Brownian Motion calibrated from real historical prices, with an interactive fan chart. |

### Where the threads meet

| Notebook | What it teaches |
|---|---|
| `Hybrid_CNN_OneClassSVM_MEWMA_Tutorial.ipynb` | A hybrid process monitor combining all three ideas: a CNN autoencoder extracts features, a One-Class SVM scores them, and a MEWMA chart monitors the scores. |

`conv1d_autoencoder.py` is a plain-script version of the autoencoder tutorial's
pipeline (same seed, identical results, no tutorial prose).

**Suggested order** if you are new to all of it: Gaussian noise → EWMA → MEWMA → SVM →
Dog Breed CNN → 1D Autoencoder → Hybrid. The stock forecast notebook stands alone.

## Setup

Create a virtual environment (Python 3.14) at the repo root and install the
dependencies:

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install numpy pandas matplotlib scikit-learn nbformat nbclient ipykernel torch torchvision requests tqdm yfinance plotly
./.venv/Scripts/python.exe -m ipykernel install --user --name python3 --display-name "Python 3"
```

Notes:

- PyTorch (CPU wheels) is used for the deep-learning notebooks. TensorFlow is **not**
  supported here — it publishes no wheel for Python 3.14.
- Two notebooks need internet access when (re)run: `Dog_Breed_CNN_Tutorial.ipynb`
  (downloads its image dataset from Wikimedia Commons, rate-limited and resumable) and
  `Stock_Price_Forecast_Tutorial.ipynb` (Yahoo Finance via `yfinance`). Everything else
  runs fully offline.
- Notebooks with interactive Plotly charts embed the plotly.js bundle, so the files are
  several MB — this keeps the charts interactive even without a running kernel or an
  internet connection.

## Running

Open any notebook in Jupyter or VS Code (kernel "Python 3" from the venv) and run top
to bottom, or run the autoencoder script directly:

```bash
./.venv/Scripts/python.exe conv1d_autoencoder.py
```
