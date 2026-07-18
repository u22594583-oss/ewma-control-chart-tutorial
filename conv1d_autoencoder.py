'''1D convolutional autoencoder for anomaly detection on a synthetic ECG-like signal.

Base-script version of Conv1D_Autoencoder_Tutorial.ipynb: same pipeline, same seed,
same results -- without the tutorial prose. Run top to bottom with the repo venv:
    ./.venv/Scripts/python.exe conv1d_autoencoder.py
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

SEED = 795
BEAT_PERIOD = 100
PERIOD_JITTER = 5
AMP_SD = 0.05
NOISE_SD = 0.05
BEAT_BUMPS = [(0.18, 0.25, 0.050), (0.42, 1.00, 0.018), (0.68, 0.35, 0.045)]
ANOMALY_TYPES = ["spike", "flat_beat", "oscillation", "variance"]

WINDOW = 128
STRIDE_TRAIN = 16
STRIDE_TEST = 32
BOTTLENECK = 16
EPOCHS = 30
BATCH = 64
LR = 1e-3
ALPHA = 0.01

DEVICE = torch.device("cpu")

COL_SIGNAL = "#4C72B0"
COL_RECON = "#DD8452"
COL_NORMAL = "#55A868"
COL_ANOM = "#C44E52"
COL_THRESH = "#8172B2"


def beat_template(length):
    t = np.arange(length) / length
    beat = np.zeros(length)
    for center, height, width in BEAT_BUMPS:
        beat += height * np.exp(-((t - center) ** 2) / (2 * width ** 2))
    return beat


def generate_normal_signal(n_beats, rng):
    beats = []
    for _ in range(n_beats):
        length = BEAT_PERIOD + int(rng.integers(-PERIOD_JITTER, PERIOD_JITTER + 1))
        beats.append(rng.normal(1.0, AMP_SD) * beat_template(length))
    signal = np.concatenate(beats)
    return signal + rng.normal(0.0, NOISE_SD, size=signal.size)


def inject_anomalies(signal, rng, n_per_type=5):
    sig = signal.copy()
    n_events = n_per_type * len(ANOMALY_TYPES)
    seg = sig.size // n_events
    types = rng.permutation(np.repeat(ANOMALY_TYPES, n_per_type))
    events = []
    for i, kind in enumerate(types):
        start = i * seg + int(rng.integers(seg // 4, seg // 2))
        if kind == "spike":
            length = int(rng.integers(1, 4))
            sig[start:start + length] += rng.uniform(2.0, 3.0)
        elif kind == "flat_beat":
            length = BEAT_PERIOD
            sig[start:start + length] *= 0.15
        elif kind == "oscillation":
            length = 80
            sig[start:start + length] += 0.4 * np.sin(2 * np.pi * 0.2 * np.arange(length))
        elif kind == "variance":
            length = 150
            sig[start:start + length] += rng.normal(0.0, 3 * NOISE_SD, size=length)
        events.append((start, start + length, kind))
    return sig, events


def make_windows(signal, window, stride):
    views = sliding_window_view(signal, window)[::stride]
    starts = np.arange(0, signal.size - window + 1, stride)
    return views.copy(), starts


def label_windows(starts, window, events):
    labels = np.zeros(starts.size, dtype=bool)
    for i, s in enumerate(starts):
        for ev_start, ev_end, _ in events:
            overlap = min(s + window, ev_end) - max(s, ev_start)
            if overlap >= min(16, ev_end - ev_start):
                labels[i] = True
                break
    return labels


class Conv1DAutoencoder(nn.Module):

    def __init__(self, bottleneck=BOTTLENECK):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.enc1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.to_code = nn.Linear(32 * 16, bottleneck)
        self.from_code = nn.Linear(bottleneck, 32 * 16)
        self.dec1 = nn.ConvTranspose1d(32, 32, kernel_size=3, stride=2,
                                       padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2,
                                       padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose1d(16, 16, kernel_size=3, stride=2,
                                       padding=1, output_padding=1)
        self.out = nn.Conv1d(16, 1, kernel_size=3, padding=1)

    def encode(self, x):
        x = self.pool(self.relu(self.enc1(x)))
        x = self.pool(self.relu(self.enc2(x)))
        x = self.pool(self.relu(self.enc3(x)))
        return self.to_code(torch.flatten(x, 1))

    def decode(self, z):
        x = self.relu(self.from_code(z)).view(-1, 32, 16)
        x = self.relu(self.dec1(x))
        x = self.relu(self.dec2(x))
        x = self.relu(self.dec3(x))
        return self.out(x)

    def forward(self, x):
        return self.decode(self.encode(x))


def to_tensor(windows):
    return torch.tensor(windows[:, None, :], dtype=torch.float32)


def train_model(model, train_win, val_win):
    loader_gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(to_tensor(train_win)), batch_size=BATCH,
                              shuffle=True, generator=loader_gen)
    val_loader = DataLoader(TensorDataset(to_tensor(val_win)), batch_size=BATCH,
                            shuffle=False)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    def run_epoch(loader, train):
        model.train(train)
        total, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            with torch.set_grad_enabled(train):
                loss = criterion(model(xb), xb)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        return total / n

    history = {"train": [], "val": []}
    for epoch in range(1, EPOCHS + 1):
        history["train"].append(run_epoch(train_loader, train=True))
        history["val"].append(run_epoch(val_loader, train=False))
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:>2}/{EPOCHS}   train MSE {history['train'][-1]:.5f}   "
                  f"val MSE {history['val'][-1]:.5f}")
    return history


def score_windows(model, windows, batch=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, windows.shape[0], batch):
            xb = to_tensor(windows[i:i + batch]).to(DEVICE)
            outs.append(model(xb).squeeze(1).cpu().numpy())
    recons = np.concatenate(outs)
    return recons, ((windows - recons) ** 2).mean(axis=1)


def evaluate(test_err, test_labels, tau, test_starts, events, flagged):
    tp = int((flagged & test_labels).sum())
    fp = int((flagged & ~test_labels).sum())
    fn = int((~flagged & test_labels).sum())
    tn = int((~flagged & ~test_labels).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"confusion at tau={tau:.4f}:  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"precision {precision:.3f}   recall {recall:.3f}   F1 {f1:.3f}   "
          f"FPR {fp / (fp + tn):.3%}")
    for kind in ANOMALY_TYPES:
        evs = [ev for ev in events if ev[2] == kind]
        detected = 0
        for ev_start, ev_end, _ in evs:
            overlap = np.array([min(s + WINDOW, ev_end) - max(s, ev_start)
                                for s in test_starts])
            covering = overlap >= min(16, ev_end - ev_start)
            detected += bool(flagged[covering].any())
        print(f"  {kind:<12} {detected}/{len(evs)} events detected")


def main():
    rng = np.random.default_rng(seed=SEED)
    torch.manual_seed(SEED)

    train_signal = generate_normal_signal(300, rng)
    val_signal = generate_normal_signal(60, rng)
    test_signal = generate_normal_signal(200, rng)
    test_signal, events = inject_anomalies(test_signal, rng)

    # standardize with train statistics only
    mu, sd = train_signal.mean(), train_signal.std()
    train_win, _ = make_windows((train_signal - mu) / sd, WINDOW, STRIDE_TRAIN)
    val_win, _ = make_windows((val_signal - mu) / sd, WINDOW, STRIDE_TRAIN)
    test_win, test_starts = make_windows((test_signal - mu) / sd, WINDOW, STRIDE_TEST)
    test_labels = label_windows(test_starts, WINDOW, events)

    model = Conv1DAutoencoder().to(DEVICE)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    history = train_model(model, train_win, val_win)

    # threshold from validation (normal-only) errors, then score the test stream once
    _, val_err = score_windows(model, val_win)
    tau = float(np.quantile(val_err, 1 - ALPHA))
    print(f"threshold tau = {tau:.4f}  ({1 - ALPHA:.0%} quantile of validation errors)")

    _, test_err = score_windows(model, test_win)
    flagged = test_err > tau
    evaluate(test_err, test_labels, tau, test_starts, events, flagged)

    # figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax = axes[0, 0]
    for ev_start, ev_end, _ in events:
        ax.axvspan(ev_start, ev_end, color=COL_ANOM, alpha=0.2)
    ax.plot(test_signal[:4000], color=COL_SIGNAL, lw=0.8)
    ax.set_xlim(0, 4000)
    ax.set_title("test stream (red spans = injected anomalies)")

    ax = axes[0, 1]
    epochs_x = np.arange(1, EPOCHS + 1)
    ax.plot(epochs_x, history["train"], color=COL_SIGNAL, label="train")
    ax.plot(epochs_x, history["val"], color=COL_RECON, label="val")
    ax.set_yscale("log")
    ax.set_title("training curves (MSE, log scale)")
    ax.legend()

    ax = axes[1, 0]
    bins = np.linspace(np.log10(test_err).min(), np.log10(test_err).max(), 50)
    ax.hist(np.log10(test_err[~test_labels]), bins=bins, color=COL_NORMAL,
            alpha=0.7, label="normal")
    ax.hist(np.log10(test_err[test_labels]), bins=bins, color=COL_ANOM,
            alpha=0.7, label="anomalous")
    ax.axvline(np.log10(tau), color=COL_THRESH, lw=2, label="tau")
    ax.set_title("log10 reconstruction error by true class")
    ax.legend()

    ax = axes[1, 1]
    for ev_start, ev_end, _ in events:
        ax.axvspan(ev_start, ev_end, color=COL_ANOM, alpha=0.2)
    ax.scatter(test_starts[~test_labels], test_err[~test_labels], s=8,
               color=COL_NORMAL, label="normal")
    ax.scatter(test_starts[test_labels], test_err[test_labels], s=10,
               color=COL_ANOM, label="anomalous")
    ax.axhline(tau, color=COL_THRESH, lw=2, label="tau")
    ax.set_yscale("log")
    ax.set_title("error along the test stream")
    ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
