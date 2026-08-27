import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "DUNs_SNR.csv"

METHOD_ORDER = [
    "LISTA",
    "LAMP",
    "ISTA",
    "AMP",
    "DNN (layer-wise training)",
]

LEGEND_LABELS = {
    "LISTA": "LISTA",
    "LAMP": "LAMP",
    "ISTA": "ISTA",
    "AMP": "AMP",
    "DNN (end-to-end training)": "DNN (end-to-end training)",
    "DNN (layer-wise training)": "DNN",
}

METHOD_ALIASES = {
    "DNN (End to end training)": "DNN (end-to-end training)",
    "DNN (Layer-wise training)": "DNN (layer-wise training)",
}

STYLES = {
    "LISTA": {
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "*",
    },
    "LAMP": {
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "v",
    },
    "ISTA": {
        "color": "#2e8b57",
        "linestyle": "--",
        "marker": "*",
    },
    "AMP": {
        "color": "#2e8b57",
        "linestyle": "--",
        "marker": "v",
    },
    "DNN (end-to-end training)": {
        "color": "#9b1c14",
        "linestyle": "--",
        "marker": "v",
    },
    "DNN (layer-wise training)": {
        "color": "#9b1c14",
        "linestyle": "--",
        "marker": "*",
    },
}


def load_snr_nmse(csv_file):
    data = defaultdict(list)
    with csv_file.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            method = METHOD_ALIASES.get(row["method"], row["method"])
            snr = float(row["SNR"])
            nmse = float(row["NMSE"])
            data[method].append((snr, nmse))

    for method in data:
        data[method].sort(key=lambda item: item[0])
    return data


def draw_figure(data):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "mathtext.fontset": "stix",
        }
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.1), dpi=300)

    for method in METHOD_ORDER:
        if method not in data:
            continue

        points = data[method]
        x_values = [item[0] for item in points]
        y_values = [item[1] for item in points]
        style = STYLES[method]

        ax.plot(
            x_values,
            y_values,
            label=LEGEND_LABELS[method],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.5,
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
        )

    all_snr = sorted({snr for points in data.values() for snr, _ in points})
    all_nmse = [nmse for points in data.values() for _, nmse in points]
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("NMSE")
    ax.set_xticks(all_snr)
    ax.set_xlim(min(all_snr), max(all_snr))
    ax.set_ylim(min(all_nmse) - 0.5, max(all_nmse) + 0.5)
    ax.set_yticks([-5, -10, -15, -20, -25])

    ax.grid(True, which="major", color="#b0b0b0", linestyle="-", linewidth=0.5, alpha=0.6)
    ax.tick_params(direction="in", length=3.0, width=0.8)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.005, 0.005),
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#b0b0b0",
        columnspacing=1.6,
        handlelength=2.4,
    )

    fig.tight_layout()
    return fig


def main():
    data = load_snr_nmse(CSV_FILE)
    fig = draw_figure(data)
    fig.savefig(BASE_DIR / "DUNs_SNR.png", bbox_inches="tight", dpi=300)
    fig.savefig(BASE_DIR / "DUNs_SNR.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
