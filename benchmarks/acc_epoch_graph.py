import matplotlib.pyplot as plt
import pandas as pd

# 1. Load the experimental results
df = pd.read_csv("jumper_exp_results.csv")

# Clean column white spaces if any exist
df.columns = df.columns.str.strip()

# 2. Set up the plotting canvas for publication standards
plt.figure(figsize=(9, 5.5))
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.linewidth"] = 1.2

# Define clean, high-contrast scientific colors and line styles
styles = {
    "Jumper": {"color": "#002F6C", "linestyle": "-", "linewidth": 2.5},  # Deep Navy Blue
    "SGD": {"color": "#D95319", "linestyle": "--", "linewidth": 2.0},  # Deep Orange
    "AdamW": {"color": "#7E2F8E", "linestyle": ":", "linewidth": 2.0},  # Slate Purple
}

# 3. Iterate through each optimizer and plot its trajectory
for opt, group in df.groupby("optimizer"):
    group = group.sort_values("epoch")

    # Get style mappings or fall back to defaults if an unexpected optimizer is present
    opt_style = styles.get(
        opt, {"color": "black", "linestyle": "-", "linewidth": 1.5}
    )

    plt.plot(
        group["epoch"],
        group["test_acc"],
        label=opt,
        color=opt_style["color"],
        linestyle=opt_style["linestyle"],
        linewidth=opt_style["linewidth"],
    )

# 4. Refine gridlines and axes formatting
plt.grid(True, linestyle="--", alpha=0.5, linewidth=0.7)
plt.xlabel("Epoch", fontsize=12, labelpad=10)
plt.ylabel("Test Accuracy (%)", fontsize=12, labelpad=10)
plt.title(
    "Generalization Trajectory Comparison (Validation Accuracy vs. Epoch)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)

# Fix axis margins and range
plt.xlim(df["epoch"].min(), df["epoch"].max())
plt.ylim(50, 100)  # Zooms in on the primary performance variance window

# 5. Add a polished legend box
plt.legend(
    loc="lower right",
    frameon=True,
    facecolor="white",
    edgecolor="#E0E0E0",
    fontsize=11,
)

# Tight layout adjustments to protect labels from clipping in paper layout exports
plt.tight_layout()

# Save as vector graphic format (.pdf or .eps) for LaTeX integration
plt.savefig("optimizer_accuracy_curves.pdf", dpi=300)
plt.show()