import pandas as pd

# 1. Load the experimental results
df = pd.read_csv("jumper_exp_results.csv")

# Clean column spaces if any exist
df.columns = df.columns.str.strip()

# Initialize an empty list to compile summary metrics
summary_data = []

# 2. Extract core metrics per optimizer
for opt, group in df.groupby("optimizer"):
    group = group.sort_values("epoch")

    # Generalization Performance
    max_test_acc = group["test_acc"].max()
    final_epoch_acc = group["test_acc"].iloc[-1]

    # Hardware Metrics
    avg_vram = group["vram_mb"].mean()
    peak_vram = group["vram_mb"].max()

    # Append core metrics to our summary collection
    summary_data.append(
        {
            "Optimizer": opt,
            "Max Test Acc (%)": f"{max_test_acc:.2f}%",
            "Final Epoch Acc (%)": f"{final_epoch_acc:.2f}%",
            "Avg VRAM (MB)": f"{avg_vram:.1f}",
            "Peak VRAM (MB)": f"{peak_vram:.1f}",
        }
    )

# 3. Construct and display the paper-ready summary table
summary_df = pd.DataFrame(summary_data)
print("\n=== Comprehensive Optimizer Performance Summary ===")
print(summary_df.to_string(index=False))

# Optional: Generate LaTeX format for direct insertion into your paper document
print("\n=== LaTeX Code for LaTeX Document Insertion ===")
print(summary_df.to_latex(index=False, column_format="lcccc"))