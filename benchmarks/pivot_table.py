import pandas as pd

# Load the experimental results
df = pd.read_csv('jumper_exp_results.csv')

# Clean optimizer names just in case there are case mismatch issues
df['optimizer'] = df['optimizer'].str.strip()

# Target accuracies to check
target_accuracies = [90, 91, 92, 93, 94, 95]
optimizers = ['Jumper', 'SGD', 'AdamW']

# Dictionary to store results
results = []

# Loop through each optimizer and target accuracy
for opt in optimizers:
    # Filter for the specific optimizer
    opt_df = df[df['optimizer'].str.lower() == opt.lower()].sort_values('epoch')
    
    for target in target_accuracies:
        # Find the first row where test accuracy reaches or exceeds the target
        match = opt_df[opt_df['test_acc'] >= target]
        
        if not match.empty:
            first_reach = match.iloc[0]
            results.append({
                'Optimizer': opt,
                'Target Accuracy (%)': target,
                'Reached Accuracy (%)': first_reach['test_acc'],
                'Epoch': int(first_reach['epoch']),
                'Time (seconds)': round(first_reach['time'], 2)
            })
        else:
            # If the optimizer never reached that accuracy
            results.append({
                'Optimizer': opt,
                'Target Accuracy (%)': target,
                'Reached Accuracy (%)': 'N/A',
                'Epoch': 'N/A',
                'Time (seconds)': 'Never Reached'
            })

# Convert results to a DataFrame for clean formatting
summary_df = pd.DataFrame(results)

# Display as a pivot table for easy comparison
pivot_summary = summary_df.pivot(index='Target Accuracy (%)', columns='Optimizer', values='Time (seconds)')
print("--- Time to Reach Target Accuracy (Seconds) ---")
print(pivot_summary)