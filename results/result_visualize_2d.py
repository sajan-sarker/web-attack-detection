import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# File paths
file_paths = ["./results/v1_results.json", "./results/v2_results.json", "./results/v3_results.json"]

# Load datasets
datasets = []
for file_path in file_paths:
    with open(file_path, 'r') as f:
        datasets.append(json.load(f))

# Define versions and base model names
version_labels = ['V1', 'V2', 'V3']
base_model_names = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'Ada Boost', 'MLP', 'Ensemble (DT + RF)']

# Full model names without gaps
model_names_with_gaps = []
for i in range(3):
    model_names_with_gaps.extend([f"{model} {version_labels[i]}" for model in base_model_names])

# Updated metrics
metrics = ['Avg Acc', 'Binary Acc', 'Multi-class Acc', 'F1', 'Prec', 'Rec', 'TPR', 'FPR']
z_values = np.full((len(metrics), len(model_names_with_gaps)), np.nan)  # Fill with NaN

# Fill z_values, skipping gaps
col = 0
for ver_idx, ver in enumerate(version_labels):
    dataset = datasets[ver_idx]
    for model in base_model_names:
        model_key = f"{model} {ver}" if ver != 'V1' else model
        if model_key in dataset:
            avg_results = dataset[model_key].get('Average Classification Results', {})
            binary_results = dataset[model_key].get('Binary Classification Results', {})
            multi_results = dataset[model_key].get('Multi-class Classification Results', {})
            any_results = next(iter(dataset[model_key].values()))
            z_values[0, col] = binary_results.get('Test Acc', 0)
            z_values[1, col] = multi_results.get('Test Acc', 0)
            z_values[2, col] = avg_results.get('Test Acc', 0)
            z_values[3, col] = any_results.get('F1-score', 0)
            z_values[4, col] = any_results.get('Precision', 0)
            z_values[5, col] = any_results.get('Recall', 0)
            z_values[6, col] = any_results.get('TPR', 0)
            z_values[7, col] = any_results.get('FPR', 0)
        col += 1

# Define a custom sky blue gradient (dark to light for low to high values)
colors = ['#4682B4', '#87CEEB']  # Steel blue (dark) to light sky blue
sky_blue_cmap = LinearSegmentedColormap.from_list("skyblue_gradient", colors)

# Plotting
fig, ax = plt.subplots(figsize=(22, 7))
im = ax.imshow(z_values, cmap=sky_blue_cmap, aspect='auto')

# Axis ticks and labels
ax.set_xticks(np.arange(len(model_names_with_gaps)))
ax.set_yticks(np.arange(len(metrics)))
ax.set_xticklabels(model_names_with_gaps, rotation=45, ha="right")
ax.set_yticklabels(metrics)

# Annotate only non-gap cells
for i in range(len(metrics)):
    for j in range(len(model_names_with_gaps)):
        value = z_values[i, j]
        if not np.isnan(value):
            ax.text(j, i, f"{value:.4f}", ha="center", va="center", color="white" if value < 0.5 else "black")

# Labels and title
ax.set_xlabel("Models + Versions", fontsize=12)
ax.set_ylabel("Metrics", fontsize=12)
ax.set_title("Metric Scores per Model-Version", fontsize=14)
fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()