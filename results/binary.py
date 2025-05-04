import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open("./results/binary.json", "r") as file:
    data = json.load(file)

# Specify the exact metric order
all_metrics = ["Accuracy", "F1-score", "Precision", "Recall", "TPR", "FPR"]

# Prepare data for plotting
model_names = list(data.keys())
metric_scores = {metric: [] for metric in all_metrics}

for metric in all_metrics:
    for model in model_names:
        metric_scores[metric].append(data[model].get(metric, 0))  # 0 if missing

# Plotting
x = np.arange(len(all_metrics))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Offset each model's bars
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, (model, color) in enumerate(zip(model_names, colors)):
    values = [metric_scores[metric][i] for metric in all_metrics]
    bars = ax.bar(x + i * width, values, width, label=model, color=color)
    
    # Add score labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Labels and title
ax.set_ylabel('Scores')
ax.set_xlabel('Metrics')
ax.set_title('Model Performance Comparison (Binary Classification)')
ax.set_xticks(x + width)
ax.set_xticklabels(all_metrics)
ax.legend(title="Models", loc="upper right")
ax.set_ylim(0, 1.3)  # Adjust y-axis for visibility

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
