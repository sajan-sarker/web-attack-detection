import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('./results/proposed_result.json') as f:
    data = json.load(f)

# Metrics to plot
metrics = ["Accuracy", "F1-score", "Precision", "Recall", "False Positive Rate (FPR)"]
categories = ["Binary Scores", "Multi-class Scores", "Average Scores"]
colors = ['skyblue', 'salmon', 'lightgreen']

# Extract scores
scores = {category: [data[category][metric] for metric in metrics] for category in categories}

# Plot setup
x = np.arange(len(metrics))  # label locations
width = 0.25  # width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Plotting each category
for i, category in enumerate(categories):
    bar = ax.bar(x + i*width - width, scores[category], width, label=category, color=colors[i])
    # Add score on top of each bar
    for rect in bar:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, rotation=45)

# Labels and formatting
ax.set_ylabel('Score')
ax.set_title('Comparison of Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=13)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right')

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
