import json
import matplotlib.pyplot as plt
import numpy as np

# Read the JSON file
with open('./results/result.json', 'r') as file:
    data = json.load(file)

# Extract model names and categories
categories = ['Small', 'Medium', 'Large']
models = list(data['Small'].keys())
n_models = len(models)

# Prepare data for plotting
accuracies = {cat: [data[cat][model] for model in models] for cat in categories}

# Set up the plot
fig, ax = plt.subplots(figsize=(20, 12))

# Bar width and positions
bar_width = 0.25
x = np.arange(n_models)

# Plot bars for each category and add text labels
bars_small = plt.bar(x - bar_width, accuracies['Small'], bar_width, label='Small', color='skyblue')
bars_medium = plt.bar(x, accuracies['Medium'], bar_width, label='Medium', color='lightgreen')
bars_large = plt.bar(x + bar_width, accuracies['Large'], bar_width, label='Large', color='salmon')

# Add accuracy scores on top of each bar
for bars, category in zip([bars_small, bars_medium, bars_large], categories):
    for bar, score in zip(bars, accuracies[category]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{score:.4f}', ha='center', va='bottom', fontsize=14, rotation=45)

# Customize the plot
plt.xlabel('Models')
plt.ylabel('Average Accuracy Scores')
plt.title('Model Accuracy Comparison by Dataset Size')
plt.xticks(x, models, rotation=0, fontsize=13)
plt.legend(title='Dataset Size', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 1.2)  # Adjust y-axis to accommodate text labels
plt.tight_layout()

# Show the plot
plt.show()