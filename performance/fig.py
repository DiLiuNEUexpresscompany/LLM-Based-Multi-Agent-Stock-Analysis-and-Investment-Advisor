import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# Data
categories = [
    "Ethical and Professional Standards",
    "Quantitative Methods",
    "Economics",
    "Financial Reporting",
    "Corporate Finance",
    "Equity Investments",
    "Fixed Income",
    "Derivatives",
    "Alternative Investments",
    "Portfolio Management",
    "Weighted Average"
]

data = {
    "GPT-4o-mini": [77.77, 71.43, 66.66, 79.2, 80, 50, 78.57, 50, 100, 83.3, 73.49],
    "LLaMa3.1-8B-Instruction-Finance": [60.12, 68.49, 60.08, 70.14, 55.07, 47.26, 35.79, 36.72, 83.41, 58.27, 57.33],
    "Meta-Llama Instruct 8B": [55.55, 64.28, 58, 66.6, 50, 41.6, 28.57, 33.3, 75, 50, 52.77],
    "Meta-Llama Instruct 70B": [66.6, 85.71, 58.33, 70.83, 80, 66.6, 50, 33.3, 100, 100, 69.86]
}

# Setup the plot
fig, ax = plt.subplots(figsize=(15, 8))

# Darker Color Mapping
colors = {
    "GPT-4o-mini": '#1F276C',
    "LLaMa3.1-8B-Instruction-Finance": '#3A5998',
    "Meta-Llama Instruct 8B": '#4697D1',
    "Meta-Llama Instruct 70B": '#0095D4'
}

# Bar settings
x = np.arange(len(categories))
bar_width = 0.2

def add_shadow(bar_container):
    for rect in bar_container:
        plt.gca().add_patch(plt.Rectangle(
            (rect.get_x(), rect.get_y()), rect.get_width(), rect.get_height(),
            fill=False, edgecolor='gray', alpha=0.3, lw=5, zorder=1))

# Plot bars
for idx, (model, values) in enumerate(data.items()):
    bars = ax.bar(
        x + idx * bar_width,
        values,
        bar_width,
        label=model,
        color=colors[model],
        edgecolor='black',
        alpha=0.85,
        zorder=3
    )
    
    add_shadow(bars)
    
    # Add value labels above bars
    for i, v in enumerate(values):
        ax.text(
            x[i] + idx * bar_width,
            v + 1,
            f'{v:.1f}',
            ha='center',
            fontsize=8,
            fontweight='bold'
        )

# Customize chart
ax.set_title('Performance of Different Models on CFA Level 1 Topics',
             fontsize=14, pad=20, fontweight='bold')
ax.set_ylabel('Accuracy/Performance (%)', fontsize=12)
ax.set_xticks(x + bar_width * (len(data) / 2 - 0.5))
# ax.set_xticklabels(categories, fontsize=7, fontweight='bold')
ax.set_xticklabels(categories, rotation=20, ha='right', fontweight='bold')
ax.set_ylim(0, 110)

# Add legend
ax.legend(title='Models', loc='upper left', fontsize=9, title_fontsize=10)

# Add grid
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# Adjust layout
plt.tight_layout()

# Save plot as PNG
plt.savefig("performance_chart.png", format="png", dpi=300)

# Show plot
plt.show()
