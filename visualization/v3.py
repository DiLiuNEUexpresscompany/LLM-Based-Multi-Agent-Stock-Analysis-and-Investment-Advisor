# revenue_net_income.py
import matplotlib.pyplot as plt
import numpy as np

labels = ['Revenue', 'Net Income']
growth = [3.3, 3.3]  # Percentage growth
x = np.arange(len(labels))

plt.figure(figsize=(8, 6))
bars = plt.bar(x, growth, color=['#9B59B6', '#16A085'])

plt.title('YoY Growth (2023 vs 2024)', fontsize=14, fontweight='bold')
plt.xticks(x, labels)
plt.ylabel('Growth (%)', fontsize=12)
plt.ylim(0, 4)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height}%', ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('growth.png', dpi=300)
plt.show()