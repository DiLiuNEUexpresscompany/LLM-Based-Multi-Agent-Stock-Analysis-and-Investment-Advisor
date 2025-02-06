# financial_metrics.py
import matplotlib.pyplot as plt

categories = ['Market Cap', 'Volume', 'Closing Price', 'Employees']
values = [3.5e12, 41.63e6, 232.8, 164000]
colors = ['#3498DB', '#F1C40F', '#2ECC71', '#E74C3C']

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=colors)

# Format y-axis labels
plt.gca().yaxis.set_major_formatter(lambda x, _: f'{x/1e12:.1f}T' if x >= 1e12 else f'{x/1e6:.1f}M' if x >= 1e6 else f'${x:.1f}' if x < 1e3 else f'{int(x/1e3)}k')

plt.title('Apple Key Financial Metrics', fontsize=14, fontweight='bold')
plt.ylabel('Value', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('financial_metrics.png', dpi=300)
plt.show()