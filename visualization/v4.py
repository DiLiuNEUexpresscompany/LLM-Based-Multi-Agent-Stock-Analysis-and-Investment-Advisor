# profit_margin.py
import matplotlib.pyplot as plt

labels = ['Profit (26.5%)', 'Expenses (73.5%)']
sizes = [26.5, 73.5]
colors = ['#27AE60', '#E74C3C']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
plt.title('Apple Profit Margin (Q2 2024)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('profit_margin.png', dpi=300)
plt.show()