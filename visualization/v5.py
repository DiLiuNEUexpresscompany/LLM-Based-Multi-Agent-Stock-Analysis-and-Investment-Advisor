# risk_radar.py
import matplotlib.pyplot as plt
import numpy as np

categories = ['Short-Term Risks', 'Long-Term Risks', 'Volatility', 'Sector Risk']
values = [4, 3, 4, 2]  # Hypothetical scores (1-5)
N = len(categories)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, color='#D35400', linewidth=2)
ax.fill(angles, values, color='#D35400', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_title('Risk Assessment Radar Chart', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('risk_radar.png', dpi=300)
plt.show()