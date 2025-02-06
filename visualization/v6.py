# confidence_gauge.py
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))

# Draw the gauge
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)
ax.set_ylim(0, 100)

# Create the arc
theta = np.linspace(0, np.pi, 100)
r = np.ones(100) * 100
ax.plot(theta, r, color='black', linewidth=2)
ax.fill_between(theta, 0, r, color='#F1C40F', alpha=0.3)

# Add needle
confidence = 85
needle_angle = np.pi - confidence / 100 * np.pi
ax.plot([needle_angle, needle_angle], [0, 90], color='#E74C3C', linewidth=4)

# Add text
ax.text(0.5, 0.6, f'{confidence}%', transform=ax.transAxes, ha='center', va='center', fontsize=24, fontweight='bold')
ax.set_title('Confidence Level', fontsize=14, fontweight='bold', pad=20)
ax.axis('off')
plt.tight_layout()
plt.savefig('confidence_gauge.png', dpi=300)
plt.show()