# stock_analysis_dashboard.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc
import seaborn as sns

plt.style.use('seaborn-darkgrid')

# --------------------------
# Data Preparation
# --------------------------
# Price data
dates = pd.date_range(start='2025-01-08', end='2025-02-04', freq='D')
closing_prices = [225, 228, 235, 240, 245, 242, 238, 233, 230, 232.8, 232.8]

# Financial metrics
metrics = {
    'Market Cap': 3.5e12,
    'Daily Volume': 41.63e6,
    'Cash Reserves': 73.8e9,
    'Employees': 164e3
}

# --------------------------
# Create Figure
# --------------------------
fig = plt.figure(figsize=(18, 12), facecolor='#F5F6F6')
gs = fig.add_gridspec(3, 3)

# --------------------------
# 1. Price Trend (Top-left)
# --------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(dates, closing_prices, marker='o', color='#2E86C1', linewidth=2, markersize=8)
ax1.axhline(226.65, color='#E74C3C', linestyle='--', label='Support')
ax1.axhline(233.13, color='#28B463', linestyle='--', label='Resistance')
ax1.fill_between(dates, closing_prices, 220, color='#2E86C150')
ax1.set_title('AAPL Price Trend & Key Levels', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend()

# --------------------------
# 2. Financial Metrics (Top-right)
# --------------------------
ax2 = fig.add_subplot(gs[0, 1:])
bars = ax2.bar(metrics.keys(), metrics.values(), 
              color=['#3498DB', '#F1C40F', '#2ECC71', '#E74C3C'])
ax2.set_title('Key Financial Metrics', fontsize=14, fontweight='bold')
ax2.yaxis.set_major_formatter(lambda x, _: f'${x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, 
            f'{height/1e9:.1f}B' if height >=1e9 else f'{height/1e6:.0f}M',
            ha='center', va='bottom')

# --------------------------
# 3. Growth Comparison (Middle-left)
# --------------------------
ax3 = fig.add_subplot(gs[1, 0])
growth_data = {'Revenue': 3.3, 'Net Income': 3.3}
ax3.barh(list(growth_data.keys()), list(growth_data.values()), 
        color=['#9B59B6', '#16A085'])
ax3.set_title('Year-over-Year Growth (%)', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 4)
for i, (k, v) in enumerate(growth_data.items()):
    ax3.text(v + 0.1, i, f'+{v}%', va='center', fontweight='bold')

# --------------------------
# 4. Profit Margin (Middle-center)
# --------------------------
ax4 = fig.add_subplot(gs[1, 1])
profit = [26.5, 73.5]
ax4.pie(profit, labels=['Profit', 'Expenses'], 
       colors=['#27AE60', '#E74C3C'], 
       autopct='%1.1f%%', startangle=90,
       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax4.set_title('Profit Margin Analysis', fontsize=14, fontweight='bold')

# --------------------------
# 5. Risk Radar (Middle-right)
# --------------------------
ax5 = fig.add_subplot(gs[1, 2], polar=True)
categories = ['Market Risks', 'Competition', 'Volatility', 'Macro Factors']
values = [4, 3.5, 4.2, 3.8]
values += values[:1]

N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

ax5.plot(angles, values, color='#D35400', linewidth=2)
ax5.fill(angles, values, color='#D3540030')
ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories)
ax5.set_yticks([1, 2, 3, 4, 5])
ax5.set_title('Risk Assessment Radar', fontsize=14, fontweight='bold', pad=20)

# --------------------------
# 6. Confidence Gauge (Bottom)
# --------------------------
ax6 = fig.add_subplot(gs[2, :])
ax6.set_facecolor('#F5F6F6')
ax6.axis('off')

# Draw gauge
confidence = 85
theta = np.linspace(np.pi/2, -np.pi/2, 100)
r = np.ones(100) * 5
ax6.plot(theta, r, color='gray', lw=2)

# Color segments
for i in range(0, 50):
    ax6.plot(theta[i:i+2], [4.9, 4.9], color='#E74C3C', lw=3)
for i in range(50, 80):
    ax6.plot(theta[i:i+2], [4.9, 4.9], color='#F1C40F', lw=3)
for i in range(80, 100):
    ax6.plot(theta[i:i+2], [4.9, 4.9], color='#2ECC71', lw=3)

# Needle
needle_angle = np.pi/2 - np.deg2rad(confidence*180/100)
ax6.plot([needle_angle, 0], [0, 4.5], color='#2E86C1', lw=3)
ax6.text(0, 3.5, f'{confidence}% Confidence', ha='center', va='center', 
        fontsize=18, fontweight='bold')
ax6.set_title('Investment Confidence Level', fontsize=14, fontweight='bold', y=0.7)

# --------------------------
# Final Layout
# --------------------------
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig('stock_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()