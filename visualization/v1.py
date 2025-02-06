# closing_price_trend.py
import matplotlib.pyplot as plt
import pandas as pd

dates = [
    '2025-01-08', '2025-01-11', '2025-01-14', '2025-01-17',
    '2025-01-20', '2025-01-23', '2025-01-26', '2025-01-29',
    '2025-02-01', '2025-02-04'
]
closing_prices = [225, 228, 235, 240, 245, 242, 238, 233, 230, 232.8]

plt.figure(figsize=(12, 6))
plt.plot(pd.to_datetime(dates), closing_prices, marker='o', color='#2E86C1', linewidth=2, markersize=8)
plt.axhline(y=226.65, color='#E74C3C', linestyle='--', label='Support ($226.65)')
plt.axhline(y=233.13, color='#28B463', linestyle='--', label='Resistance ($233.13)')
plt.title('AAPL Closing Price Trend (Jan-Feb 2025)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('closing_price_trend.png', dpi=300)
plt.show()