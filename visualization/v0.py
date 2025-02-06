import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta

# Set style configurations
plt.style.use('default')  # Use default style instead of seaborn
sns.set_theme(style="whitegrid")  # Set seaborn theme
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def create_stock_analysis_dashboard():
    try:
        # Create a figure with a specific size
        fig = plt.figure(figsize=(20, 15))
        
        # Define grid layout
        gs = plt.GridSpec(3, 3, figure=fig)
        
        # 1. Stock Price Chart
        ax1 = fig.add_subplot(gs[0, :])
        dates = ['Jan 8', 'Jan 11', 'Jan 14', 'Jan 17', 'Jan 20', 'Jan 23', 'Jan 26', 'Jan 29']
        prices = [225, 228, 232, 235, 238, 240, 242, 232.80]
        
        ax1.plot(dates, prices, marker='o', linewidth=2, color='#2E86C1')
        ax1.fill_between(dates, prices, min(prices), alpha=0.2, color='#2E86C1')
        ax1.set_title('AAPL Stock Price Movement', fontsize=14, pad=20)
        ax1.grid(True, alpha=0.3)
        
        # 2. Key Metrics
        ax2 = fig.add_subplot(gs[1, 0])
        metrics = ['Market Cap', 'Volume', 'Close Price', 'Employees']
        values = ['$3.5T', '41.63M', '$232.80', '164K']
        y_pos = np.arange(len(metrics))
        
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F1C40F']
        ax2.barh(y_pos, [1]*len(metrics), color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(metrics)
        ax2.set_xticks([])
        
        for i, v in enumerate(values):
            ax2.text(0.5, i, v, fontweight='bold', va='center', ha='center', color='white')
        
        ax2.set_title('Key Metrics', fontsize=14, pad=20)
        
        # 3. Financial Health Gauge
        ax3 = fig.add_subplot(gs[1, 1])
        
        def create_gauge(value, title, color):
            theta = np.linspace(0, 180, 100)
            r = 1
            x = r * np.cos(np.radians(theta))
            y = r * np.sin(np.radians(theta))
            
            ax3.plot(x, y, color='gray', alpha=0.3)
            
            value_theta = np.linspace(0, 180 * (value/100), 100)
            value_x = r * np.cos(np.radians(value_theta))
            value_y = r * np.sin(np.radians(value_theta))
            ax3.plot(value_x, value_y, color=color, linewidth=3)
            
            ax3.text(0, -0.2, f'{value}%', ha='center', va='center', fontsize=12, fontweight='bold')
            ax3.text(0, 0.5, title, ha='center', va='center', fontsize=10)
        
        create_gauge(85, 'Confidence Rating', '#9B59B6')
        ax3.set_xlim(-1.2, 1.2)
        ax3.set_ylim(-0.5, 1.2)
        ax3.axis('off')
        
        # 4. Growth Indicators
        ax4 = fig.add_subplot(gs[1, 2])
        growth_metrics = ['Revenue Growth', 'Profit Margin']
        growth_values = [3.3, 26.5]
        
        ax4.bar(growth_metrics, growth_values, color=['#27AE60', '#E67E22'])
        ax4.set_title('Growth Indicators', fontsize=14, pad=20)
        
        for i, v in enumerate(growth_values):
            ax4.text(i, v, f'{v}%', ha='center', va='bottom')
        
        # 5. Risk Assessment
        ax5 = fig.add_subplot(gs[2, :], projection='polar')
        risk_factors = ['Market Risk', 'Financial Risk', 'Operational Risk', 'Competition Risk']
        risk_values = [0.3, 0.2, 0.4, 0.35]
        
        angles = np.linspace(0, 2*np.pi, len(risk_factors), endpoint=False)
        risk_values = np.concatenate((risk_values, [risk_values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax5.plot(angles, risk_values, 'o-', linewidth=2, color='#C0392B')
        ax5.fill(angles, risk_values, alpha=0.25, color='#C0392B')
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(risk_factors)
        ax5.set_title('Risk Assessment', fontsize=14, pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='#2E86C1', label='Stock Price'),
            plt.Line2D([0], [0], color='#9B59B6', label='Confidence'),
            plt.Line2D([0], [0], color='#27AE60', label='Growth'),
            plt.Line2D([0], [0], color='#C0392B', label='Risk')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.98))
        
        # Add title
        fig.suptitle('Apple Inc. (AAPL) Stock Analysis Dashboard', fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    try:
        # Generate dashboard
        fig = create_stock_analysis_dashboard()
        
        if fig is not None:
            # Save the figure
            plt.savefig('apple_stock_analysis.png', dpi=300, bbox_inches='tight')
            print("Dashboard successfully created and saved as 'apple_stock_analysis.png'")
            plt.close()
        else:
            print("Failed to create dashboard")
            
    except Exception as e:
        print(f"Error saving the dashboard: {str(e)}")

if __name__ == "__main__":
    main()