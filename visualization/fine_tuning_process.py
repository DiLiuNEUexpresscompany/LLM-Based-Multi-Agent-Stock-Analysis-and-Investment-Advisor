import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

class TrainingVisualizer:
    COLORS = {
        'primary': '#3b82f6',          # 明亮的蓝色
        'secondary': '#10b981',        # 清新的绿色
        'background': '#FFFFFF',       # 浅灰背景色
        'grid': '#e2e8f0',            # 网格线颜色
        'text': '#1e293b',            # 主文本颜色
        'title': '#0f172a',           # 标题颜色
        'card_bg': '#f8fafc',         # 卡片背景色
        'accent': '#6366f1'           # 强调色
    }
    
    def __init__(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'figure.facecolor': self.COLORS['background'],
            'text.color': self.COLORS['text'],
            'axes.labelcolor': self.COLORS['text'],
            'xtick.color': self.COLORS['text'],
            'ytick.color': self.COLORS['text']
        })
    
    def _create_card(self, ax, alpha=0.8):
        """创建圆角矩形卡片背景"""
        card = patches.Rectangle(
            (0, 0), 1, 1, 
            transform=ax.transAxes,
            facecolor=self.COLORS['card_bg'],
            edgecolor='none',
            alpha=alpha,
            zorder=1,
            clip_on=False
        )
        ax.add_patch(card)
    
    def create_dashboard(self, loss_data, save_path=None):
        # 创建具有特定比例的图形
        fig = plt.figure(figsize=(14, 9))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 2.5], width_ratios=[1.2, 0.8])
        
        # 设置整体边距
        fig.patch.set_facecolor(self.COLORS['background'])
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        
        # 配置信息面板
        info_ax = fig.add_subplot(gs[0, 0])
        self._create_card(info_ax)
        info_text = """
Training Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Architecture:    LLaMa3.1-8B-Instruction
• Batch Size:      2 (micro)
• Gradient Accum:  4 steps
• Learning Rate:   2e-4
• Warmup Steps:    50
"""
        info_ax.text(0.05, 0.95, info_text,
                    family='JetBrains Mono, monospace',
                    size=11,
                    transform=info_ax.transAxes,
                    verticalalignment='top',
                    color=self.COLORS['text'],
                    zorder=2)
        info_ax.set_title('Model Parameters', 
                         pad=20,
                         color=self.COLORS['title'],
                         weight='bold',
                         size=14)
        info_ax.axis('off')
        
        # 指标面板
        metrics_ax = fig.add_subplot(gs[0, 1])
        self._create_card(metrics_ax)
        metrics = {
            '• Time': '02:12:45',
            '• Memory': '10.8/32GB',
            '• Throughput': '132.5 tok/s',
            '• Checkpoint': f'Step {len(loss_data)}'
        }
        metric_text = "\n".join([f"{k:<15} {v:>10}" for k, v in metrics.items()])
        metrics_ax.text(0.05, 0.95, metric_text,
                       family='JetBrains Mono, monospace',
                       size=11,
                       transform=metrics_ax.transAxes,
                       verticalalignment='top',
                       color=self.COLORS['text'],
                       zorder=2)
        metrics_ax.set_title('Runtime Metrics', 
                           pad=20,
                           color=self.COLORS['title'],
                           weight='bold',
                           size=14)
        metrics_ax.axis('off')
        
        # 损失曲线图
        plot_ax = fig.add_subplot(gs[1, :])
        self._create_card(plot_ax, alpha=0.9)
        
        # 计算移动平均
        steps = np.arange(1, len(loss_data) + 1)
        ma_loss = np.convolve(loss_data, np.ones(10)/10, mode='valid')
        ma_steps = steps[9:]
        
        # 绘制阴影区域
        plot_ax.fill_between(steps, loss_data, 
                           alpha=0.1, 
                           color=self.COLORS['primary'],
                           zorder=2)
        
        # 绘制主要曲线
        plot_ax.plot(steps, loss_data,
                    color=self.COLORS['primary'],
                    lw=1.5,
                    label='Training Loss',
                    zorder=3)
        plot_ax.plot(ma_steps, ma_loss,
                    color=self.COLORS['secondary'],
                    lw=2.5,
                    label='Moving Average (10 steps)',
                    zorder=4)
        
        # 设置网格和样式
        plot_ax.grid(True,
                    color=self.COLORS['grid'],
                    linestyle='--',
                    alpha=0.7,
                    zorder=1)
        plot_ax.set_xlabel('Training Steps', labelpad=10)
        plot_ax.set_ylabel('Loss Value', labelpad=10)
        
        # 优化图例
        legend = plot_ax.legend(loc='upper right',
                              frameon=True,
                              fancybox=True,
                              shadow=True,
                              framealpha=0.9,
                              edgecolor=self.COLORS['grid'])
        legend.get_frame().set_facecolor(self.COLORS['card_bg'])
        
        plot_ax.set_title('Training Progress', 
                         pad=20,
                         color=self.COLORS['title'],
                         weight='bold',
                         size=14)
        
        # 设置y轴范围,留出一定边距
        y_min, y_max = min(loss_data), max(loss_data)
        y_margin = (y_max - y_min) * 0.1
        plot_ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # 保存图表
        if save_path is None:
            save_path = f'training_monitor.png'
        fig.savefig(save_path, 
                   dpi=300,
                   bbox_inches='tight',
                   facecolor=fig.get_facecolor(),
                   edgecolor='none')
        plt.close(fig)
        return save_path

def generate_sample_data(length=150):
    np.random.seed(42)
    base = np.linspace(1.2, 0.5, length)
    noise = np.random.normal(0, 0.05, length)
    trend = np.sin(np.linspace(0, 4*np.pi, length)) * 0.05
    return base + noise + trend

if __name__ == "__main__":
    loss_data = generate_sample_data()
    visualizer = TrainingVisualizer()
    saved_path = visualizer.create_dashboard(loss_data)
    print(f"Visualization saved to: {saved_path}")