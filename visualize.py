"""
File: visualize.py
Description: Generates IEEE-style comparison plots for Accuracy, Power, and ROC.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# 配置
CSV_FILE = 'results_metrics.csv'
JSON_FILE = 'results_roc.json'

def set_ieee_style():
    """设置学术绘图风格"""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })

def plot_charts():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please run train.py first.")
        return

    # 读取并去重（防止多次运行有重复行）
    df = pd.read_csv(CSV_FILE)
    df = df.drop_duplicates(subset=['model'], keep='last')
    
    print(f"Loaded results for {len(df)} models.")

    # ==========================================
    # 图表 1: 准确率对比 (Accuracy Comparison)
    # ==========================================
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='model', y='accuracy', palette='viridis')
    plt.title('Figure 1: Model Accuracy Comparison', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    
    # 在柱子上显示数值
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig('fig1_accuracy.png', dpi=300)
    print("-> Generated fig1_accuracy.png")

    # ==========================================
    # 图表 2: 能耗对比 (Total Energy Consumption)
    # ==========================================
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='model', y='energy_wh', palette='magma')
    plt.title('Figure 2: Total Energy Consumption (Green AI)', fontweight='bold')
    plt.ylabel('Energy Consumed (Watt-hours)')
    plt.xticks(rotation=45)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}Wh', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom')

    plt.tight_layout()
    plt.savefig('fig2_energy.png', dpi=300)
    print("-> Generated fig2_energy.png")

    # ==========================================
    # 图表 3: 效率气泡图 (Accuracy vs Power Trade-off)
    # X轴: 功率, Y轴: 准确率, 气泡大小: 训练时间
    # ==========================================
    plt.figure(figsize=(10, 8))
    
    # 归一化时间大小以便绘图
    sizes = df['time_sec'] / df['time_sec'].max() * 1000 + 100
    
    sns.scatterplot(data=df, x='avg_power_watts', y='accuracy', 
                    size=sizes, sizes=(200, 1000), legend=False,
                    hue='model', palette='deep', alpha=0.7, edgecolor='black')
    
    # 标注模型名称
    for i in range(df.shape[0]):
        plt.text(df.avg_power_watts.iloc[i]+0.5, df.accuracy.iloc[i]+0.005, 
                 df.model.iloc[i], fontsize=9, fontweight='bold')

    plt.title('Figure 3: Efficiency Trade-off (Accuracy vs Power)', fontweight='bold')
    plt.xlabel('Average Power Consumption (Watts)')
    plt.ylabel('Validation Accuracy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('fig3_efficiency_bubble.png', dpi=300)
    print("-> Generated fig3_efficiency_bubble.png")

    # ==========================================
    # 图表 4: ROC 曲线对比
    # ==========================================
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            roc_data = json.load(f)
        
        # 去重
        unique_roc = {item['model']: item for item in roc_data}
        
        plt.figure(figsize=(10, 8))
        for name, data in unique_roc.items():
            if data['auc'] > 0: # 只画有效的
                plt.plot(data['fpr'], data['tpr'], lw=2, 
                         label=f"{name} (AUC={data['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Figure 4: Receiver Operating Characteristic (ROC)', fontweight='bold')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('fig4_roc_curves.png', dpi=300)
        print("-> Generated fig4_roc_curves.png")

if __name__ == "__main__":
    set_ieee_style()
    print("Generating Visualizations...")
    plot_charts()
    print("\nAll Done! Please check the 4 PNG files generated.")