import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = osp.join(HERE, '..', '..', '..', 'results')

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ============================================================================
# 整体性能指标
# ============================================================================
def plot_overall_metrics(result_df):
    """绘制整体性能指标柱状图"""
    # 计算整体性能指标
    tp = sum((result_df['generated_text_code'] == 1) & (result_df['answer_code'] == 1))
    pred_positive = sum(result_df['generated_text_code'] == 1)
    actual_positive = sum(result_df['answer_code'] == 1)

    accuracy = sum(result_df['generated_text_code'] == result_df['answer_code']) / len(result_df)
    precision = tp / pred_positive if pred_positive > 0 else 0.0
    recall = tp / actual_positive if actual_positive > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 绘制柱状图
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.title('Overall Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.gca().set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '1_metrics_overall.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def plot_category_performance(result_df):
    """绘制各类别的性能指标对比"""
    # 计算各类别的性能指标
    categories = result_df['category'].unique()
    metrics_data = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    for category in categories:
        cat_df = result_df[result_df['category'] == category]
        
        cat_tp = sum((cat_df['generated_text_code'] == 1) & (cat_df['answer_code'] == 1))
        cat_pred_positive = sum(cat_df['generated_text_code'] == 1)
        cat_actual_positive = sum(cat_df['answer_code'] == 1)
        
        cat_accuracy = sum(cat_df['generated_text_code'] == cat_df['answer_code']) / len(cat_df)
        cat_precision = cat_tp / cat_pred_positive if cat_pred_positive > 0 else 0.0
        cat_recall = cat_tp / cat_actual_positive if cat_actual_positive > 0 else 0.0
        cat_f1 = 2 * (cat_precision * cat_recall) / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0.0
        
        metrics_data['Accuracy'].append(cat_accuracy)
        metrics_data['Precision'].append(cat_precision)
        metrics_data['Recall'].append(cat_recall)
        metrics_data['F1 Score'].append(cat_f1)

    # 绘制分组柱状图
    x = np.arange(len(categories))
    width = 0.2

    plt.figure(figsize=(14, 7))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for i, (metric, values) in enumerate(metrics_data.items()):
        plt.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)

    plt.xlabel('Category', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.title('Performance Metrics by Category', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x + width * 1.5, categories, rotation=45, ha='right')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.gca().set_axisbelow(True)
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(osp.join(RESULTS_DIR, '2_performance_category.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 打印详细数据
    print("\nPerformance by Category:")
    for i, category in enumerate(categories):
        print(f"\n{category}:")
        print(f"  Accuracy:  {metrics_data['Accuracy'][i]:.4f}")
        print(f"  Precision: {metrics_data['Precision'][i]:.4f}")
        print(f"  Recall:    {metrics_data['Recall'][i]:.4f}")
        print(f"  F1 Score:  {metrics_data['F1 Score'][i]:.4f}")


# ============================================================================
# 混淆矩阵
# ============================================================================
def plot_confusion_matrix(result_df):
    """绘制整体混淆矩阵"""
    # 创建混淆矩阵
    confusion_matrix = pd.crosstab(result_df['generated_text_code'], 
                                    result_df['answer_code'])

    plt.figure(figsize=(8, 6))
    im = plt.imshow(confusion_matrix, cmap='YlOrRd', aspect='auto')

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('Count', fontsize=12, fontweight='bold')

    # 设置刻度
    plt.xticks(np.arange(len(confusion_matrix.columns)), confusion_matrix.columns)
    plt.yticks(np.arange(len(confusion_matrix.index)), confusion_matrix.index)

    # 添加数值标签
    for i in range(len(confusion_matrix.index)):
        for j in range(len(confusion_matrix.columns)):
            text = plt.text(j, i, confusion_matrix.iloc[i, j],
                        ha="center", va="center", color="black", 
                        fontsize=16, fontweight='bold')

    plt.xlabel('Actual Label (answer_code)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Label (generated_text_code)', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(osp.join(RESULTS_DIR, '3_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("\nConfusion Matrix:")
    print(confusion_matrix)


def plot_confusion_matrix_by_category(result_df):
    """绘制各类别的混淆矩阵"""
    categories = sorted(result_df['category'].unique())
    
    # 计算子图布局
    n_categories = len(categories)
    n_cols = 3
    n_rows = (n_categories + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_categories > 1 else [axes]
    
    for idx, category in enumerate(categories):
        cat_df = result_df[result_df['category'] == category]
        
        # 创建混淆矩阵
        confusion_matrix = pd.crosstab(cat_df['generated_text_code'], 
                                        cat_df['answer_code'])
        
        # 确保混淆矩阵是2x2的
        for i in [0, 1]:
            if i not in confusion_matrix.index:
                confusion_matrix.loc[i] = [0, 0]
            if i not in confusion_matrix.columns:
                confusion_matrix[i] = 0
        confusion_matrix = confusion_matrix.sort_index().sort_index(axis=1)
        
        # 绘制混淆矩阵
        ax = axes[idx]
        im = ax.imshow(confusion_matrix, cmap='YlOrRd', aspect='auto')
        
        # 设置刻度
        ax.set_xticks(np.arange(len(confusion_matrix.columns)))
        ax.set_yticks(np.arange(len(confusion_matrix.index)))
        ax.set_xticklabels(confusion_matrix.columns)
        ax.set_yticklabels(confusion_matrix.index)
        
        # 添加数值标签
        for i in range(len(confusion_matrix.index)):
            for j in range(len(confusion_matrix.columns)):
                ax.text(j, i, int(confusion_matrix.iloc[i, j]),
                       ha="center", va="center", color="black", 
                       fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Actual', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_title(f'{category}', fontsize=12, fontweight='bold', pad=10)
        
        # 计算性能指标
        tp = confusion_matrix.iloc[1, 1] if 1 in confusion_matrix.index and 1 in confusion_matrix.columns else 0
        fp = confusion_matrix.iloc[1, 0] if 1 in confusion_matrix.index and 0 in confusion_matrix.columns else 0
        fn = confusion_matrix.iloc[0, 1] if 0 in confusion_matrix.index and 1 in confusion_matrix.columns else 0
        tn = confusion_matrix.iloc[0, 0] if 0 in confusion_matrix.index and 0 in confusion_matrix.columns else 0
        
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 打印统计信息
        print(f"\n{category} Confusion Matrix:")
        print(confusion_matrix)
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
    
    # 隐藏多余的子图
    for idx in range(n_categories, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrix by Category', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(osp.join(RESULTS_DIR, '4_confusion_matrix_by_category.png'), dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# 分布分析
# ============================================================================
def plot_distribution(result_df):
    """绘制预测与实际标签的分布对比"""
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 预测分布
    pred_counts = result_df['generated_text_code'].value_counts().sort_index()
    colors1 = ['#3498db', '#e74c3c']
    labels1 = [f'Predicted 0\n({pred_counts.iloc[0]})', 
            f'Predicted 1\n({pred_counts.iloc[1]})']

    wedges1, texts1, autotexts1 = ax1.pie(pred_counts.values, 
                                            labels=labels1, 
                                            autopct='%1.1f%%', 
                                            colors=colors1, 
                                            startangle=90,
                                            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('Prediction Distribution', fontsize=14, fontweight='bold')

    # 实际分布
    actual_counts = result_df['answer_code'].value_counts().sort_index()
    colors2 = ['#9b59b6', '#1abc9c']
    labels2 = [f'Actual 0\n({actual_counts.iloc[0]})', 
            f'Actual 1\n({actual_counts.iloc[1]})']

    wedges2, texts2, autotexts2 = ax2.pie(actual_counts.values, 
                                            labels=labels2, 
                                            autopct='%1.1f%%', 
                                            colors=colors2, 
                                            startangle=90,
                                            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Actual Label Distribution', fontsize=14, fontweight='bold')

    plt.suptitle('Prediction vs Actual Distribution', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(osp.join(RESULTS_DIR, '5_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 打印分布信息
    print("\nDistribution Summary:")
    print(f"\nPrediction Distribution:")
    print(f"  Predicted 0: {pred_counts.iloc[0]} ({pred_counts.iloc[0]/len(result_df)*100:.1f}%)")
    print(f"  Predicted 1: {pred_counts.iloc[1]} ({pred_counts.iloc[1]/len(result_df)*100:.1f}%)")
    print(f"\nActual Distribution:")
    print(f"  Actual 0: {actual_counts.iloc[0]} ({actual_counts.iloc[0]/len(result_df)*100:.1f}%)")
    print(f"  Actual 1: {actual_counts.iloc[1]} ({actual_counts.iloc[1]/len(result_df)*100:.1f}%)")


# ============================================================================
# 熵分析
# ============================================================================
def plot_entropy_analysis(result_df):
    """绘制幻觉与非幻觉答案的熵分布箱线图"""
    # 准备数据
    data_to_plot = [
        result_df[result_df['judgment'] == 1]['avg_entropy'].dropna(),
        result_df[result_df['judgment'] == 0]['avg_entropy'].dropna()
    ]

    plt.figure(figsize=(10, 6))

    bp = plt.boxplot(data_to_plot, 
                    labels=['Non-Hallucinated\n(judgment=1)', 'Hallucinated\n(judgment=0)'],
                    patch_artist=True, 
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    # 设置颜色
    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.ylabel('Average Entropy', fontsize=14, fontweight='bold')
    plt.title('Entropy Distribution: Hallucinated vs Non-Hallucinated Answers', 
            fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加均值标注
    means = [data.mean() for data in data_to_plot]
    for i, mean in enumerate(means):
        plt.text(i + 1, mean, f'Mean: {mean:.4f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(osp.join(RESULTS_DIR, '6_entropy.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    print("\nEntropy Statistics:")
    print(f"Non-Hallucinated (judgment=1):")
    print(f"  Mean:   {data_to_plot[0].mean():.4f}")
    print(f"  Median: {data_to_plot[0].median():.4f}")
    print(f"  Std:    {data_to_plot[0].std():.4f}")
    print(f"\nHallucinated (judgment=0):")
    print(f"  Mean:   {data_to_plot[1].mean():.4f}")
    print(f"  Median: {data_to_plot[1].median():.4f}")
    print(f"  Std:    {data_to_plot[1].std():.4f}")


def plot_entropy_by_category(result_df):
    """绘制各类别的平均熵对比"""
    # 计算各类别的熵
    categories = result_df['category'].unique()

    non_hall_entropy = []
    hall_entropy = []

    for category in categories:
        cat_df = result_df[result_df['category'] == category]
        non_hall_entropy.append(cat_df[cat_df['judgment'] == 1]['avg_entropy'].mean())
        hall_entropy.append(cat_df[cat_df['judgment'] == 0]['avg_entropy'].mean())

    # 绘制柱状图
    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(12, 6))

    bars1 = plt.bar(x - width/2, non_hall_entropy, width, label='Non-Hallucinated', 
                color='#2ecc71', alpha=0.8)
    bars2 = plt.bar(x + width/2, hall_entropy, width, label='Hallucinated', 
                color='#e74c3c', alpha=0.8)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Category', fontsize=14, fontweight='bold')
    plt.ylabel('Average Entropy', fontsize=14, fontweight='bold')
    plt.title('Average Entropy by Category and Judgment', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.gca().set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(osp.join(RESULTS_DIR, '7_entropy_by_category.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 打印详细数据
    print("\nAverage Entropy by Category:")
    for i, category in enumerate(categories):
        print(f"\n{category}:")
        print(f"  Non-Hallucinated: {non_hall_entropy[i]:.4f}")
        print(f"  Hallucinated:     {hall_entropy[i]:.4f}")
        if not np.isnan(non_hall_entropy[i]) and not np.isnan(hall_entropy[i]):
            print(f"  Difference:       {abs(non_hall_entropy[i] - hall_entropy[i]):.4f}")


# ============================================================================
# 主函数
# ============================================================================
if __name__ == "__main__":
    result_df = pd.read_csv(osp.join(RESULTS_DIR, 'results.csv'))
    
    # 1. 整体性能指标
    plot_overall_metrics(result_df)
    plot_category_performance(result_df)
    
    # 2. 混淆矩阵分析
    plot_confusion_matrix(result_df)
    plot_confusion_matrix_by_category(result_df)
    
    # 3. 分布分析
    plot_distribution(result_df)
    
    # 4. 熵分析
    plot_entropy_analysis(result_df)
    plot_entropy_by_category(result_df)