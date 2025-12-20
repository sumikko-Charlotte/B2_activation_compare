"""
MLP激活函数对比实验 - 最终提交版本 (v3)

本实验对比了四种激活函数（ReLU, Sigmoid, Tanh, Identity）在非线性分类任务上的表现。
通过Loss曲线和决策边界可视化，深入理解不同激活函数的特性。

作者：人工智能专业学生
数据集：make_moons (非线性二分类)
"""

import os
import csv
import warnings
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示（如果系统有中文字体）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    pass  # 如果没有中文字体，使用默认字体（可能显示为方块，但不影响功能）

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.exceptions import ConvergenceWarning


def train_one(activation, X_train, y_train, X_val, y_val, epochs=200, seed=42, lr=0.01):
    """
    训练一个MLP模型，并记录训练过程中的Loss变化。
    
    为什么使用 warm_start=True？
    - sklearn的MLPClassifier默认在一次fit()调用中完成所有epochs的训练
    - 为了手动记录每个epoch的Loss，我们需要设置 max_iter=1 和 warm_start=True
    - 这样每次调用fit()只训练1个epoch，然后通过warm_start继续训练，实现手动控制训练过程
    
    参数:
        activation: 激活函数类型 ('relu', 'logistic', 'tanh', 'identity')
        X_train, y_train: 训练集
        X_val, y_val: 验证集
        epochs: 训练轮数（为什么是200？足够观察收敛趋势，同时不会过长）
        seed: 随机种子（保证可复现性）
        lr: 学习率（为什么是0.01？Adam优化器的常用初始学习率，适合大多数激活函数）
    
    返回:
        acc: 验证集准确率
        train_losses: 训练Loss列表（每个epoch）
        val_losses: 验证Loss列表（每个epoch）
        clf: 训练好的分类器（用于后续决策边界可视化）
    """
    # 为什么隐藏层是 (32, 32)？
    # - 两层32个神经元提供了足够的非线性映射能力来学习make_moons的复杂边界
    # - 太少的神经元（如8, 8）可能无法充分拟合非线性模式
    # - 太多的神经元（如128, 128）容易过拟合，且计算开销大
    # - 两层结构允许网络学习更复杂的特征组合
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 32),  # 两层隐藏层，每层32个神经元
        activation=activation,         # 激活函数类型
        solver="adam",                 # Adam优化器：自适应学习率，收敛快且稳定
        learning_rate_init=lr,        # 初始学习率
        max_iter=1,                   # 每次fit()只训练1个epoch（配合warm_start使用）
        warm_start=True,              # 允许在已有模型基础上继续训练（关键！）
        random_state=seed             # 随机种子，保证权重初始化可复现
    )

    train_losses = []
    val_losses = []

    # 手动控制训练过程，记录每个epoch的Loss
    for epoch in range(epochs):
        clf.fit(X_train, y_train)  # 训练1个epoch（因为max_iter=1）

        # 记录训练Loss（sklearn自动计算的交叉熵损失）
        train_losses.append(float(clf.loss_))

        # 记录验证Loss（使用log_loss计算，与训练Loss保持一致）
        proba = clf.predict_proba(X_val)
        val_losses.append(float(log_loss(y_val, proba)))

    # 计算最终验证集准确率
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    return acc, train_losses, val_losses, clf


def plot_decision_boundary(clf, X, y, ax, title):
    """
    绘制决策边界可视化图。
    
    为什么需要决策边界可视化？
    - make_moons是一个非线性可分数据集（两个"月亮"形状的类别）
    - 通过可视化决策边界，可以直观看到不同激活函数的学习能力
    - 线性激活函数（Identity）无法学习非线性边界，而ReLU/Sigmoid/Tanh可以
    
    参数:
        clf: 训练好的分类器
        X: 完整数据集（用于绘制散点）
        y: 标签
        ax: matplotlib子图对象
        title: 子图标题
    """
    # 创建网格用于绘制决策边界
    # 为什么使用h=0.02？
    # - 网格分辨率：h越小，边界越平滑，但计算量越大
    # - 0.02是一个平衡点，既能显示细节又不会太慢
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 对网格上的每个点进行预测，得到概率
    # 使用predict_proba获取类别1的概率（用于颜色映射）
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # 绘制等高线填充图（背景颜色表示预测概率）
    # 为什么使用contourf？
    # - contourf用颜色深浅表示预测概率，直观展示模型的"置信度"
    # - 颜色越深（接近1）表示模型越确信该区域属于类别1
    # - 颜色越浅（接近0）表示模型越确信该区域属于类别0
    # - 边界线（0.5概率处）就是决策边界
    ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap=plt.cm.RdYlBu)
    
    # 绘制原始数据点
    # 为什么用散点图？
    # - 可以直观看到数据分布和模型分类结果的关系
    # - 不同颜色表示不同类别，便于理解模型的分类效果
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', 
                        linewidths=0.5, cmap=plt.cm.RdYlBu, s=20)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('特征1', fontsize=10)
    ax.set_ylabel('特征2', fontsize=10)
    ax.grid(True, alpha=0.3)


def main():
    """
    主函数：执行完整的激活函数对比实验。
    
    实验流程：
    1. 数据准备（make_moons数据集）
    2. 对每种激活函数进行多次训练（不同随机种子）
    3. 记录Loss曲线和准确率
    4. 可视化Loss曲线
    5. 可视化决策边界
    6. 保存结果到文件
    """
    # ========== 1. 实验设置 ==========
    # 为什么设置随机种子？
    # - 保证实验结果可复现：相同的随机种子会产生相同的数据划分和权重初始化
    # - 这对于科学实验至关重要，确保结果的可信度
    np.random.seed(42)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # 忽略收敛警告（因为我们手动控制训练）

    # 创建输出目录
    os.makedirs("results", exist_ok=True)

    # ========== 2. 数据集准备 ==========
    # 为什么使用make_moons？
    # - 这是一个经典的非线性二分类数据集，形状像两个"月亮"
    # - 线性分类器无法完美分类，需要非线性激活函数才能学习复杂边界
    # - 非常适合展示不同激活函数的差异
    # 为什么n_samples=1000, noise=0.2？
    # - 1000个样本：足够训练，不会太慢
    # - noise=0.2：适度的噪声，增加分类难度，更能体现激活函数的差异
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

    # ========== 3. 超参数设置 ==========
    epochs = 200  # 训练轮数：足够观察收敛趋势
    lr = 0.01     # 学习率：Adam优化器的常用值

    # 为什么使用多个随机种子？
    # - 神经网络训练具有随机性（权重初始化、数据划分）
    # - 使用多个种子（42, 43, 44）训练，计算均值和标准差
    # - 这样可以评估结果的稳定性和可靠性，避免偶然性
    seeds = [42, 43, 44]  # 3次独立运行

    # 四种激活函数对比
    # 为什么选择这四种？
    # - ReLU: 现代深度学习最常用的激活函数，解决梯度消失问题
    # - Sigmoid: 经典激活函数，输出范围(0,1)，但容易梯度消失
    # - Tanh: 类似Sigmoid但输出范围(-1,1)，零中心化
    # - Identity: 线性激活函数，作为对比基线，展示非线性激活的重要性
    activations = [
        ("ReLU", "relu"),           # ReLU: f(x) = max(0, x)，解决梯度消失
        ("Sigmoid", "logistic"),    # Sigmoid: f(x) = 1/(1+e^(-x))，输出(0,1)
        ("Tanh", "tanh"),           # Tanh: f(x) = tanh(x)，输出(-1,1)，零中心化
        ("Identity", "identity"),   # Identity: f(x) = x，线性函数（无法学习非线性）
    ]

    # 存储Loss曲线（仅第一个seed，用于绘图）
    curves_for_plot = {}
    
    # 存储训练好的模型（仅第一个seed，用于决策边界可视化）
    trained_models = {}
    
    # 存储统计结果（用于CSV和summary）
    records = []

    # ========== 4. 训练所有激活函数 ==========
    for name, act in activations:
        print(f"\n{'='*60}")
        print(f"训练激活函数: {name}")
        print(f"{'='*60}")
        
        acc_list = []
        final_train_list = []
        final_val_list = []

        for i, seed in enumerate(seeds):
            # 为什么使用stratify？
            # - 保证训练集和验证集中各类别的比例一致
            # - 避免因数据划分不均导致的评估偏差
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )

            print(f"  [seed={seed}] 训练中...")
            acc, train_loss, val_loss, clf = train_one(
                act, X_train, y_train, X_val, y_val, epochs=epochs, seed=seed, lr=lr
            )

            acc_list.append(acc)
            final_train_list.append(train_loss[-1])
            final_val_list.append(val_loss[-1])

            # 仅保存第一个seed的模型和曲线（用于可视化）
            if i == 0:
                curves_for_plot[name] = {
                    "train": train_loss,
                    "val": val_loss
                }
                trained_models[name] = clf  # 保存模型用于决策边界可视化

        # 计算统计量（均值±标准差）
        acc_mean, acc_std = float(np.mean(acc_list)), float(np.std(acc_list))
        tr_mean, tr_std = float(np.mean(final_train_list)), float(np.std(final_train_list))
        va_mean, va_std = float(np.mean(final_val_list)), float(np.std(final_val_list))

        print(f"  结果: 准确率 = {acc_mean:.4f}±{acc_std:.4f}")

        records.append({
            "activation_name": name,
            "activation_key": act,
            "val_acc_mean": acc_mean,
            "val_acc_std": acc_std,
            "final_train_loss_mean": tr_mean,
            "final_train_loss_std": tr_std,
            "final_val_loss_mean": va_mean,
            "final_val_loss_std": va_std,
            "epochs": epochs,
            "learning_rate": lr,
            "seeds": ",".join(map(str, seeds)),
        })

    # ========== 5. 绘制Loss曲线图 ==========
    # 为什么将训练Loss和验证Loss放在一张图上？
    # - 可以同时观察训练和验证Loss，判断是否过拟合
    # - 如果训练Loss持续下降但验证Loss不降，说明过拟合
    # - 如果两者都下降，说明模型在学习
    print(f"\n{'='*60}")
    print("生成Loss曲线图...")
    print(f"{'='*60}")
    
    plt.figure(figsize=(12, 5))
    
    # 左子图：训练Loss
    plt.subplot(1, 2, 1)
    for name, curves in curves_for_plot.items():
        plt.plot(curves["train"], label=f"{name}", linewidth=2)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Training Loss", fontsize=11)
    plt.title("训练Loss曲线对比 (seed=42)", fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 右子图：验证Loss
    plt.subplot(1, 2, 2)
    for name, curves in curves_for_plot.items():
        plt.plot(curves["val"], label=f"{name}", linewidth=2)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Loss", fontsize=11)
    plt.title("验证Loss曲线对比 (seed=42)", fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/loss_curves.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] 已保存: results/loss_curves.png")

    # ========== 6. 绘制决策边界图 ==========
    # 为什么需要决策边界可视化？
    # - 这是本次升级的核心需求，能直观展示不同激活函数的学习能力
    # - Identity（线性）无法学习非线性边界，而ReLU/Sigmoid/Tanh可以
    # - 通过2x2子图对比，一目了然地看出差异
    print(f"\n{'='*60}")
    print("生成决策边界可视化图...")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # 为每种激活函数绘制决策边界
    for idx, (name, act) in enumerate(activations):
        clf = trained_models[name]
        
        # 计算准确率用于标题
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        # 绘制决策边界
        title = f"{name}\n(验证准确率: {acc:.2%})"
        plot_decision_boundary(clf, X, y, axes[idx], title)
    
    plt.suptitle("激活函数决策边界对比 (make_moons数据集)", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig("results/decision_boundaries.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] 已保存: results/decision_boundaries.png")
    
    # 为什么Identity表现差？
    # - Identity是线性激活函数：f(x) = x
    # - 多层线性变换等价于单层线性变换，无法学习非线性模式
    # - make_moons是非线性数据集，需要非线性激活函数才能分类
    # - 从决策边界图可以清楚看到：Identity只能画出一条直线，无法拟合"月亮"形状

    # ========== 7. 保存结果文件 ==========
    print(f"\n{'='*60}")
    print("保存结果文件...")
    print(f"{'='*60}")
    
    # 保存summary.txt（人类可读格式）
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("MLP激活函数对比实验 - 结果汇总")
    summary_lines.append("=" * 60)
    summary_lines.append(f"数据集: make_moons (n_samples=1000, noise=0.2)")
    summary_lines.append(f"模型结构: MLP(2 -> 32 -> 32 -> 2)")
    summary_lines.append(f"训练轮数: {epochs} epochs")
    summary_lines.append(f"学习率: {lr}")
    summary_lines.append(f"随机种子: {seeds}")
    summary_lines.append("")
    summary_lines.append("结果 (均值±标准差，基于3次独立运行):")
    summary_lines.append("-" * 60)
    
    # 按准确率排序（从高到低）
    sorted_records = sorted(records, key=lambda x: x['val_acc_mean'], reverse=True)
    for r in sorted_records:
        summary_lines.append(
            f"{r['activation_name']:10s} ({r['activation_key']:10s}): "
            f"准确率={r['val_acc_mean']:.4f}±{r['val_acc_std']:.4f}, "
            f"训练Loss={r['final_train_loss_mean']:.4f}±{r['final_train_loss_std']:.4f}, "
            f"验证Loss={r['final_val_loss_mean']:.4f}±{r['final_val_loss_std']:.4f}"
        )
    
    summary_lines.append("")
    summary_lines.append("=" * 60)
    summary_lines.append("生成的文件:")
    summary_lines.append("  - results/loss_curves.png (Loss曲线对比)")
    summary_lines.append("  - results/decision_boundaries.png (决策边界可视化)")
    summary_lines.append("  - results/summary.txt (本文件)")
    summary_lines.append("  - results/summary.csv (CSV格式结果)")
    summary_lines.append("=" * 60)

    with open("results/summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("  [OK] 已保存: results/summary.txt")

    # 保存summary.csv（便于后续分析）
    csv_path = "results/summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print("  [OK] 已保存: results/summary.csv")

    # ========== 8. 打印最终结果 ==========
    print(f"\n{'='*60}")
    print("实验完成！结果汇总:")
    print(f"{'='*60}")
    for r in sorted_records:
        print(f"{r['activation_name']:10s}: 准确率 = {r['val_acc_mean']:.4f}±{r['val_acc_std']:.4f}")
    print(f"\n所有结果已保存到 results/ 目录")


if __name__ == "__main__":
    main()
