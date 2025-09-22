import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# --------------------------
# 1. 定义模型与参数
# --------------------------
# 生成二维参数空间 (w1, w2)
w1 = np.linspace(-3, 5, 200)
w2 = np.linspace(-3, 5, 200)
w1_grid, w2_grid = np.meshgrid(w1, w2)

# 假设最小二乘解（残差项的最小值点）
w_ls = np.array([3.0, 2.0])  # 最小二乘解位置


# 残差项：||t - Φw||²，其等高线为椭圆
# 修正矩阵运算，解决维度不匹配问题
def residual_term(w1, w2, w_ls):
    # 构造椭圆形状的残差函数
    # (w - w_ls)^T * A * (w - w_ls)，A为正定矩阵控制椭圆形状
    A = np.array([[2, 0.5], [0.5, 1]])  # 控制椭圆形状的矩阵

    # 计算每个点与最小二乘解的差异
    w1_diff = w1 - w_ls[0]
    w2_diff = w2 - w_ls[1]

    # 计算二次型，避免高维矩阵乘法错误
    # 展开公式: (w1_diff, w2_diff) · A · (w1_diff; w2_diff)
    return A[0, 0] * w1_diff ** 2 + (A[0, 1] + A[1, 0]) * w1_diff * w2_diff + A[1, 1] * w2_diff ** 2


# 计算残差项的等高线值（现在可以正确处理网格数据）
residual_values = residual_term(w1_grid, w2_grid, w_ls)


# L2正则化项：λ||w||²，其等高线为圆
def l2_regularizer(w1, w2, lambd):
    return lambd * (w1 ** 2 + w2 ** 2)


# 定义不同的正则化强度
lambdas = [0.0, 0.5, 2.0]  # 0表示无正则化，值越大正则化越强
lambda_labels = ["λ=0 (无正则化)", "λ=0.5 (中等正则化)", "λ=2.0 (强正则化)"]

# --------------------------
# 2. 计算不同λ下的最优解
# --------------------------
# 为了求解最优解，我们需要最小化：残差项 + 正则化项
optimal_points = []

for lambd in lambdas:
    # 总目标函数
    total_cost = residual_values + l2_regularizer(w1_grid, w2_grid, lambd)

    # 找到最小值点的索引
    min_idx = np.unravel_index(np.argmin(total_cost), total_cost.shape)
    optimal_w = np.array([w1[min_idx[1]], w2[min_idx[0]]])
    optimal_points.append(optimal_w)

# --------------------------
# 3. 可视化几何意义
# --------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (lambd, ax) in enumerate(zip(lambdas, axes)):
    # 绘制残差项等高线（椭圆）
    contour_res = ax.contourf(w1_grid, w2_grid, residual_values,
                              levels=20, cmap=cm.Blues, alpha=0.6)
    ax.contour(w1_grid, w2_grid, residual_values,
               levels=10, colors='blue', linewidths=0.5)

    # 绘制正则化项等高线（圆）
    # 计算最优解处的正则化值，用于绘制通过最优解的圆
    reg_value_at_opt = l2_regularizer(optimal_points[i][0], optimal_points[i][1], lambd)
    contour_reg = ax.contour(w1_grid, w2_grid, l2_regularizer(w1_grid, w2_grid, lambd),
                             levels=[reg_value_at_opt], colors='red', linewidths=2)

    # 标注关键点
    ax.scatter(w_ls[0], w_ls[1], color='green', s=100, marker='*', label='最小二乘解')
    ax.scatter(optimal_points[i][0], optimal_points[i][1],
               color='purple', s=100, marker='^', label='岭回归最优解')
    ax.scatter(0, 0, color='orange', s=80, marker='o', label='原点 (0,0)')

    # 添加通过最优解的圆（正则化项）
    if lambd > 0:
        radius = np.sqrt(reg_value_at_opt / lambd)  # 圆的半径
        circle = Circle((0, 0), radius, fill=False, edgecolor='red', linestyle='--', linewidth=1.5)
        ax.add_patch(circle)

    # 设置图形属性
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_title(f'岭回归几何解释：{lambda_labels[i]}')
    ax.axis('equal')  # 等比例坐标轴，确保圆不被拉伸
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()
# --------------------------
# 4. 绘制解的轨迹图（不同λ下最优解的变化）
# --------------------------
# 生成更密集的λ值来绘制轨迹
lambda_dense = np.logspace(-1, 1, 50)  # λ从0.1到10
optimal_trace = []

for lambd in lambda_dense:
    total_cost = residual_values + l2_regularizer(w1_grid, w2_grid, lambd)
    min_idx = np.unravel_index(np.argmin(total_cost), total_cost.shape)
    optimal_w = np.array([w1[min_idx[1]], w2[min_idx[0]]])
    optimal_trace.append(optimal_w)

# 转换为数组
optimal_trace = np.array(optimal_trace)

# 绘图
plt.figure(figsize=(8, 8))
# 绘制残差项背景
plt.contourf(w1_grid, w2_grid, residual_values, levels=20, cmap=cm.Blues, alpha=0.3)
plt.contour(w1_grid, w2_grid, residual_values, levels=10, colors='blue', linewidths=0.5)

# 绘制解的轨迹
plt.plot(optimal_trace[:, 0], optimal_trace[:, 1], 'ro-', markersize=4,
         label='最优解轨迹（λ增大方向）')

# 标注关键点
plt.scatter(w_ls[0], w_ls[1], color='green', s=100, marker='*', label='最小二乘解 (λ=0)')
plt.scatter(0, 0, color='orange', s=80, marker='o', label='原点 (λ→∞)')

# 标注几个λ值的位置
label_indices = [0, 15, 30, 49]  # 选择几个点进行标注
for idx in label_indices:
    plt.annotate(f'λ≈{lambda_dense[idx]:.2f}',
                 xy=(optimal_trace[idx, 0], optimal_trace[idx, 1]),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.xlabel('w1')
plt.ylabel('w2')
plt.title('不同λ值下岭回归最优解的轨迹')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.legend()
plt.show()
