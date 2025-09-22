import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# --------------------------
# 1. 生成模拟数据
# --------------------------
np.random.seed(42)  # 固定随机种子，保证结果可复现

# 训练集与测试集输入
x_train = np.linspace(0, 5, 20).reshape(-1, 1)  # 20个训练样本，(20,1)
x_test = np.linspace(0, 5, 100).reshape(-1, 1)   # 100个测试样本，(100,1)

# 真实参数与噪声设置
w_true = np.array([1.0, 0.5])  # 真实参数 (w1, w2)
beta = 5.0                     # 噪声精度（噪声方差=1/beta=0.2）
sigma_noise = 1 / np.sqrt(beta)# 噪声标准差

# 构造设计矩阵Phi（特征：x, x²）
def build_design_matrix(x):
    return np.hstack([x, x**2])  # 每行：[x_i, x_i²]

Phi_train = build_design_matrix(x_train)  # (20, 2)
Phi_test = build_design_matrix(x_test)    # (100, 2)

# 生成带噪声的观测目标
eps_train = np.random.normal(0, sigma_noise, size=x_train.shape[0])  # 训练集噪声
eps_test = np.random.normal(0, sigma_noise, size=x_test.shape[0])    # 测试集噪声
t_train = Phi_train @ w_true + eps_train  # 训练集观测值 (20,)
t_test = Phi_test @ w_true + eps_test     # 测试集观测值 (100,)

# --------------------------
# 2. 计算最小二乘解与MAP解
# --------------------------
# 最小二乘解 (Phi^T Phi)^-1 Phi^T t
w_LS = np.linalg.inv(Phi_train.T @ Phi_train) @ Phi_train.T @ t_train

# MAP解：设置alpha，计算lambda=alpha/beta
alpha_medium = 2.0  # 中等先验精度
lambda_medium = alpha_medium / beta  # lambda=0.4
w_MAP_medium = np.linalg.inv(Phi_train.T @ Phi_train + lambda_medium * np.eye(2)) @ Phi_train.T @ t_train

print(f"真实参数 w_true: {w_true}")
print(f"最小二乘解 w_LS: {w_LS.round(4)}")
print(f"MAP解（lambda={lambda_medium:.2f}）w_MAP: {w_MAP_medium.round(4)}")

# --------------------------
# 3. 绘制似然函数等高线
# --------------------------
# 生成参数网格（w1, w2的取值范围）
w1_range = np.linspace(0.5, 1.3, 100)  # w1范围：围绕LS解
w2_range = np.linspace(0.3, 0.7, 100)  # w2范围：围绕LS解
w1_grid, w2_grid = np.meshgrid(w1_range, w2_range)

# 计算每个网格点的负对数似然
neg_log_likelihood = np.zeros_like(w1_grid)
for i in range(len(w2_range)):
    for j in range(len(w1_range)):
        w = np.array([w1_grid[i,j], w2_grid[i,j]])  # 当前参数
        # 负对数似然：beta/2 * ||t - Phi w||²
        neg_log_likelihood[i,j] = 0.5 * beta * np.sum((t_train - Phi_train @ w)**2)

# 绘图
plt.figure(figsize=(8, 6))
contour = plt.contourf(w1_grid, w2_grid, neg_log_likelihood, levels=30, cmap=cm.Blues)
plt.colorbar(contour, label='负对数似然值（越小越好）')
plt.contour(w1_grid, w2_grid, neg_log_likelihood, levels=10, colors='black', linewidths=0.5)

# 标注关键位置
plt.scatter(w_LS[0], w_LS[1], color='red', s=100, label='最小二乘解 (w_LS)', marker='*')
plt.scatter(w_true[0], w_true[1], color='orange', s=80, label='真实参数 (w_true)', marker='o')

plt.xlabel('w1（x的系数）', fontsize=12)
plt.ylabel('w2（x²的系数）', fontsize=12)
plt.title('似然函数的负对数等高线（观测数据约束）', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()


# --------------------------
# 4. 绘制先验分布等高线
# --------------------------
# 扩大参数范围，展示先验的全局性
w1_prior_range = np.linspace(-1, 2, 100)
w2_prior_range = np.linspace(-0.5, 1.5, 100)
w1_p_grid, w2_p_grid = np.meshgrid(w1_prior_range, w2_prior_range)

# 计算每个网格点的负对数先验
neg_log_prior = np.zeros_like(w1_p_grid)
for i in range(len(w2_prior_range)):
    for j in range(len(w1_prior_range)):
        w = np.array([w1_p_grid[i,j], w2_p_grid[i,j]])
        # 负对数先验：alpha/2 * ||w||²
        neg_log_prior[i,j] = 0.5 * alpha_medium * np.sum(w**2)

# 绘图
plt.figure(figsize=(8, 6))
contour_p = plt.contourf(w1_p_grid, w2_p_grid, neg_log_prior, levels=30, cmap=cm.Greens)
plt.colorbar(contour_p, label='负对数先验值（越小越好）')
plt.contour(w1_p_grid, w2_p_grid, neg_log_prior, levels=10, colors='black', linewidths=0.5)

# 标注原点（先验均值）
plt.scatter(0, 0, color='purple', s=100, label='先验均值 (0,0)', marker='s')

plt.xlabel('w1（x的系数）', fontsize=12)
plt.ylabel('w2（x²的系数）', fontsize=12)
plt.title('高斯先验分布的负对数等高线（参数约束）', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()



# --------------------------
# 5. 绘制联合分布等高线与MAP解
# --------------------------
# 复用似然的参数网格（聚焦解的附近）
w1_grid, w2_grid = np.meshgrid(w1_range, w2_range)

# 计算每个网格点的负对数联合分布（似然负对数 + 先验负对数）
neg_log_joint = np.zeros_like(w1_grid)
for i in range(len(w2_range)):
    for j in range(len(w1_range)):
        w = np.array([w1_grid[i,j], w2_grid[i,j]])
        neg_log_like = 0.5 * beta * np.sum((t_train - Phi_train @ w)**2)
        neg_log_pri = 0.5 * alpha_medium * np.sum(w**2)
        neg_log_joint[i,j] = neg_log_like + neg_log_pri

# 绘图
plt.figure(figsize=(8, 6))
contour_j = plt.contourf(w1_grid, w2_grid, neg_log_joint, levels=30, cmap=cm.Oranges)
plt.colorbar(contour_j, label='负对数联合分布值（越小越好）')
plt.contour(w1_grid, w2_grid, neg_log_joint, levels=10, colors='black', linewidths=0.5)

# 标注关键位置
plt.scatter(w_LS[0], w_LS[1], color='red', s=100, label='最小二乘解 (w_LS)', marker='*')
plt.scatter(w_MAP_medium[0], w_MAP_medium[1], color='blue', s=100, label='MAP解 (w_MAP)', marker='^')
plt.scatter(w_true[0], w_true[1], color='orange', s=80, label='真实参数 (w_true)', marker='o')

plt.xlabel('w1（x的系数）', fontsize=12)
plt.ylabel('w2（x²的系数）', fontsize=12)
plt.title('联合分布的负对数等高线（似然+先验）', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()



