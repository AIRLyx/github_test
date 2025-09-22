import numpy as np
import matplotlib.pyplot as plt

# 1. 定义线性变换函数（输入x1,x2，输出y1,y2）
def linear_transform(x1, x2):
    y1 = x1 + 2 * x2  # 变换公式1
    y2 = 3 * x1 + 4 * x2  # 变换公式2
    return y1, y2


plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "DejaVu Sans"]
# 2. 定义原x1-x2空间的单位正方形顶点（按顺序排列，最后回到起点闭合图形）
square_vertices = np.array([[0, 0],   # A点：原点
                           [1, 0],    # B点：沿x1轴到(1,0)
                           [1, 1],    # C点：沿x2轴到(1,1)
                           [0, 1],    # D点：沿x1轴负方向到(0,1)
                           [0, 0]])   # 回到A点，闭合图形
x1_square = square_vertices[:, 0]  # 所有顶点的x1坐标
x2_square = square_vertices[:, 1]  # 所有顶点的x2坐标

# 3. 计算变换后y1-y2空间的平行四边形顶点
transformed_vertices = []
for x1, x2 in square_vertices:
    y1, y2 = linear_transform(x1, x2)
    transformed_vertices.append([y1, y2])
transformed_vertices = np.array(transformed_vertices)  # 转成numpy数组
y1_parallelogram = transformed_vertices[:, 0]  # 变换后顶点的y1坐标
y2_parallelogram = transformed_vertices[:, 1]  # 变换后顶点的y2坐标

# 4. 计算Jacobian矩阵和行列式（验证面积伸缩因子）
J = np.array([[1, 2], [3, 4]])  # 手动构造Jacobian矩阵（线性变换的J是常数）
det_J = np.linalg.det(J)        # 用numpy计算行列式
print(f"Jacobian矩阵 J = \n{J}")
print(f"Jacobian行列式 det(J) = {det_J}")
print(f"面积伸缩因子 = |det(J)| = {abs(det_J)}")  # 应该等于2

# 5. 计算原基向量和变换后的基向量
i = np.array([1, 0])  # 原x1方向基向量
j = np.array([0, 1])  # 原x2方向基向量
Ji = J @ i            # 变换后的x1方向基向量（J的第一列）
Jj = J @ j            # 变换后的x2方向基向量（J的第二列）
print(f"原基向量 i = {i}, 变换后 Ji = {Ji}")
print(f"原基向量 j = {j}, 变换后 Jj = {Jj}")

# 6. 画图：左右分图，左边原空间，右边变换后空间
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1行2列的子图

# 左图：x1-x2空间的单位正方形和基向量
ax1.plot(x1_square, x2_square, color='blue', linewidth=2, label='单位正方形')
# 画基向量i（红色箭头）
ax1.arrow(0, 0, i[0], i[1], head_width=0.05, color='red', label=f'i={i}')
# 画基向量j（绿色箭头）
ax1.arrow(0, 0, j[0], j[1], head_width=0.05, color='green', label=f'j={j}')
# 设置左图参数
ax1.set_xlim(-0.5, 1.5)  # x1范围
ax1.set_ylim(-0.5, 1.5)  # x2范围
ax1.set_xlabel('x1轴', fontsize=12)
ax1.set_ylabel('x2轴', fontsize=12)
ax1.set_title('原x1-x2空间：单位正方形与基向量', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.5)
ax1.axis('equal')  # 保证x1和x2轴单位长度一致

# 右图：y1-y2空间的平行四边形和新基向量
ax2.plot(y1_parallelogram, y2_parallelogram, color='orange', linewidth=2, label='变换后的平行四边形')
# 画变换后的基向量Ji（红色箭头，和左图对应）
ax2.arrow(0, 0, Ji[0], Ji[1], head_width=0.05, color='red', label=f'Ji={Ji}')
# 画变换后的基向量Jj（绿色箭头，和左图对应）
ax2.arrow(0, 0, Jj[0], Jj[1], head_width=0.05, color='green', label=f'Jj={Jj}')
# 标注面积伸缩因子
ax2.text(0.5, 3, f'面积伸缩因子=|det(J)|={abs(det_J)}', fontsize=12, color='purple')
# 设置右图参数
ax2.set_xlim(-1, 4)  # y1范围
ax2.set_ylim(-1, 8)  # y2范围
ax2.set_xlabel('y1轴', fontsize=12)
ax2.set_ylabel('y2轴', fontsize=12)
ax2.set_title('y1-y2空间：平行四边形与新基向量', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.5)
ax2.axis('equal')  # 保证y1和y2轴单位长度一致

# 调整子图间距，避免重叠
plt.tight_layout()
plt.show()