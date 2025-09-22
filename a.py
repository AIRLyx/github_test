import numpy as np
import matplotlib.pyplot as plt

# 1. 定义目标函数和约束函数
def f(x, y):
    return x**2 + y**2  # 目标函数，等高线是同心圆

def g(x, y):
    return x + y - 1    # 约束函数，g=0对应直线x+y=1

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "DejaVu Sans"]
# 2. 生成网格数据（用于绘制等高线）
x = np.linspace(-0.5, 1.5, 100)  # x范围包含极值点(0.5,0.5)，留了点余量
y = np.linspace(-0.5, 1.5, 100)
X, Y = np.meshgrid(x, y)  # 生成二维网格
Z_f = f(X, Y)  # 计算每个网格点上f的值，用于画等高线

# 3. 计算极值点和梯度（验证平行关系）
extreme_x, extreme_y = 0.5, 0.5  # 极值点坐标
# 计算∇f在极值点的值：∇f=(2x, 2y)
grad_f_x = 2 * extreme_x
grad_f_y = 2 * extreme_y
# 计算∇g在极值点的值：∇g=(1, 1)（因为g=x+y-1，偏导数都是1）
grad_g_x = 1
grad_g_y = 1

# 4. 开始画图
plt.figure(figsize=(10, 8))  # 设置图的大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 4.1 绘制f的等高线
# levels=10表示画10条等高线，cmap选了viridis配色，比较清晰
contour_f = plt.contour(X, Y, Z_f, levels=10, cmap='viridis')
plt.clabel(contour_f, inline=True, fontsize=10)  # 给等高线标上数值
plt.title('f(x,y)=x^2+y^2的等高线与约束g(x,y)=x+y-1=0的极值点', fontsize=12)
plt.xlabel('x轴', fontsize=10)
plt.ylabel('y轴', fontsize=10)
plt.grid(True, alpha=0.3)  # 画网格线，方便看坐标

# 4.2 绘制约束直线g(x,y)=0（即x+y=1）
# 直线方程变形为y=1-x，取两个端点画直线
x_g = np.linspace(-0.5, 1.5, 100)  # x范围和网格一致
y_g = 1 - x_g
plt.plot(x_g, y_g, 'r-', label='约束曲线：g(x,y)=x+y-1=0', linewidth=2)

# 4.3 标记极值点
plt.scatter(extreme_x, extreme_y, color='red', s=100, label='极值点(0.5, 0.5)', zorder=5)
# 给极值点加注释，用箭头指过去，更清楚
plt.annotate('极值点 (0.5, 0.5)', xy=(extreme_x, extreme_y),
             xytext=(extreme_x+0.1, extreme_y+0.1),
             arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)

# 4.4 绘制梯度向量（用quiver函数，参数：起点x,起点y,向量x分量,向量y分量）
# 为了图好看，把梯度向量缩小了0.2倍，不然太长会挡住其他元素
scale = 0.2
# 绘制∇f向量（蓝色）
plt.quiver(extreme_x, extreme_y, grad_f_x*scale, grad_f_y*scale,
           angles='xy', scale_units='xy', scale=1, color='blue',
           label='∇f=(1,1)（缩小0.2倍）', zorder=4)
# 绘制∇g向量（绿色）
plt.quiver(extreme_x, extreme_y, grad_g_x*scale, grad_g_y*scale,
           angles='xy', scale_units='xy', scale=1, color='green',
           label='∇g=(1,1)（缩小0.2倍）', zorder=4)

# 4.5 显示图例和保存图片
plt.legend(loc='upper right', fontsize=10)
# 保存图片，dpi=300保证清晰度，bbox_inches='tight'防止标签被截断
plt.savefig('条件极值几何展示.png', dpi=300, bbox_inches='tight')
plt.show()