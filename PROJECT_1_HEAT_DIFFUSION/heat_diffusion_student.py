"""
学生模板：铝棒热传导问题
文件：heat_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

# 物理参数
K = 237       # 热导率 (W/m/K)
C = 900       # 比热容 (J/kg/K)
rho = 2700    # 密度 (kg/m^3)
D = K/(C*rho) # 热扩散系数，计算方式为热导率除以比热容与密度的乘积
L = 1         # 铝棒长度 (m)
dx = 0.01     # 空间步长 (m)
dt = 0.5      # 时间步长 (s)
Nx = int(L/dx) + 1 # 空间格点数，根据铝棒长度和空间步长计算得到
Nt = 2000  # 时间步数

# 任务1: 基本热传导模拟
def basic_heat_diffusion():
    """
    任务1: 基本热传导模拟
    使用显式有限差分法求解一维热传导方程
    """
    r = D*dt/(dx**2)  # 计算稳定性参数r
    print(f"任务1 - 稳定性参数 r = {r}")
    
    u = np.zeros((Nx, Nt))  # 初始化温度分布矩阵
    u[:, 0] = 100  # 初始时刻铝棒温度设为100K
    u[0, :] = 0  # 左端边界条件设为0K
    u[-1, :] = 0  # 右端边界条件设为0K
    
    # 显式有限差分法迭代计算温度分布
    for j in range(Nt-1):
        u[1:-1, j+1] = (1-2*r)*u[1:-1, j] + r*(u[2:, j] + u[:-2, j])
    
    return u

# 任务2: 解析解与数值解比较
def analytical_solution(n_terms=100):
    """
    解析解函数
    计算热传导方程的解析解
    """
    x = np.linspace(0, dx*(Nx-1), Nx)  # 空间坐标
    t = np.linspace(0, dt*Nt, Nt)  # 时间坐标
    x, t = np.meshgrid(x, t)  # 生成网格坐标矩阵
    s = 0  # 初始化解析解
    # 级数求和计算解析解
    for i in range(n_terms):
        j = 2*i + 1
        s += 400/(j*np.pi) * np.sin(j*np.pi*x/L) * np.exp(-(j*np.pi/L)**2 * t * D)
    return s.T  # 返回解析解的转置

# 任务3: 数值解稳定性分析
def stability_analysis():
    """
    任务3: 数值解稳定性分析
    研究稳定性参数r对数值解的影响
    """
    dx = 0.01
    dt = 0.6  # 选择较大的时间步长使r>0.5
    r = D*dt/(dx**2)
    print(f"任务3 - 稳定性参数 r = {r} (r>0.5)")
    
    Nx = int(L/dx) + 1
    Nt = 2000
    
    u = np.zeros((Nx, Nt))  # 初始化温度分布矩阵
    u[:, 0] = 100  # 初始时刻铝棒温度设为100K
    u[0, :] = 0  # 左端边界条件设为0K
    u[-1, :] = 0  # 右端边界条件设为0K
    
    # 显式有限差分法迭代计算温度分布
    for j in range(Nt-1):
        u[1:-1, j+1] = (1-2*r)*u[1:-1, j] + r*(u[2:, j] + u[:-2, j])
    
    # 可视化不稳定解
    plot_3d_solution(u, dx, dt, Nt, title='Task 3: Unstable Solution (r>0.5)')

# 任务4: 不同初始条件模拟
def different_initial_condition():
    """
    任务4: 不同初始条件模拟
    模拟两根不同初始温度的铝棒连接处的热传导
    """
    dx = 0.01
    dt = 0.5
    r = D*dt/(dx**2)
    print(f"任务4 - 稳定性参数 r = {r}")
    
    Nx = int(L/dx) + 1
    Nt = 1000
    
    u = np.zeros((Nx, Nt))  # 初始化温度分布矩阵
    u[:51, 0] = 100  # 左半部分初始温度设为100K
    u[50:, 0] = 50   # 右半部分初始温度设为50K
    u[0, :] = 0  # 左端边界条件设为0K
    u[-1, :] = 0  # 右端边界条件设为0K
    
    # 显式有限差分法迭代计算温度分布
    for j in range(Nt-1):
        u[1:-1, j+1] = (1-2*r)*u[1:-1, j] + r*(u[2:, j] + u[:-2, j])
    
    # 可视化温度分布
    plot_3d_solution(u, dx, dt, Nt, title='Task 4: Temperature Evolution with Different Initial Conditions')
    return u

# 任务5: 包含牛顿冷却定律的热传导
def heat_diffusion_with_cooling():
    """
    任务5: 包含牛顿冷却定律的热传导
    考虑铝棒与外部环境的热交换
    """
    r = D*dt/(dx**2)
    h = 0.1  # 冷却系数
    print(f"任务5 - 稳定性参数 r = {r}, 冷却系数 h = {h}")
    
    Nx = int(L/dx) + 1
    Nt = 100
    
    u = np.zeros((Nx, Nt))  # 初始化温度分布矩阵
    u[:, 0] = 100  # 初始时刻铝棒温度设为100K
    u[0, :] = 0  # 左端边界条件设为0K
    u[-1, :] = 0  # 右端边界条件设为0K
    
    # 修改显式差分法的迭代公式以包含牛顿冷却项
    for j in range(Nt-1):
        u[1:-1, j+1] = (1-2*r-h*dt)*u[1:-1, j] + r*(u[2:, j] + u[:-2, j])
    
    # 可视化温度分布
    plot_3d_solution(u, dx, dt, Nt, title='Task 5: Heat Diffusion with Newton Cooling')

def plot_3d_solution(u, dx, dt, Nt, title):
    """
    绘制三维温度分布图
    """
    Nx = u.shape[0]
    x = np.linspace(0, dx*(Nx-1), Nx)  # 空间坐标
    t = np.linspace(0, dt*Nt, Nt)  # 时间坐标
    X, T = np.meshgrid(x, t)  # 生成网格坐标矩阵
    
    fig = plt.figure(figsize=(10, 6))  # 创建图形窗口
    ax = fig.add_subplot(111, projection='3d')  # 添加3D子图
    ax.plot_surface(X, T, u.T, cmap='rainbow')  # 绘制3D曲面图
    ax.set_xlabel('Position x (m)')  # 设置x轴标签
    ax.set_ylabel('Time t (s)')  # 设置y轴标签
    ax.set_zlabel('Temperature T (K)')  # 设置z轴标签
    ax.set_title(title)  # 设置图形标题
    plt.show()  # 显示图形

if __name__ == "__main__":
    # 程序入口
    print("=== 铝棒热传导问题参考答案 ===")
    
    print("\n1. 基本热传导模拟")
    u = basic_heat_diffusion()  # 调用任务1的函数
    plot_3d_solution(u, dx, dt, Nt, title='Task 1: Heat Diffusion Solution')  # 绘制结果

    print("\n2. 解析解")
    s = analytical_solution()  # 调用任务2的函数
    plot_3d_solution(s, dx, dt, Nt, title='Analytical Solution')  # 绘制结果

    print("\n3. 数值解稳定性分析")
    stability_analysis()  # 调用任务3的函数
    
    print("\n4. 不同初始条件模拟")
    different_initial_condition()  # 调用任务4的函数
    
    print("\n5. 包含牛顿冷却定律的热传导")
    heat_diffusion_with_cooling()  # 调用任务5的函数
    主函数 - 演示和测试各任务功能
    
    执行顺序:
    1. 基本热传导模拟
    2. 解析解计算
    3. 数值解稳定性分析
    4. 不同初始条件模拟
    5. 包含冷却效应的热传导
    
    注意:
        学生需要先实现各任务函数才能正常运行
    """
    print("=== 铝棒热传导问题学生实现 ===")
    print("请先实现各任务函数后再运行主程序")
