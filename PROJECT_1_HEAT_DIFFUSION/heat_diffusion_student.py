"""
学生模板：铝棒热传导问题
文件：heat_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
"""
学生模板：铝棒热传导问题
文件：heat_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理参数
K = 237       # 热导率 (W/m/K)
C = 900       # 比热容 (J/kg/K)
rho = 2700    # 密度 (kg/m^3)
D = K/(C*rho) # 热扩散系数
L = 1         # 铝棒长度 (m)
dx = 0.01     # 空间步长 (m)
dt = 0.5      # 时间步长 (s)
Nx = int(L/dx) + 1 # 空间格点数
Nt = 2000     # 时间步数

def basic_heat_diffusion():
    """
    任务1: 基本热传导模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    return np.zeros((Nx, Nt))

def analytical_solution(n_terms=100):
    """
    任务2: 解析解函数
    
    参数:
        n_terms (int): 傅里叶级数项数
    
    返回:
        np.ndarray: 解析解温度分布
    """
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    return np.zeros((Nx, Nt))

def stability_analysis():
    """
    任务3: 数值解稳定性分析
    """
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def different_initial_condition():
    """
    任务4: 不同初始条件模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")
    return np.zeros((Nx, Nt))

def heat_diffusion_with_cooling():
    """
    任务5: 包含牛顿冷却定律的热传导
    """
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def plot_3d_solution(u, dx, dt, Nt, title):
    """
    绘制3D温度分布图
    
    参数:
        u (np.ndarray): 温度分布数组
        dx (float): 空间步长
        dt (float): 时间步长
        Nt (int): 时间步数
        title (str): 图表标题
    
    返回:
        None
    
    示例:
        >>> u = np.zeros((100, 200))
        >>> plot_3d_solution(u, 0.01, 0.5, 200, "示例")
    """
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

if __name__ == "__main__":
    """
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
