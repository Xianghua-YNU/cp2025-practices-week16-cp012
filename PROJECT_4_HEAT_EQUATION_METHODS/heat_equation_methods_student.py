#!/usr/bin/env python3
"""
学生模板：热传导方程数值解法比较
文件：heat_equation_methods_student.py
重要：函数名称必须与参考答案一致！
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg as la
import time


class HeatEquationSolver:
    """
    热传导方程求解器，实现四种不同的数值方法。

    求解一维热传导方程：du/dt = alpha * d²u/dx²
    边界条件：u(0,t) = 0, u(L,t) = 0
    初始条件：u(x,0) = phi(x)
    """

    def __init__(self, L=20.0, alpha=10.0, nx=21, T_final=25.0):
        """
        初始化热传导方程求解器。

        参数:
            L (float): 空间域长度 [0, L]
            alpha (float): 热扩散系数
            nx (int): 空间网格点数
            T_final (float): 最终模拟时间
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T_final = T_final

        # 空间网格
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)

        # 初始化解数组
        self.u_initial = self._set_initial_condition()

    def _set_initial_condition(self):
        """
        设置初始条件：u(x,0) = 1 当 10 <= x <= 11，否则为 0。

        返回:
            np.ndarray: 初始温度分布
        """
        # 创建长度为 nx 的零数组
        u = np.zeros(self.nx)
        # 设置初始条件：在 10 <= x <= 11 的网格点处设为 1
        mask = (self.x >= 10) & (self.x <= 11)
        u[mask] = 1
        # 边界条件已在零数组中自动满足：u[0] = 0, u[-1] = 0
        return u

    def solve_explicit(self, dt=0.01, plot_times=None):
        """
        使用显式有限差分法（FTCS）求解。

        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点

        返回:
            dict: 包含时间点和温度数组的解数据
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]

        # 计算稳定性参数 r = alpha * dt / dx²
        r = self.alpha * dt / self.dx ** 2
        # 检查稳定性条件 r <= 0.5
        if r > 0.5:
            print(f"显式方法警告：稳定性参数 r={r:.4f} > 0.5，可能不稳定")

        # 初始化解数组和时间变量
        u = self.u_initial.copy()
        t = 0
        # 创建结果存储字典并存储初始条件
        results = {0: u.copy()}
        plot_idx = 1  # 从 plot_times[1] 开始，因为 t=0 已存储

        # 时间步进循环
        while t < self.T_final:
            # 使用 laplace 计算空间二阶导数
            lapl = laplace(u, mode='constant')
            # 更新解（仅内部节点）
            u[1:-1] += r * lapl[1:-1]
            # 边界条件保持为 0，无需显式设置
            t += dt
            # 在指定时间点存储解
            if plot_idx < len(plot_times) and t >= plot_times[plot_idx]:
                results[plot_times[plot_idx]] = u.copy()
                plot_idx += 1
        return results

    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        使用隐式有限差分法（BTCS）求解。

        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点

        返回:
            dict: 包含时间点和温度数组的解数据
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]

        # 计算扩散数 r
        r = self.alpha * dt / self.dx ** 2
        # 构建三对角矩阵（内部节点 nx-2 个）
        ab = np.zeros((3, self.nx - 2))
        ab[1, :] = 1 + 2 * r  # 主对角线
        ab[0, 1:] = -r  # 上对角线
        ab[2, :-1] = -r  # 下对角线

        # 初始化解数组和结果存储
        u = self.u_initial.copy()
        results = {0: u.copy()}
        t = 0
        plot_idx = 1

        # 时间步进循环
        while t < self.T_final:
            # 构建右端项（内部节点）
            b = u[1:-1]
            # 使用 scipy.linalg.solve_banded 求解
            u_new = la.solve_banded((1, 1), ab, b)
            # 更新解（仅内部节点）
            u[1:-1] = u_new
            # 边界条件保持为 0
            t += dt
            # 在指定时间点存储解
            if plot_idx < len(plot_times) and t >= plot_times[plot_idx]:
                results[plot_times[plot_idx]] = u.copy()
                plot_idx += 1
        return results

    def solve_crank_nicolson(self, dt=0.5, plot_times=None):
        """
        使用Crank-Nicolson方法求解。

        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点

        返回:
            dict: 包含时间点和温度数组的解数据
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]

        # 计算扩散数 r
        r = self.alpha * dt / self.dx ** 2
        # 构建左端矩阵 A（内部节点 nx-2 个）
        ab = np.zeros((3, self.nx - 2))
        ab[1, :] = 1 + r  # 主对角线
        ab[0, 1:] = -r / 2  # 上对角线
        ab[2, :-1] = -r / 2  # 下对角线

        # 初始化解数组和结果存储
        u = self.u_initial.copy()
        results = {0: u.copy()}
        t = 0
        plot_idx = 1

        # 时间步进循环
        while t < self.T_final:
            # 构建右端向量
            rhs = (r / 2) * u[:-2] + (1 - r) * u[1:-1] + (r / 2) * u[2:]
            # 求解线性系统
            u_new = la.solve_banded((1, 1), ab, rhs)
            # 更新解
            u[1:-1] = u_new
            t += dt
            # 在指定时间点存储解
            if plot_idx < len(plot_times) and t >= plot_times[plot_idx]:
                results[plot_times[plot_idx]] = u.copy()
                plot_idx += 1
        return results

    def _heat_equation_ode(self, t, u_internal):
        """
        用于solve_ivp方法的ODE系统。

        参数:
            t (float): 当前时间
            u_internal (np.ndarray): 内部节点温度

        返回:
            np.ndarray: 内部节点的时间导数
        """
        # 重构包含边界条件的完整解
        u_full = np.zeros(self.nx)
        u_full[1:-1] = u_internal
        u_full[0] = 0
        u_full[-1] = 0
        # 使用 laplace 计算二阶导数
        lapl = laplace(u_full, mode='constant')
        # 返回内部节点的时间导数
        du_dt = self.alpha * lapl[1:-1] / self.dx ** 2
        return du_dt

    def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        使用scipy.integrate.solve_ivp求解。

        参数:
            method (str): 积分方法（'RK45', 'BDF', 'Radau'等）
            plot_times (list): 绘图时间点

        返回:
            dict: 包含时间点和温度数组的解数据
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]

        # 提取内部节点初始条件
        u0 = self.u_initial[1:-1]
        # 调用 solve_ivp 求解
        sol = solve_ivp(self._heat_equation_ode, (0, self.T_final), u0,
                        method=method, t_eval=plot_times)
        # 重构包含边界条件的完整解
        results = {}
        for i, t in enumerate(sol.t):
            u_full = np.zeros(self.nx)
            u_full[1:-1] = sol.y[:, i]
            u_full[0] = 0
            u_full[-1] = 0
            results[t] = u_full
        return results

    def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5,
                        ivp_method='BDF', plot_times=None):
        """
        比较所有四种数值方法。

        参数:
            dt_explicit (float): 显式方法时间步长
            dt_implicit (float): 隐式方法时间步长
            dt_cn (float): Crank-Nicolson方法时间步长
            ivp_method (str): solve_ivp积分方法
            plot_times (list): 比较时间点

        返回:
            dict: 所有方法的结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]

        print("开始比较四种数值方法...")
        methods = {}
        # 显式方法
        start = time.time()
        methods['explicit'] = self.solve_explicit(dt=dt_explicit, plot_times=plot_times)
        t_explicit = time.time() - start
        r_explicit = self.alpha * dt_explicit / self.dx ** 2
        # 隐式方法
        start = time.time()
        methods['implicit'] = self.solve_implicit(dt=dt_implicit, plot_times=plot_times)
        t_implicit = time.time() - start
        # Crank-Nicolson 方法
        start = time.time()
        methods['crank_nicolson'] = self.solve_crank_nicolson(dt=dt_cn, plot_times=plot_times)
        t_cn = time.time() - start
        # solve_ivp 方法
        start = time.time()
        methods['solve_ivp'] = self.solve_with_solve_ivp(method=ivp_method, plot_times=plot_times)
        t_ivp = time.time() - start

        # 打印计算时间和稳定性参数
        print(f"显式方法: 计算时间={t_explicit:.4f}s, r={r_explicit:.4f}")
        print(f"隐式方法: 计算时间={t_implicit:.4f}s")
        print(f"Crank-Nicolson方法: 计算时间={t_cn:.4f}s")
        print(f"solve_ivp方法: 计算时间={t_ivp:.4f}s")
        return methods

    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。

        参数:
            methods_results (dict): compare_methods的结果
            save_figure (bool): 是否保存图像
            filename (str): 保存的文件名
        """
        # 创建 2x2 子图
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        methods = ['explicit', 'implicit', 'crank_nicolson', 'solve_ivp']

        # 为每种方法绘制解曲线
        for i, method in enumerate(methods):
            ax = axs[i // 2, i % 2]
            for t in methods_results[method]:
                ax.plot(self.x, methods_results[method][t], label=f't={t}')
            ax.set_title(method)
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.legend()

        # 调整布局
        plt.tight_layout()
        # 可选保存图像
        if save_figure:
            plt.savefig(filename)
        else:
            plt.show()

    def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        分析不同方法的精度。

        参数:
            methods_results (dict): compare_methods的结果
            reference_method (str): 参考方法

        返回:
            dict: 精度分析结果
        """
        # 验证参考方法存在
        if reference_method not in methods_results:
            raise ValueError(f"参考方法 {reference_method} 未找到")

        # 选择参考解（在 t=25 时）
        ref = methods_results[reference_method][25]
        errors = {}

        # 计算各方法与参考解的误差
        for method in methods_results:
            if method != reference_method:
                u = methods_results[method][25]
                error = np.max(np.abs(u - ref))
                errors[method] = error

        # 打印精度分析结果
        print(f"\n精度分析（与 {reference_method} 在 t=25 比较的最大误差）:")
        for method, error in errors.items():
            print(f"{method}: {error:.6f}")
        return errors


def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver()
    # 比较所有方法
    results = solver.compare_methods()
    # 绘制比较图
    solver.plot_comparison(results)
    # 分析精度
    accuracy = solver.analyze_accuracy(results)
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()
