#!/usr/bin/env python3
"""
多种数值方法求解热传导方程
文件: heat_equation_methods_solution.py

本模块实现了四种不同的数值方法来求解一维热传导方程：
1. 显式有限差分（FTCS）
2. 隐式有限差分（BTCS）
3. Crank-Nicolson方法
4. scipy.integrate.solve_ivp方法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg
import time

class HeatEquationSolver:
    """
    一个全面的求解器，使用多种数值方法求解一维热传导方程。
    
    热传导方程: du/dt = alpha * d²u/dx²
    边界条件: u(0,t) = 0, u(L,t) = 0
    初始条件: u(x,0) = phi(x)
    """
    
    def __init__(self, L=20.0, alpha=10.0, nx=21, T_final=25.0):
        """
        初始化热传导方程求解器。
        
        参数:
            L (float): 域长度 [0, L]
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
        设置初始条件: 当10 <= x <= 11时u(x,0) = 1，否则为0。
        
        返回:
            np.ndarray: 初始温度分布
        """
        u0 = np.zeros(self.nx)
        mask = (self.x >= 10) & (self.x <= 11)
        u0[mask] = 1.0
        # 应用边界条件
        u0[0] = 0.0
        u0[-1] = 0.0
        return u0
    
    def solve_explicit(self, dt=0.01, plot_times=None):
        """
        使用显式有限差分方法（FTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 稳定性检查
        r = self.alpha * dt / (self.dx**2)
        if r > 0.5:
            print(f"Warning: Stability condition violated! r = {r:.4f} > 0.5")
            print(f"Consider reducing dt to < {0.5 * self.dx**2 / self.alpha:.6f}")
        
        # 初始化
        u = self.u_initial.copy()
        t = 0.0
        nt = int(self.T_final / dt) + 1
        
        # 存储结果
        results = {'times': [], 'solutions': [], 'method': 'Explicit FTCS'}
        
        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进
        for n in range(1, nt):
            # 使用scipy.ndimage.laplace应用拉普拉斯算子
            du_dt = r * laplace(u)
            u += du_dt
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            t = n * dt
            
            # 在指定时间点存储解
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in [res_t for res_t in results['times']]:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
        return results
    
    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        使用隐式有限差分方法（BTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 参数
        r = self.alpha * dt / (self.dx**2)
        nt = int(self.T_final / dt) + 1
        
        # 初始化
        u = self.u_initial.copy()
        
        # 为内部节点构建三对角矩阵
        num_internal = self.nx - 2
        banded_matrix = np.zeros((3, num_internal))
        banded_matrix[0, 1:] = -r  # 上对角线
        banded_matrix[1, :] = 1 + 2*r  # 主对角线
        banded_matrix[2, :-1] = -r  # 下对角线
        
        # 存储结果
        results = {'times': [], 'solutions': [], 'method': 'Implicit BTCS'}
        
        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进
        for n in range(1, nt):
            # 右侧项（仅内部节点）
            rhs = u[1:-1].copy()
            
            # 求解三对角系统
            u_internal_new = scipy.linalg.solve_banded((1, 1), banded_matrix, rhs)
            
            # 更新解
            u[1:-1] = u_internal_new
            u[0] = 0.0  # 边界条件
            u[-1] = 0.0
            
            t = n * dt
            
            # 在指定时间点存储解
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in [res_t for res_t in results['times']]:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
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
            
        # 参数
        r = self.alpha * dt / (self.dx**2)
        nt = int(self.T_final / dt) + 1
        
        # 初始化
        u = self.u_initial.copy()
        
        # 为内部节点构建系数矩阵
        num_internal = self.nx - 2
        
        # 左侧矩阵A
        banded_matrix_A = np.zeros((3, num_internal))
        banded_matrix_A[0, 1:] = -r/2  # 上对角线
        banded_matrix_A[1, :] = 1 + r  # 主对角线
        banded_matrix_A[2, :-1] = -r/2  # 下对角线
        
        # 存储结果
        results = {'times': [], 'solutions': [], 'method': 'Crank-Nicolson'}
        
        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进
        for n in range(1, nt):
            # 右侧向量
            u_internal = u[1:-1]
            rhs = (r/2) * u[:-2] + (1 - r) * u_internal + (r/2) * u[2:]
            
            # 求解三对角系统 A * u^{n+1} = rhs
            u_internal_new = scipy.linalg.solve_banded((1, 1), banded_matrix_A, rhs)
            
            # 更新解
            u[1:-1] = u_internal_new
            u[0] = 0.0  # 边界条件
            u[-1] = 0.0
            
            t = n * dt
            
            # 在指定时间点存储解
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in [res_t for res_t in results['times']]:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
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
        # 重构带有边界条件的完整解
        u_full = np.concatenate(([0.0], u_internal, [0.0]))
        
        # 使用拉普拉斯算子计算二阶导数
        d2u_dx2 = laplace(u_full) / (self.dx**2)
        
        # 仅返回内部节点的导数
        return self.alpha * d2u_dx2[1:-1]
    
    def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        使用scipy.integrate.solve_ivp求解。
        
        参数:
            method (str): 积分方法 ('RK45', 'BDF', 'Radau' 等)
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 仅内部节点的初始条件
        u0_internal = self.u_initial[1:-1]
        
        start_time = time.time()
        
        # 求解ODE系统
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u0_internal,
            method=method,
            t_eval=plot_times,
            rtol=1e-8,
            atol=1e-10
        )
        
        computation_time = time.time() - start_time
        
        # 重构带有边界条件的完整解
        results = {
            'times': sol.t.tolist(),
            'solutions': [],
            'method': f'solve_ivp ({method})',
            'computation_time': computation_time
        }
        
        for i in range(len(sol.t)):
            u_full = np.concatenate(([0.0], sol.y[:, i], [0.0]))
            results['solutions'].append(u_full)
        
        return results
    
    def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5, 
                       ivp_method='BDF', plot_times=None):
        """
        比较所有四种数值方法。
        
        参数:
            dt_explicit (float): 显式方法的时间步长
            dt_implicit (float): 隐式方法的时间步长
            dt_cn (float): Crank-Nicolson方法的时间步长
            ivp_method (str): solve_ivp的积分方法
            plot_times (list): 比较时间点
            
        返回:
            dict: 所有方法的结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        print("Solving heat equation using four different methods...")
        print(f"Domain: [0, {self.L}], Grid points: {self.nx}, Final time: {self.T_final}")
        print(f"Thermal diffusivity: {self.alpha}")
        print("-" * 60)
        
        # 使用所有方法求解
        methods_results = {}
        
        # 显式方法
        print("1. Explicit finite difference (FTCS)...")
        methods_results['explicit'] = self.solve_explicit(dt_explicit, plot_times)
        print(f"   Computation time: {methods_results['explicit']['computation_time']:.4f} s")
        print(f"   Stability parameter r: {methods_results['explicit']['stability_parameter']:.4f}")
        
        # 隐式方法
        print("2. Implicit finite difference (BTCS)...")
        methods_results['implicit'] = self.solve_implicit(dt_implicit, plot_times)
        print(f"   Computation time: {methods_results['implicit']['computation_time']:.4f} s")
        print(f"   Stability parameter r: {methods_results['implicit']['stability_parameter']:.4f}")
        
        # Crank-Nicolson方法
        print("3. Crank-Nicolson method...")
        methods_results['crank_nicolson'] = self.solve_crank_nicolson(dt_cn, plot_times)
        print(f"   Computation time: {methods_results['crank_nicolson']['computation_time']:.4f} s")
        print(f"   Stability parameter r: {methods_results['crank_nicolson']['stability_parameter']:.4f}")
        
        # solve_ivp方法
        print(f"4. solve_ivp method ({ivp_method})...")
        methods_results['solve_ivp'] = self.solve_with_solve_ivp(ivp_method, plot_times)
        print(f"   Computation time: {methods_results['solve_ivp']['computation_time']:.4f} s")
        
        print("-" * 60)
        print("All methods completed successfully!")
        
        return methods_results
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。
        
        参数:
            methods_results (dict): compare_methods的结果
            save_figure (bool): 是否保存图像
            filename (str): 保存的文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        method_names = ['explicit', 'implicit', 'crank_nicolson', 'solve_ivp']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, method_name in enumerate(method_names):
            ax = axes[idx]
            results = methods_results[method_name]
            
            # 在不同时间绘制解
            for i, (t, u) in enumerate(zip(results['times'], results['solutions'])):
                ax.plot(self.x, u, color=colors[i], label=f't = {t:.1f}', linewidth=2)
            
            ax.set_title(f"{results['method']}\n(Time: {results['computation_time']:.4f} s)")
            ax.set_xlabel('Position x')
            ax.set_ylabel('Temperature u(x,t)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, self.L)
            ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        if save_figure:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {filename}")
        
        plt.show()
    
    def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        分析不同方法的精度。
        
        参数:
            methods_results (dict): compare_methods的结果
            reference_method (str): 用作参考的方法
            
        返回:
            dict: 精度分析结果
        """
        if reference_method not in methods_results:
            raise ValueError(f"Reference method '{reference_method}' not found in results")
        
        reference = methods_results[reference_method]
        accuracy_results = {}
        
        print(f"\nAccuracy Analysis (Reference: {reference['method']})")
        print("-" * 50)
        
        for method_name, results in methods_results.items():
            if method_name == reference_method:
                continue
                
            errors = []
            for i, (ref_sol, test_sol) in enumerate(zip(reference['solutions'], results['solutions'])):
                if i < len(results['solutions']):
                    error = np.linalg.norm(ref_sol - test_sol, ord=2)
                    errors.append(error)
            
            max_error = max(errors) if errors else 0
            avg_error = np.mean(errors) if errors else 0
            
            accuracy_results[method_name] = {
                'max_error': max_error,
                'avg_error': avg_error,
                'errors': errors
            }
            
            print(f"{results['method']:25} - Max Error: {max_error:.2e}, Avg Error: {avg_error:.2e}")
        
        return accuracy_results

def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=21, T_final=25.0)
    
    # 比较所有方法
    plot_times = [0, 1, 5, 15, 25]
    results = solver.compare_methods(
        dt_explicit=0.01,
        dt_implicit=0.1, 
        dt_cn=0.5,
        ivp_method='BDF',
        plot_times=plot_times
    )
    
    # 绘制比较图
    solver.plot_comparison(results, save_figure=True)
    
    # 分析精度
    accuracy = solver.analyze_accuracy(results, reference_method='solve_ivp')
    
    return solver, results, accuracy

if __name__ == "__main__":
    solver, results, accuracy = main()
