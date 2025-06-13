"""学生模板：量子隧穿效应
文件：quantum_tunneling_student.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class QuantumTunnelingSolver:
    """量子隧穿求解器类

    该类实现了一维含时薛定谔方程的数值求解，用于模拟量子粒子的隧穿效应。
    使用变形的Crank-Nicolson方法进行时间演化，确保数值稳定性和概率守恒。
    """

    def __init__(self, Nx=220, Nt=300, x0=40, k0=0.5, d=10, barrier_width=3, barrier_height=1.0):
        """初始化量子隧穿求解器

        参数:
            Nx (int): 空间网格点数，默认220
            Nt (int): 时间步数，默认300
            x0 (float): 初始波包中心位置，默认40
            k0 (float): 初始波包动量(波数)，默认0.5
            d (float): 初始波包宽度参数，默认10
            barrier_width (int): 势垒宽度，默认3
            barrier_height (float): 势垒高度，默认1.0
        """
        self.Nx = Nx
        self.Nt = Nt
        self.x0 = x0
        self.k0 = k0
        self.d = d
        self.barrier_width = int(barrier_width)
        self.barrier_height = barrier_height

        # 创建空间网格
        self.x = np.arange(self.Nx)

        # 设置势垒
        self.V = self.setup_potential()

        # 初始化波函数矩阵和系数矩阵
        self.C = np.zeros((self.Nx, self.Nt), dtype=complex)
        self.B = np.zeros((self.Nx, self.Nt), dtype=complex)

    def wavefun(self, x):
        """高斯波包函数

        参数:
            x (np.ndarray): 空间坐标数组

        返回:
            np.ndarray: 初始波函数值

        数学公式:
            ψ(x,0) = exp(ik₀x) * exp(-(x-x₀)²ln10(2)/d²)
        """
        return np.exp(1j * self.k0 * x) * np.exp(-(x - self.x0) ** 2 * np.log(2) / self.d ** 2)

    def setup_potential(self):
        """设置势垒函数

        返回:
            np.ndarray: 势垒数组
        """
        self.V = np.zeros(self.Nx)
        barrier_start = self.Nx // 2
        barrier_end = barrier_start + self.barrier_width
        self.V[barrier_start:barrier_end] = self.barrier_height
        return self.V

    def build_coefficient_matrix(self):
        """构建变形的Crank-Nicolson格式的系数矩阵

        返回:
            np.ndarray: 系数矩阵A
        """
        main_diag = -2 + 2j - self.V
        off_diag = np.ones(self.Nx - 1)
        A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A

    def solve_schrodinger(self):
        """求解一维含时薛定谔方程

        返回:
            tuple: (x, V, B, C) - 空间网格, 势垒, 波函数矩阵, chi矩阵
        """
        A = self.build_coefficient_matrix()

        # 设置初始波函数
        self.B[:, 0] = self.wavefun(self.x)

        # 归一化初始波函数
        norm = np.sqrt(np.sum(np.abs(self.B[:, 0]) ** 2))
        self.B[:, 0] /= norm

        # 时间演化
        for t in range(self.Nt - 1):
            rhs = self.B[:, t]
            self.C[:, t + 1] = 4j*np.linalg.solve(A, rhs)
            self.B[:, t + 1] = self.C[:, t + 1] - self.B[:, t]

        return self.x, self.V, self.B, self.C

    def calculate_coefficients(self):
        """计算透射和反射系数

        返回:
            tuple: (T, R) - 透射系数和反射系数
        """
        barrier_start = self.Nx // 2
        barrier_end = barrier_start + self.barrier_width

        # 计算最终时间步的概率密度
        prob_density = np.abs(self.B[:, -1]) ** 2

        # 计算透射和反射概率
        T = np.sum(np.abs(prob_density[barrier_end:])) / np.sum(np.abs(prob_density))
        R = np.sum(np.abs(prob_density[:barrier_start])) / np.sum(np.abs(prob_density))

        return T, R

    def plot_evolution(self, time_indices=None):
        """绘制波函数演化图

        参数:
            time_indices (list): 要绘制的时间索引列表
        """
        if time_indices is None:
            time_indices = [0, self.Nt // 4, self.Nt // 2, 3 * self.Nt // 4, self.Nt - 1]

        plt.figure(figsize=(12, 8))

        for i, t in enumerate(time_indices):
            plt.subplot(len(time_indices), 1, i + 1)
            plt.plot(self.x, np.abs(self.B[:, t]) ** 2, label=f'|ψ|² at t={t}')
            plt.plot(self.x, self.V, 'r--', label='Potential')
            plt.legend()
            plt.ylabel('Probability Density')
            if i == len(time_indices) - 1:
                plt.xlabel('Position x')

        plt.suptitle('Quantum Tunneling Evolution')
        plt.tight_layout()
        plt.show()

    def create_animation(self, interval=20):
        """创建波包演化动画

        参数:
            interval (int): 动画帧间隔(毫秒)

        返回:
            matplotlib.animation.FuncAnimation
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, self.Nx)
        ax.set_ylim(0, 1.2 * np.max(np.abs(self.B) ** 2))
        line, = ax.plot([], [], lw=2)
        potential_line, = ax.plot([], [], 'r--', lw=2)

        def init():
            line.set_data([], [])
            potential_line.set_data([], [])
            return line, potential_line

        def animate(i):
            line.set_data(self.x, np.abs(self.B[:, i]) ** 2)
            potential_line.set_data(self.x, self.V)
            ax.set_title(f'Time step {i}/{self.Nt}')
            return line, potential_line

        anim = animation.FuncAnimation(fig, animate, frames=self.Nt,
                                       init_func=init, interval=interval,
                                       blit=True)
        plt.show()
        return anim

    def verify_probability_conservation(self):
        """验证概率守恒

        返回:
            np.ndarray: 每个时间步的总概率
        """
        total_prob = np.sum(np.abs(self.B) ** 2, axis=0)
        return total_prob

    def demonstrate(self):
        """演示量子隧穿效应

        返回:
            animation对象
        """
        print("Solving Schrödinger equation...")
        self.solve_schrodinger()

        T, R = self.calculate_coefficients()
        print(f"Transmission coefficient: {T:.4f}")
        print(f"Reflection coefficient: {R:.4f}")
        print(f"Total probability (T+R): {T + R:.4f}")

        print("Plotting evolution...")
        self.plot_evolution()

        print("Verifying probability conservation...")
        prob = self.verify_probability_conservation()
        plt.figure()
        plt.plot(prob)
        plt.title('Probability Conservation Check')
        plt.xlabel('Time step')
        plt.ylabel('Total probability')
        plt.show()

        print("Creating animation...")
        anim = self.create_animation()
        plt.show()
        return anim


def demonstrate_quantum_tunneling():
    """便捷的演示函数

    返回:
        animation对象
    """
    solver = QuantumTunnelingSolver()
    return solver.demonstrate()


if __name__ == "__main__":
    # 运行演示
    barrier_width = 3
    barrier_height = 1.0
    solver = QuantumTunnelingSolver(barrier_width=barrier_width, barrier_height=barrier_height)
    animation = solver.demonstrate()
