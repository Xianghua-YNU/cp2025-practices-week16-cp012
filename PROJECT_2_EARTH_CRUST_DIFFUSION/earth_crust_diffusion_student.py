"""
学生模板：地壳热扩散数值模拟
文件：earth_crust_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus']= False

def solve_earth_crust_diffusion():
    """
    实现显式差分法求解地壳热扩散问题
    
    返回:
        tuple: (depth_array, temperature_matrix)
        depth_array: 深度坐标数组 (m)
        temperature_matrix: 温度场矩阵 (°C)
    
    物理背景: 模拟地壳中温度随深度和时间的周期性变化
    数值方法: 显式差分格式
    
    实现步骤:
    1. 设置物理参数和网格参数
    2. 初始化温度场
    3. 应用边界条件
    4. 实现显式差分格式
    5. 返回计算结果
    """
    # 设置物理参数
    D = 0.1  # 热扩散率 (m^2/day)
    tau = 365  # 周期 (day)
    A = 10  # 地表平均温度 (°C)
    B = 12  # 地表温度振幅 (°C)
    T_bottom = 11  # 底部温度 (°C)

    # 设置网格参数
    L = 20  # 深度范围 (m)
    Nz = 101  # 深度方向网格点数
    dt = 0.1  # 时间步长 (day)
    total_time = 9 * tau  # 总模拟时间 (day)
    Nt = int(total_time / dt)  # 时间步数

    # 初始化数组
    dz = L / (Nz - 1)  # 深度步长 (m)
    z = np.linspace(0, L, Nz)  # 深度坐标数组 (m)
    T = np.zeros((Nt + 1, Nz))  # 温度场矩阵 (°C)

    # 设置初始条件（可以假设初始温度为常数）
    T[0, :] = A

    # 设置边界条件
    T[:, 0] = A + B * np.sin(2 * np.pi * np.arange(0, Nt + 1) * dt / tau)  # 地表温度
    T[:, -1] = T_bottom  # 底部温度

    # 计算稳定性条件
    alpha = D * dt / dz ** 2
    if alpha > 0.5:
        print(f"警告: 显式差分格式可能不稳定，alpha = {alpha:.4f} > 0.5")

    # 显式差分格式求解
    for n in range(Nt):
        for i in range(1, Nz - 1):
            T[n + 1, i] = T[n, i] + alpha * (T[n, i + 1] - 2 * T[n, i] + T[n, i - 1])

    # 返回计算结果
    return z, T

if __name__ == "__main__":
    # 测试代码
    try:
        # 运行模拟
        depth, T = solve_earth_crust_diffusion()
        print(f"计算完成，温度场形状: {T.shape}")
    
        # 可视化结果
        plt.figure(figsize=(12, 10))
    
        # 1. 长期演化分析
        plt.subplot(2, 2, 1)
        time_years = np.arange(T.shape[0]) * 0.1 / 365  # 转换为年
        depths_to_plot = [0, 5, 10, 15, 20]  # 要绘制的深度 (m)
        depth_indices = [int(d / 20 * (len(depth) - 1)) for d in depths_to_plot]
    
        for i, d_idx in enumerate(depth_indices):
            plt.plot(time_years, T[:, d_idx], label=f'{depths_to_plot[i]} m')
    
        plt.xlabel('时间 (年)')
        plt.ylabel('温度 (°C)')
        plt.title('不同深度处的温度随时间变化')
        plt.legend()
        plt.grid(True)
    
        # 2. 振幅衰减分析
        plt.subplot(2, 2, 2)
        # 只分析最后一年的数据
        last_year_data = T[-int(365 / 0.1):, :]
        amplitudes = np.max(last_year_data, axis=0) - np.min(last_year_data, axis=0)
    
        plt.plot(depth, amplitudes)
        plt.xlabel('深度 (m)')
        plt.ylabel('温度振幅 (°C)')
        plt.title('温度振幅随深度的衰减')
        plt.grid(True)
    
        # 3. 相位延迟分析
        plt.subplot(2, 2, 3)
        # 找出地表温度达到最大值的时间点
        surface_max_idx = np.argmax(T[-int(365 / 0.1):, 0])
    
        # 找出不同深度温度达到最大值的时间点
        phase_delays = []
        for i in range(len(depth)):
            max_idx = np.argmax(T[-int(365 / 0.1):, i])
            phase_delay = (max_idx - surface_max_idx) * 0.1  # 转换为天
            phase_delays.append(phase_delay)
    
        plt.plot(depth, phase_delays)
        plt.xlabel('深度 (m)')
        plt.ylabel('相位延迟 (天)')
        plt.title('温度相位延迟随深度的变化')
        plt.grid(True)
    
        # 4. 季节性温度轮廓
        plt.subplot(2, 2, 4)
        # 选择第10年的4个时间点（代表四季）
        year_start = -int(365 / 0.1)
        spring = year_start + int(90 / 0.1)
        summer = year_start + int(180 / 0.1)
        autumn = year_start + int(270 / 0.1)
        winter = year_start
    
        plt.plot(T[winter, :], depth, 'b-', label='冬季')
        plt.plot(T[spring, :], depth, 'g-', label='春季')
        plt.plot(T[summer, :], depth, 'r-', label='夏季')
        plt.plot(T[autumn, :], depth, 'y-', label='秋季')
    
        plt.xlabel('温度 (°C)')
        plt.ylabel('深度 (m)')
        plt.title('不同季节的温度随深度变化')
        plt.legend()
        plt.gca().invert_yaxis()  # 深度向下增加
        plt.grid(True)
    
        plt.tight_layout()
        plt.savefig('earth_crust_diffusion_results.png', dpi=300)
        plt.show()
    except NotImplementedError as e:
        print(e)
