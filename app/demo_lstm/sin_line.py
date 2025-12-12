import matplotlib.pyplot as plt
import numpy as np

# 核心修改：使用你系统中存在的字体（优先选Arial Unicode MS）
plt.rcParams["font.family"] = "Arial Unicode MS"  # 首选，支持所有中文和符号
# 如果你想换其他字体，可替换为：
# plt.rcParams["font.family"] = "Songti SC"  # 宋体
# plt.rcParams["font.family"] = "STHeiti"    # 黑体
# plt.rcParams["font.family"] = "PingFang HK"# 苹方香港版（也支持简体中文）

plt.rcParams['axes.unicode_minus'] = False  # 必须保留，解决负号显示为方块的问题

# 你的sin_wave绘图代码（这里用正弦波示例代替，替换为你自己的代码即可）
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("正弦波曲线")  # 中文标题
plt.xlabel("横坐标（弧度）")  # 中文X轴标签
plt.ylabel("正弦值")  # 中文Y轴标签
plt.tight_layout()
plt.show()