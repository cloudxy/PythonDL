#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
trig_visualizer.py
~~~~~~~~~~~~~~~~~~

一个“一站式”三角函数可视化脚本，支持

1.  2‑D 六大三角函数的 2×3 子图
2.  3‑D 表面（默认: z = sin(x) + cos(y)）
3.  导出 STL / OBJ 供 3‑D 打印或后续建模使用

依赖: numpy, matplotlib, trimesh

使用方法
    python trig_visualizer.py  # 2‑D 与 3‑D 同时绘制并保存模型
"""

# ----------------------------------------------------------------------
# 0. 依赖 & 兼容
# ----------------------------------------------------------------------
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import trimesh  # 用于网格构建 & 对象导出
except Exception as exc:
    raise RuntimeError(
        "缺少必需的第三方库，请先执行 `pip install numpy matplotlib trimesh`"
    ) from exc

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import matplotlib

# ----------------------------------------------------------------------
# 1. 统一配置（字体、图例、路径）
# ----------------------------------------------------------------------
# 字体列表原始给出，若系统里没有可自行删减
FONT_LIST = [
    "BiauKaiHK",
    "Songti SC",
    "LiSong Pro",
    "Arial Unicode MS",
    "STIXNonUnicode",
    "Kaiti SC",
    "PingFang HK",
    "SimSong",
    "Kai",
    "Noto Sans Kaithi",
    "Heiti TC",
    "STHeiti",
    "Kailasa",
]

# 在 init() 内完成设置
def _apply_font_rc() -> None:
    matplotlib.rcParams["font.family"] = FONT_LIST
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# 2. 数据结构
# ----------------------------------------------------------------------
@dataclass
class SurfaceData:
    """三角函数 3‑D 网格数据."""

    vertices: np.ndarray  # shape (N, 3)
    faces: np.ndarray     # shape (M, 3)


# ----------------------------------------------------------------------
# 3. 主类
# ----------------------------------------------------------------------
class TrigVisualizer:
    """
    三角函数可视化与导出工具类.

    用法示例::

        vis = TrigVisualizer()
        vis.plot_2d()          # 2‑D 子图
        vis.visualize_3d()     # 3‑D 图
        vis.export_mesh()      # 导出 STL/OBJ
    """

    def __init__(self, title: str = "三角函数可视化") -> None:
        """
        Parameters
        ----------
        title : str
            整体窗口标题（会在 2‑D 画布与 3‑D 画布默认显示）。
        """
        self.title = title
        _apply_font_rc()

    # ------------------------------------------------------------------
    # 3‑D 表面生成
    # ------------------------------------------------------------------
    def generate_surface(
        self,
        nx: int = 400,
        ny: int = 400,
        *,
        z_expr: str = "np.sin(X) + np.cos(Y)",
    ) -> SurfaceData:
        """
        生成一个基于三角函数的 3‑D 表面网格.

        Parameters
        ----------
        nx, ny : int
            采样点数（越大图形越光滑，但文件体积也越大）。
        z_expr : str
            定义 `Z` 的表达式，默认 `np.sin(X) + np.cos(Y)`，你可以改成
            `np.sin(X) * np.cos(Y)`、`np.sin(X) - np.cos(Y)` …。

        Returns
        -------
        SurfaceData
            包含顶点和面片的结构体。
        """
        x = np.linspace(0, 2 * np.pi, nx)
        y = np.linspace(0, 2 * np.pi, ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # 动态求值
        Z = eval(z_expr, {"np": np, "X": X, "Y": Y})

        vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        faces = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                idx = i * ny + j
                a, b, c, d = idx, idx + 1, idx + ny, idx + ny + 1
                faces.append([a, b, d])
                faces.append([a, d, c])

        return SurfaceData(vertices=vertices, faces=np.array(faces, dtype=np.int32))

    # ------------------------------------------------------------------
    # 2‑D 六大函图
    # ------------------------------------------------------------------
    def plot_2d(self) -> None:
        """
        用 matplotlib 展示正弦、余弦、正切、余切、正割、余割六大函数。
        """

        # 生成数据
        x = np.linspace(-2 * np.pi, 2 * np.pi, 12000)
        sin_x, cos_x = np.sin(x), np.cos(x)
        tan_x = np.tan(x)
        cot_x = 1 / np.tan(x)
        sec_x = 1 / np.cos(x)
        csc_x = 1 / np.sin(x)

        # 处理无穷大
        mask = lambda a: np.ma.masked_where(np.abs(a) > 10, a)
        tan_x, cot_x, sec_x, csc_x = map(mask, [tan_x, cot_x, sec_x, csc_x])

        # 绘图区
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.subplots_adjust(hspace=0.3, wspace=0.2)

        titles = [
            ("正弦 ($\\sin x$)", sin_x),
            ("余弦 ($\\cos x$)", cos_x),
            ("正切 ($\\tan x$)", tan_x),
            ("余切 ($\\cot x$)", cot_x),
            ("正割 ($\\sec x$)", sec_x),
            ("余割 ($\\csc x$)", csc_x),
        ]

        for ax, (t, y) in zip(axes.flat, titles):
            ax.plot(x, y, color="tab:blue" if "正弦" in t or "余弦" in t else "tab:orange")
            ax.set_title(t)
            ax.set_xlabel("x")
            ax.set_ylabel(t.split("(", 1)[1][:-1])  # 取函数符（sin、cos…）
            if "正切" in t or "余切" in t or "正割" in t or "余割" in t:
                ax.set_ylim(-10, 10)
                for a in np.arange(-2.5 * np.pi, 2.5 * np.pi, np.pi):
                    ax.axvline(a, color="red", linestyle=":", linewidth=1)
            ax.grid(True, linestyle="--", linewidth=0.5)

        plt.suptitle(self.title, fontsize=16)
        plt.show()

    # ------------------------------------------------------------------
    # 3‑D 绘制 + 导出
    # ------------------------------------------------------------------
    def visualize_3d(self, surface: SurfaceData | None = None, z_expr="np.sin(X) + np.cos(Y)") -> None:
        """
        在 3‑D 视图中显示给定的表面.
        如果没有传 `surface`，会根据 `z_expr` 产生一个默认表面。
        """
        if surface is None:
            surface = self.generate_surface(z_expr=z_expr)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        x, y, z = surface.vertices[:, 0], surface.vertices[:, 1], surface.vertices[:, 2]
        ax.plot_trisurf(
            x,
            y,
            z,
            triangles=surface.faces,
            cmap="viridis",
            linewidth=0.2,
            antialiased=True,
            shade=True,
        )

        ax.set_title("3‑D 三角函数表面", fontsize=15, pad=20)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=30, azim=55)

        plt.tight_layout()
        plt.show()

    def export_mesh(
        self,
        surface: SurfaceData | None = None,
        *,
        stl_path: str | Path = "trig_surface.stl",
        obj_path: str | Path = "trig_surface.obj",
    ) -> Tuple[Path, Path]:
        """
        导出 STL 与 OBJ 文件。

        Returns
        -------
        Tuple[Path, Path]
            STL 路径, OBJ 路径
        """
        if surface is None:
            surface = self.generate_surface()

        mesh = trimesh.Trimesh(vertices=surface.vertices, faces=surface.faces, process=False)

        stl_p = Path(stl_path)
        obj_p = Path(obj_path)
        mesh.export(stl_p)
        mesh.export(obj_p)

        print(f"✅ STL exported to {stl_p.resolve()}")
        print(f"✅ OBJ exported to {obj_p.resolve()}")
        return stl_p.resolve(), obj_p.resolve()

    # ------------------------------------------------------------------
    # 入口 – 统一运行流程
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        nx: int = 400,
        ny: int = 400,
        z_expr: str = "np.sin(X) + np.cos(Y)",
    ) -> None:
        """
        1. 画 2‑D 子图
        2. 画 3‑D 图
        3. 导出 STL / OBJ
        """
        # 2‑D
        self.plot_2d()

        # 3‑D
        surface = self.generate_surface(nx=nx, ny=ny, z_expr=z_expr)
        self.visualize_3d(surface=surface)

        # 导出
        self.export_mesh(surface=surface)


# ----------------------------------------------------------------------
# 4. 入口脚本
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 这里你可以自行调整 nx/ny，或者修改 z_expr
    visualizer = TrigVisualizer(title="三角函数可视化")
    visualizer.run(nx=500, ny=500, z_expr="np.sin(X) * np.cos(Y)")
