import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch

# ============================================================
# 1. 创建画布
# ============================================================

fig, ax = plt.subplots(figsize=(18, 7.5))

ax.set_xlim(0, 20)
ax.set_ylim(0, 8)
ax.axis("off")


# ============================================================
# 2. 颜色
# ============================================================

blue = "#1F4E79"       # 前向传播
red = "#C72516"        # 反向传播
gray = "#555555"       # 圆形边框
light_gray = "#F2F2F2" # 圆形背景

green = "#DDEED5"      # 函数模块背景
green_edge = "#668F4D" # 函数模块边框

formula_red = "#B52A1C"


# ============================================================
# 3. 辅助函数
# ============================================================

def add_circle(x, y, radius, text, fontsize=30):
    """绘制圆形节点"""

    circle = Circle(
        (x, y),
        radius,
        facecolor=light_gray,
        edgecolor=gray,
        linewidth=3,
        zorder=3
    )

    ax.add_patch(circle)

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#222222",
        zorder=4
    )


def add_arrow(
        x1, y1,
        x2, y2,
        color,
        linewidth=2.5,
        mutation_scale=18,
        linestyle="-"):

    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        zorder=5
    )

    ax.add_patch(arrow)


# ============================================================
# 4. 左侧输入节点
# ============================================================

# x
add_circle(
    2.5, 5.9,
    1.05,
    r"$x$",
    fontsize=34
)

# omega
add_circle(
    2.5, 2.25,
    1.05,
    r"$\omega$",
    fontsize=34
)


# ============================================================
# 5. 中间函数模块 f
# ============================================================

function_box = FancyBboxPatch(
    (8.2, 0.95),
    5.8,
    6.1,
    boxstyle="round,pad=0.02,rounding_size=0.45",
    facecolor=green,
    edgecolor=green_edge,
    linewidth=3,
    zorder=1
)

ax.add_patch(function_box)


# 函数 f
ax.text(
    11.1,
    3.95,
    r"$f$",
    ha="center",
    va="center",
    fontsize=52,
    color="#334422",
    zorder=4
)


# ============================================================
# 6. 中间两个偏导数方框
# ============================================================

# ∂z / ∂x
box1 = Rectangle(
    (8.62, 5.05),
    1.25,
    1.55,
    facecolor="white",
    edgecolor=formula_red,
    linewidth=3,
    zorder=3
)

ax.add_patch(box1)

ax.text(
    9.245,
    5.83,
    r"$\frac{\partial z}{\partial x}$",
    ha="center",
    va="center",
    fontsize=27,
    color="#222222",
    zorder=4
)


# ∂z / ∂ω
box2 = Rectangle(
    (8.62, 1.50),
    1.25,
    1.55,
    facecolor="white",
    edgecolor=formula_red,
    linewidth=3,
    zorder=3
)

ax.add_patch(box2)

ax.text(
    9.245,
    2.28,
    r"$\frac{\partial z}{\partial \omega}$",
    ha="center",
    va="center",
    fontsize=27,
    color="#222222",
    zorder=4
)


# ============================================================
# 7. 右侧 z 节点
# ============================================================

add_circle(
    17.15, 4.25,
    0.78,
    r"$z$",
    fontsize=34
)


# ============================================================
# 8. x -> f
# ============================================================

add_arrow(
    3.55, 5.9,
    8.18, 5.9,
    blue,
    linewidth=3,
    mutation_scale=20
)

# x 标签
ax.text(
    5.95,
    6.18,
    r"$x$",
    ha="center",
    va="center",
    fontsize=28,
    color=blue
)


# ============================================================
# 9. omega -> f
# ============================================================

add_arrow(
    3.55, 2.25,
    8.18, 2.25,
    blue,
    linewidth=3,
    mutation_scale=20
)

# omega 标签
ax.text(
    5.95,
    2.53,
    r"$\omega$",
    ha="center",
    va="center",
    fontsize=28,
    color=blue
)


# ============================================================
# 10. f -> z
# ============================================================

add_arrow(
    14.0, 4.25,
    16.35, 4.25,
    blue,
    linewidth=3,
    mutation_scale=20
)

ax.text(
    15.15,
    4.58,
    r"$f(x,\omega)$",
    ha="center",
    va="center",
    fontsize=27,
    color=blue
)


# ============================================================
# 11. z -> Loss
# ============================================================

add_arrow(
    17.93, 4.25,
    19.15, 4.25,
    blue,
    linewidth=2.5,
    mutation_scale=17,
    linestyle=":"
)

ax.text(
    19.45,
    4.25,
    "Loss",
    ha="center",
    va="center",
    fontsize=23,
    color="#444444"
)


# ============================================================
# 12. z -> f 的反向传播
# ============================================================

add_arrow(
    16.35, 3.55,
    14.0, 3.55,
    red,
    linewidth=3,
    mutation_scale=20
)

# ∂L / ∂z
ax.text(
    15.15,
    3.05,
    r"$\frac{\partial L}{\partial z}$",
    ha="center",
    va="center",
    fontsize=29,
    color=red
)


# ============================================================
# 13. f -> x 的反向传播
# ============================================================

add_arrow(
    8.18, 5.20,
    3.55, 5.20,
    red,
    linewidth=3,
    mutation_scale=20
)

# 链式法则
ax.text(
    5.65,
    4.52,
    r"$\frac{\partial L}{\partial x}"
    r"="
    r"\frac{\partial L}{\partial z}"
    r"\cdot"
    r"\frac{\partial z}{\partial x}$",
    ha="center",
    va="center",
    fontsize=27,
    color=red
)


# ============================================================
# 14. f -> omega 的反向传播
# ============================================================

add_arrow(
    8.18, 1.60,
    3.55, 1.60,
    red,
    linewidth=3,
    mutation_scale=20
)

# 链式法则
ax.text(
    5.65,
    0.88,
    r"$\frac{\partial L}{\partial \omega}"
    r"="
    r"\frac{\partial L}{\partial z}"
    r"\cdot"
    r"\frac{\partial z}{\partial \omega}$",
    ha="center",
    va="center",
    fontsize=27,
    color=red
)


# ============================================================
# 15. 保存
# ============================================================

plt.tight_layout()

plt.savefig(
    "backpropagation_diagram.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()