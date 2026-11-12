import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

# ============================================================
# 1. 基本设置
# ============================================================
fig, ax = plt.subplots(figsize=(14, 12))

ax.set_xlim(0, 14)
ax.set_ylim(0, 13)
ax.axis("off")

# 颜色
red = "#C8320A"
light_green = "#DCEAD4"
green_edge = "#78985F"
light_blue = "#DCECF8"
blue_edge = "#6A9BC5"
light_orange = "#FCE8D8"
orange_edge = "#C9854E"
arrow_color = "#444444"

# ============================================================
# 2. 辅助函数
# ============================================================

def add_circle(x, y, radius=0.55, text="", 
               facecolor=light_orange, edgecolor=orange_edge,
               fontsize=18, textcolor="black", linewidth=2):
    """画圆形节点"""
    circle = Circle(
        (x, y),
        radius,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3
    )
    ax.add_patch(circle)

    ax.text(
        x, y, text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        zorder=4
    )


def add_box(x, y, width, height, text,
            facecolor=light_green,
            edgecolor=green_edge,
            fontsize=18):
    """画矩形模块"""
    box = Rectangle(
        (x, y),
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2,
        joinstyle="round",
        zorder=2
    )
    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=4
    )


def add_arrow(x1, y1, x2, y2,
              color=arrow_color,
              linewidth=2.5,
              mutation_scale=18):
    """画箭头"""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        zorder=5
    )
    ax.add_patch(arrow)


# ============================================================
# 3. 第一层的大框
# ============================================================

# 第一层主体框
ax.add_patch(
    Rectangle(
        (0.4, 7.0),
        13.0,
        4.4,
        facecolor="white",
        edgecolor=red,
        linewidth=4,
        zorder=0
    )
)

# First Layer 标题
ax.add_patch(
    Rectangle(
        (7.0, 10.35),
        6.4,
        1.05,
        facecolor=red,
        edgecolor=red,
        linewidth=0
    )
)

ax.text(
    10.2, 10.87,
    "First Layer",
    ha="center",
    va="center",
    fontsize=28,
    color="white",
    fontweight="bold"
)

# ============================================================
# 4. 第一层节点
# ============================================================

# 输入 X
add_circle(
    4.0, 12.25,
    radius=0.62,
    text=r"$X$",
    fontsize=25
)

# W1
add_circle(
    1.2, 10.55,
    radius=0.62,
    text=r"$W_1$",
    facecolor=light_blue,
    edgecolor=blue_edge,
    fontsize=22
)

# MM
add_box(
    3.2, 9.9,
    1.6, 1.05,
    r"$MM$",
    fontsize=21
)

# MM 后面的中间节点
add_circle(
    4.0, 9.35,
    radius=0.28,
    text="",
    fontsize=12
)

# b1
add_circle(
    1.2, 7.85,
    radius=0.62,
    text=r"$b_1$",
    facecolor=light_blue,
    edgecolor=blue_edge,
    fontsize=22
)

# ADD
add_box(
    3.2, 7.45,
    1.6, 1.05,
    r"$ADD$",
    fontsize=21
)

# ADD 后的节点
add_circle(
    6.05, 7.95,
    radius=0.30,
    text="",
    fontsize=12
)

# sigma
add_box(
    8.0, 7.45,
    1.6, 1.05,
    r"$\sigma$",
    fontsize=23
)

# Nonlinear Function
ax.text(
    9.85, 7.98,
    "Nonlinear\nFunction",
    ha="left",
    va="center",
    fontsize=23,
    color="#333333"
)

# ============================================================
# 5. 第一层箭头
# ============================================================

# X -> MM
add_arrow(
    4.0, 11.63,
    4.0, 10.98
)

# W1 -> MM
add_arrow(
    1.82, 10.55,
    3.18, 10.55
)

# MM -> 中间节点
add_arrow(
    4.0, 9.9,
    4.0, 9.65
)

# b1 -> ADD
add_arrow(
    1.82, 7.85,
    3.18, 7.85
)

# 中间节点 -> ADD
add_arrow(
    4.0, 9.05,
    4.0, 8.53
)

# ADD -> 后面的节点
add_arrow(
    4.82, 7.98,
    5.73, 7.98
)

# 后面的节点 -> sigma
add_arrow(
    6.37, 7.98,
    7.98, 7.98
)

# ============================================================
# 6. 第一层输出 -> 第二层输入
# ============================================================

# 第二层顶部输入节点
add_circle(
    4.0, 6.25,
    radius=0.30,
    text="",
    fontsize=12
)

# 第一层 sigma -> 第二层输入
add_arrow(
    8.0, 7.45,
    4.25, 6.42,
    linewidth=2.5
)

# ============================================================
# 7. 第二层大框
# ============================================================

ax.add_patch(
    Rectangle(
        (0.4, 1.35),
        13.0,
        4.35,
        facecolor="white",
        edgecolor=red,
        linewidth=4,
        zorder=0
    )
)

# Second Layer 标题
ax.add_patch(
    Rectangle(
        (7.0, 4.72),
        6.4, 1.0,
        facecolor=red,
        edgecolor=red,
        linewidth=0
    )
)

ax.text(
    10.2, 5.22,
    "Second Layer",
    ha="center",
    va="center",
    fontsize=28,
    color="white",
    fontweight="bold"
)

# ============================================================
# 8. 第二层节点
# ============================================================

# W2
add_circle(
    1.2, 4.55,
    radius=0.62,
    text=r"$W_2$",
    facecolor=light_blue,
    edgecolor=blue_edge,
    fontsize=22
)

# MM
add_box(
    3.2, 3.95,
    1.6, 1.05,
    r"$MM$",
    fontsize=21
)

# MM 后面的节点
add_circle(
    4.0, 3.35,
    radius=0.28,
    text=""
)

# b2
add_circle(
    1.2, 2.25,
    radius=0.62,
    text=r"$b_2$",
    facecolor=light_blue,
    edgecolor=blue_edge,
    fontsize=22
)

# ADD
add_box(
    3.2, 1.85,
    1.6, 1.05,
    r"$ADD$",
    fontsize=21
)

# ADD 后节点
add_circle(
    6.05, 2.38,
    radius=0.30,
    text=""
)

# sigma
add_box(
    8.0, 1.85,
    1.6, 1.05,
    r"$\sigma$",
    fontsize=23
)

# Nonlinear Function
ax.text(
    9.85, 2.38,
    "Nonlinear\nFunction",
    ha="left",
    va="center",
    fontsize=23,
    color="#333333"
)

# ============================================================
# 9. 第二层箭头
# ============================================================

# 第二层输入 -> MM
add_arrow(
    4.0, 5.95,
    4.0, 5.05
)

# W2 -> MM
add_arrow(
    1.82, 4.55,
    3.18, 4.55
)

# MM -> 中间节点
add_arrow(
    4.0, 3.95,
    4.0, 3.63
)

# b2 -> ADD
add_arrow(
    1.82, 2.25,
    3.18, 2.25
)

# 中间节点 -> ADD
add_arrow(
    4.0, 3.07,
    4.0, 2.93
)

# ADD -> 后面的节点
add_arrow(
    4.82, 2.38,
    5.73, 2.38
)

# 后面的节点 -> sigma
add_arrow(
    6.37, 2.38,
    7.98, 2.38
)

# ============================================================
# 10. 第二层输出 y_hat
# ============================================================

add_circle(
    4.0, 0.55,
    radius=0.62,
    text=r"$\hat{y}$",
    fontsize=23
)

# sigma -> y_hat
add_arrow(
    8.0, 1.85,
    4.35, 0.85,
    linewidth=2.5
)

# ============================================================
# 12. 顶部红色线
# ============================================================

ax.plot(
    [0, 14],
    [12.98, 12.98],
    color=red,
    linewidth=5
)

# ============================================================
# 13. 保存 / 显示
# ============================================================

plt.tight_layout()

# 保存为高清图片
plt.savefig(
    "neural_network_structure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()