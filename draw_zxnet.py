import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_block(ax, xy, width, height, label, color='lightblue', fontsize=10):
    rect = patches.Rectangle(xy, width, height, linewidth=1.5, edgecolor='black', facecolor=color, zorder=2)
    ax.add_patch(rect)
    cx = xy[0] + width / 2
    cy = xy[1] + height / 2
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=3)
    return rect


def draw_arrow(ax, start, end, label=""):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=2, color='black'), zorder=1)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x - 0.2, mid_y, label, ha='right', va='center', fontsize=8)


def draw_zxnet_architecture():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title("ZXNet Siamese Architecture", fontsize=16, fontweight='bold', pad=20)

    draw_block(ax, (1, 10), 3, 1, "Graph 1\n(N x 6 Features)", color='#ffcccc')
    
    draw_arrow(ax, (2.5, 10), (2.5, 9))
    draw_block(ax, (1, 8), 3, 1, "GCNConv1 + ReLU\n(Output: N x 64)", color='#ccffcc')
    
    draw_arrow(ax, (2.5, 8), (2.5, 7))
    draw_block(ax, (1, 6), 3, 1, "GCNConv2 + ReLU\n(Output: N x 64)", color='#ccffcc')
    
    draw_arrow(ax, (2.5, 6), (2.5, 5))
    draw_block(ax, (1, 4), 3, 1, "Global Mean Pool\n(Output: 1 x 64)", color='#ccccff')

    draw_arrow(ax, (2.5, 4), (4.5, 3))

    draw_block(ax, (6, 10), 3, 1, "Graph 2\n(N x 6 Features)", color='#ffcccc') 
    
    draw_arrow(ax, (7.5, 10), (7.5, 9))
    draw_block(ax, (6, 8), 3, 1, "GCNConv1 + ReLU\n(Output: N x 64)", color='#ccffcc') 
    
    draw_arrow(ax, (7.5, 8), (7.5, 7))
    draw_block(ax, (6, 6), 3, 1, "GCNConv2 + ReLU\n(Output: N x 64)", color='#ccffcc')
    
    draw_arrow(ax, (7.5, 6), (7.5, 5))
    draw_block(ax, (6, 4), 3, 1, "Global Mean Pool\n(Output: 1 x 64)", color='#ccccff')

    draw_arrow(ax, (7.5, 4), (5.5, 3))

    draw_block(ax, (4, 2), 2, 1, "Concatenate\n(1 x 128)", color='#ffffcc')
    
    draw_arrow(ax, (5, 2), (5, 1.5))
    draw_block(ax, (3.5, 0.5), 3, 1, "FC Layers (MLP)\n-> Logits [Not Eq, Eq]", color='#ffd9cc')

    ax.text(5, 7, "Shared Weights", ha='center', va='center', fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    output_path = "zxnet_architecture.png"
    plt.savefig(output_path, dpi=150)
    print(f"Architecture diagram saved to {output_path}")


if __name__ == "__main__":
    draw_zxnet_architecture()
