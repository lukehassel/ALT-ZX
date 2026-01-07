import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_box(ax, xy, width, height, text, color='#E0E0E0', edgecolor='black', fontsize=9):
    rect = patches.FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.08", 
                                  linewidth=1.5, edgecolor=edgecolor, facecolor=color)
    ax.add_patch(rect)
    ax.text(xy[0] + width/2, xy[1] + height/2, text, 
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, fontweight='bold')
    return rect


def draw_arrow(ax, start, end, style='->', color='black'):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, lw=1.5, color=color))


def main():
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Colors
    input_color = '#D5E8D4'
    input_edge = '#82B366'
    encoder_color = '#DAE8FC'
    encoder_edge = '#6C8EBF'
    latent_color = '#E1D5E7'
    latent_edge = '#9673A6'
    decoder_color = '#FFF2CC'
    decoder_edge = '#D6B656'
    output_color = '#F8CECC'
    output_edge = '#B85450'
    loss_color = '#FFE6CC'
    loss_edge = '#D79B00'
    
    # Y positions
    y_main = 5.5
    y_loss = 1.5
    
    # Box dimensions
    box_w = 2.0
    box_h = 1.4
    small_w = 1.8
    small_h = 1.0
    
    # Title
    ax.text(10, 10.0, "GenZX: Graph Variational Autoencoder", ha='center', fontsize=16, fontweight='bold')
    
    # ===== ENCODER SECTION =====
    x = 1.0
    draw_box(ax, (x, y_main), box_w, box_h, "ZX Graph\n(A, X)", color=input_color, edgecolor=input_edge)
    ax.text(x + box_w/2, y_main + box_h + 0.3, "Input", ha='center', fontsize=10, fontweight='bold')
    
    x += box_w + 0.5
    draw_arrow(ax, (x - 0.5, y_main + box_h/2), (x, y_main + box_h/2))
    draw_box(ax, (x, y_main), box_w, box_h, "GNN\nEncoder", color=encoder_color, edgecolor=encoder_edge)
    
    x += box_w + 0.5
    draw_arrow(ax, (x - 0.5, y_main + box_h/2), (x, y_main + box_h/2))
    
    # Latent space box with mu and logvar
    latent_w = 2.2
    draw_box(ax, (x, y_main + 0.4), small_w, small_h - 0.2, "$\\mu$", color=latent_color, edgecolor=latent_edge)
    draw_box(ax, (x, y_main - 0.4), small_w, small_h - 0.2, "$\\log\\sigma^2$", color=latent_color, edgecolor=latent_edge)
    ax.text(x + small_w/2, y_main + box_h + 0.3, "Latent (1024d)", ha='center', fontsize=10, fontweight='bold')
    
    # Reparametrization
    x += small_w + 0.3
    ax.text(x + 0.6, y_main + box_h/2, "$z = \\mu + \\sigma \\cdot \\epsilon$", ha='center', fontsize=9)
    
    x += 1.5
    draw_arrow(ax, (x - 0.8, y_main + box_h/2), (x, y_main + box_h/2))
    draw_box(ax, (x, y_main), small_w, box_h, "$z$\nLatent", color=latent_color, edgecolor=latent_edge)
    
    # ===== DECODER SECTION =====
    x += small_w + 0.5
    draw_arrow(ax, (x - 0.5, y_main + box_h/2), (x, y_main + box_h/2))
    draw_box(ax, (x, y_main), box_w, box_h, "MLP\nDecoder", color=decoder_color, edgecolor=decoder_edge)
    
    x += box_w + 0.5
    draw_arrow(ax, (x - 0.5, y_main + box_h/2), (x, y_main + box_h/2))
    draw_box(ax, (x, y_main), box_w, box_h, "Recon Adj\n$\\hat{A}$", color=output_color, edgecolor=output_edge)
    ax.text(x + box_w/2, y_main + box_h + 0.3, "Output", ha='center', fontsize=10, fontweight='bold')
    
    # ===== LOSS COMPONENTS =====
    loss_y = y_loss
    loss_x_start = 2.5
    loss_spacing = 3.0
    
    ax.text(10, loss_y + small_h + 0.6, "Loss Components", ha='center', fontsize=12, fontweight='bold')
    
    losses = [
        ("Recon\nBCE", loss_color),
        ("KL\nDiv", loss_color),
        ("Gflow\nScore", '#D5E8D4'),
        ("ZXNet\nSemantic", '#DAE8FC'),
        ("Boundary\nDegree", loss_color),
    ]
    
    for i, (name, color) in enumerate(losses):
        lx = loss_x_start + i * loss_spacing
        draw_box(ax, (lx, loss_y), small_w, small_h, name, color=color, edgecolor=loss_edge)
    
    # Total loss
    total_x = loss_x_start + 2 * loss_spacing
    draw_box(ax, (total_x - 0.5, loss_y - 1.3), 2.8, 0.9, "Total Loss = Σ weighted", color='#FFFFFF', edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('GenZX/genzx_architecture.png', dpi=300, bbox_inches='tight')
    print("Saved to GenZX/genzx_architecture.png")


if __name__ == "__main__":
    main()
