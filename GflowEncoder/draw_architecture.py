import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines


def draw_box(ax, xy, width, height, text, color='#E0E0E0', edgecolor='black', fontsize=10):
    rect = patches.FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.1", 
                                  linewidth=1.5, edgecolor=edgecolor, facecolor=color)
    ax.add_patch(rect)
    ax.text(xy[0] + width/2, xy[1] + height/2, text, 
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, fontweight='bold')
    return rect


def draw_arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))


def main():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    y_anchor = 6
    y_pos = 4
    y_neg = 2
    
    input_x = 1
    encoder_x = 5
    embed_x = 9
    
    box_w = 2.5
    box_h = 1.2
    
    # Inputs
    draw_box(ax, (input_x, y_anchor), box_w, box_h, "Anchor\n(Valid Graph)", color='#D5E8D4', edgecolor='#82B366')
    draw_box(ax, (input_x, y_pos), box_w, box_h, "Positive\n(Valid Graph)", color='#D5E8D4', edgecolor='#82B366')
    draw_box(ax, (input_x, y_neg), box_w, box_h, "Negative\n(Invalid, Hard)", color='#F8CECC', edgecolor='#B85450')
    ax.text(input_x + box_w/2, 7.5, "Input Triplets", ha='center', fontsize=12, fontweight='bold')

    # Shared encoder box
    enc_color = '#DAE8FC'
    enc_edge = '#6C8EBF'
    shared_rect = patches.Rectangle((encoder_x - 0.2, 1.5), box_w + 0.4, 6.0, 
                                    linewidth=1.5, edgecolor='#6C8EBF', facecolor='none', linestyle='--')
    ax.add_patch(shared_rect)
    ax.text(encoder_x + box_w/2, 7.7, "Shared GCN Encoder", ha='center', fontsize=12, fontweight='bold', color='#6C8EBF')
    
    draw_box(ax, (encoder_x, y_anchor), box_w, box_h, "GCN Encoder", color=enc_color, edgecolor=enc_edge)
    draw_box(ax, (encoder_x, y_pos), box_w, box_h, "GCN Encoder", color=enc_color, edgecolor=enc_edge)
    draw_box(ax, (encoder_x, y_neg), box_w, box_h, "GCN Encoder", color=enc_color, edgecolor=enc_edge)

    # Embeddings
    emb_color = '#FFF2CC'
    emb_edge = '#D6B656'
    draw_box(ax, (embed_x, y_anchor), box_w, box_h, "Embedding\n$Z_A$", color=emb_color, edgecolor=emb_edge)
    draw_box(ax, (embed_x, y_pos), box_w, box_h, "Embedding\n$Z_P$", color=emb_color, edgecolor=emb_edge)
    draw_box(ax, (embed_x, y_neg), box_w, box_h, "Embedding\n$Z_N$", color=emb_color, edgecolor=emb_edge)
    ax.text(embed_x + box_w/2, 7.5, "Latent Space (64d)", ha='center', fontsize=12, fontweight='bold')

    # Arrows: Input -> Encoder
    draw_arrow(ax, (input_x + box_w, y_anchor + box_h/2), (encoder_x, y_anchor + box_h/2))
    draw_arrow(ax, (input_x + box_w, y_pos + box_h/2), (encoder_x, y_pos + box_h/2))
    draw_arrow(ax, (input_x + box_w, y_neg + box_h/2), (encoder_x, y_neg + box_h/2))
    
    # Arrows: Encoder -> Embedding
    draw_arrow(ax, (encoder_x + box_w, y_anchor + box_h/2), (embed_x, y_anchor + box_h/2))
    draw_arrow(ax, (encoder_x + box_w, y_pos + box_h/2), (embed_x, y_pos + box_h/2))
    draw_arrow(ax, (encoder_x + box_w, y_neg + box_h/2), (embed_x, y_neg + box_h/2))

    plt.tight_layout()
    plt.savefig('gflow_encoder_architecture.png', dpi=300, bbox_inches='tight')
    print("Diagram saved to gflow_encoder_architecture.png")


if __name__ == "__main__":
    main()
