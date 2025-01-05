import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects


def create_compact_network_visualization(latex_width_fraction=1.0):
    """
    Creates a visualization of a neural network architecture with compact vertical layers
    and horizontal flow, matching the original architecture specifications.
    """
    # Calculate figure dimensions
    latex_textwidth_inches = 6.3
    fig_width = latex_textwidth_inches * latex_width_fraction
    fig_height = fig_width * 0.4

    # Font sizes and layout parameters
    base_fontsize = 7
    layer_name_fontsize = base_fontsize
    block_label_fontsize = base_fontsize - 1
    bracket_label_fontsize = base_fontsize + 1

    # Set up the figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Color scheme - using original colors
    colors = {
        'conv': '#99ff99',
        'pool': '#cce6ff',
        'activ': '#ffeb99',
        'bn': '#e6e6e6',
        'fc': '#ffcccc',
        'input': '#ffcc66'
    }

    def draw_arrow(start_x, start_y, end_x, end_y):
        """Draws a connecting arrow between components."""
        ax.annotate('',
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='-|>',
                                    lw=0.5,
                                    color='black',
                                    shrinkA=0,
                                    shrinkB=0),
                    zorder=3)

    def add_group_bracket(x_start, x_end, label, color='#0d6efd'):
        """Adds a bracketing annotation above a group of layers."""
        y = 7.5
        plt.hlines(y, x_start, x_end, color=color, lw=1.2)
        plt.vlines([x_start, x_end], y - 0.2, y, color=color, lw=1.2)

        txt = plt.text((x_start + x_end) / 2, y + 0.1, label,
                       ha='center', va='bottom',
                       color=color,
                       fontsize=bracket_label_fontsize,
                       fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='white')])

    # Network architecture definition - using original architecture
    layers = [
        {
            'name': 'Layer 1',
            'blocks': [
                {'name': 'Input', 'color': colors['input']},
                {'name': 'Batch\nNorm', 'color': colors['bn']},
                {'name': '5×5\nConv. 96', 'color': colors['conv']},
                {'name': 'MaxPool\n4×4', 'color': colors['pool']},
                {'name': 'Triangle\n0.7', 'color': colors['activ']}
            ],
            'group': 'Hebb.'
        },
        {
            'name': 'Layer 2',
            'blocks': [
                {'name': 'Batch\nNorm', 'color': colors['bn']},
                {'name': '3×3\nConv. 384', 'color': colors['conv']},
                {'name': 'MaxPool\n4×4', 'color': colors['pool']},
                {'name': 'Triangle\n1.4', 'color': colors['activ']}
            ],
            'group': 'Hebb.'
        },
        {
            'name': 'Layer 3',
            'blocks': [
                {'name': 'Batch\nNorm', 'color': colors['bn']},
                {'name': '3×3\nConv. 1536', 'color': colors['conv']},
                {'name': 'AvgPool\n2×2', 'color': colors['pool']},
                {'name': 'Triangle\n1.0', 'color': colors['activ']}
            ],
            'group': 'Hebb.'
        },
        {
            'name': 'Layer 4',
            'blocks': [
                {'name': 'Flatten', 'color': colors['fc']},
                {'name': 'Dropout\n0.5', 'color': colors['fc']},
                {'name': 'FC\n10', 'color': colors['fc']}
            ],
            'group': 'SGD'
        }
    ]

    # Layout parameters
    block_width = 0.8
    block_height = 0.6
    block_spacing = 0.1
    layer_spacing = 0.9  # Reduced from 2.0 to 0.9 to make layers closer together
    current_x = 1.0

    # Track group boundaries
    current_group = None
    group_start_x = current_x

    # Draw each layer
    for i, layer in enumerate(layers):
        # Calculate vertical positioning
        total_blocks = len(layer['blocks'])
        layer_height = (total_blocks * (block_height + block_spacing)) - block_spacing
        start_y = 4.5 - (layer_height / 2)  # Center vertically

        # Add layer name
        plt.text(current_x + block_width / 2, 2.0,
                 layer['name'],
                 ha='center', va='bottom',
                 fontsize=layer_name_fontsize,
                 rotation=0)

        # Draw blocks in the layer
        current_y = start_y
        for j, block in enumerate(layer['blocks']):
            # Draw block
            rect = patches.Rectangle(
                (current_x, current_y),
                block_width,
                block_height,
                facecolor=block['color'],
                edgecolor='black',
                linewidth=0.5,
                zorder=2
            )
            ax.add_patch(rect)

            # Add block label
            txt = plt.text(current_x + block_width / 2, current_y + block_height / 2,
                           block['name'],
                           ha='center', va='center',
                           fontsize=block_label_fontsize)

            # Draw arrow to next block within layer
            if j < len(layer['blocks']) - 1:
                draw_arrow(current_x + block_width / 2,
                           current_y + block_height,
                           current_x + block_width / 2,
                           current_y + block_height + block_spacing)

            current_y += block_height + block_spacing

        # Handle group brackets
        if layer['group'] != current_group:
            if current_group is not None:
                bracket_color = '#0d6efd' if current_group == 'Hebb.' else '#dc3545'
                add_group_bracket(group_start_x - 0.2,
                                  current_x + block_width + 0.2,
                                  current_group,
                                  bracket_color)
            current_group = layer['group']
            group_start_x = current_x

        # Draw arrow to next layer
        if i < len(layers) - 1:
            draw_arrow(current_x + block_width,
                       4.5,  # Middle height
                       current_x + layer_spacing,
                       4.5)

        current_x += layer_spacing

    # Add final group bracket
    bracket_color = '#0d6efd' if current_group == 'Hebb.' else '#dc3545'
    add_group_bracket(group_start_x - 0.2,
                      current_x - layer_spacing + block_width + 0.2,
                      current_group,
                      bracket_color)

    # Adjust plot limits and appearance
    plt.ylim(1.5, 8.0)
    plt.xlim(0.5, current_x - layer_spacing + block_width + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ensure tight layout
    plt.tight_layout(pad=0.1)
    return fig


# Create and save the visualization
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

fig = create_compact_network_visualization(latex_width_fraction=1.0)
plt.savefig('network_architecture_compact.png', bbox_inches='tight', dpi=300)
plt.close()