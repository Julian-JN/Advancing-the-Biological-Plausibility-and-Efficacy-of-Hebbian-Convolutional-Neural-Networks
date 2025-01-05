import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches


def create_simple_filter_viz(mode='hard', temperature=0.5, spatial_loc=(2, 2)):
    """
    Create visualization of 5x5x1 filters with competition through one spatial location.
    Filter 1 is positioned front-facing for better visibility.
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(15, 10), facecolor='none')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('none')

    # Parameters
    n_filters = 4
    height = width = 5
    depth = 1
    spacing = 8

    # Reverse the order of filters so Filter 1 is in front
    def get_filter_position(filter_idx):
        """Get the position for a filter, with higher indices (back filters) placed first"""
        return (n_filters - 1 - filter_idx) * (depth + spacing)

    def create_cube(pos, size=1, color='blue', alpha=0.2, highlighted=False):
        """Create a single cube at the specified position"""
        x, y, z = pos
        vertices = [
            [[x, y, z], [x + size, y, z], [x + size, y + size, z], [x, y + size, z]],
            [[x, y, z + size], [x + size, y, z + size], [x + size, y + size, z + size], [x, y + size, z + size]],
            [[x, y, z], [x + size, y, z], [x + size, y, z + size], [x, y, z + size]],
            [[x, y + size, z], [x + size, y + size, z], [x + size, y + size, z + size], [x, y + size, z + size]],
            [[x, y, z], [x, y + size, z], [x, y + size, z + size], [x, y, z + size]],
            [[x + size, y, z], [x + size, y + size, z], [x + size, y + size, z + size], [x + size, y, z + size]]
        ]
        cube = Poly3DCollection(vertices)
        cube.set_facecolor(color)
        cube.set_edgecolor('black' if highlighted else 'gray')
        cube.set_linewidth(2 if highlighted else 0.5)
        cube.set_alpha(0.9 if highlighted else alpha)
        ax.add_collection3d(cube)

    def add_text_with_line(x, y, z, text, dx=0, dy=0, dz=1.5, fontsize=14, offset_text=False):
        """Add text with a thin line connecting it to a point"""
        if offset_text:
            dy += 1
            dz += 1.5

        ax.plot([x, x + dx], [y, y + dy], [z, z + dz],
                color='black', linewidth=0.5, linestyle=':', zorder=10)

        text_obj = ax.text(x + dx, y + dy, z + dz, text,
                           horizontalalignment='center',
                           verticalalignment='bottom',
                           fontsize=fontsize,
                           fontweight='bold',
                           zorder=11,
                           bbox=dict(facecolor='white',
                                     edgecolor='none',
                                     alpha=0.9,
                                     pad=2))
        text_obj.set_path_effects([
            path_effects.withStroke(linewidth=2, foreground='white')
        ])

    def create_filter(pos, color, activation, softmax_value=None):
        """Create a 5x5x1 filter with highlighted pixel"""
        x_start, y_start, z_start = pos
        for h in range(height):
            for w in range(width):
                is_highlight = (h, w) == spatial_loc
                create_cube([x_start, y_start + w, z_start + h],
                            color=color,
                            highlighted=is_highlight)

                if is_highlight:
                    if softmax_value is not None:
                        text = f'Activation: {activation:.2f}\nSoftmax: {softmax_value:.2f}'
                    else:
                        text = f'Activation: {activation:.2f}'
                    add_text_with_line(x_start + 0.5, y_start + w + 0.5, z_start + h + 1,
                                       text,
                                       fontsize=14,
                                       offset_text=True)

    def draw_competition(pos1, pos2, line_weight=1.0):
        """Draw competition line between highlighted pixels"""
        if mode == 'hard':
            color = 'red'
            alpha = 0.8
            width = 5
        else:
            color = 'purple'
            alpha = min(0.9, line_weight + 0.3)
            width = max(3, 7 * line_weight)

        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                color=color, linewidth=width, alpha=alpha,
                marker='>', markersize=15 if mode == 'hard' else 12 * line_weight,
                markevery=[-1],
                zorder=9)

    # Sample activations - reversed order
    activations = [0.31, 0.63, 0.45, 0.82]

    if mode == 'soft':
        exp_acts = np.exp(np.array(activations) / temperature)
        softmax_values = exp_acts / exp_acts.sum()
    else:
        winner_idx = np.argmax(activations)
        softmax_values = np.zeros_like(activations)
        softmax_values[winner_idx] = 1.0

    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFCC99'][::-1]

    # Draw filters in reverse order (back to front)
    for i in range(n_filters - 1, -1, -1):
        x_pos = get_filter_position(i)
        create_filter([x_pos, 0, 0], colors[i], activations[i],
                      softmax_values[i] if mode == 'soft' else None)

        add_text_with_line(x_pos + depth / 2, width / 2, -0.5,
                           f'Filter {n_filters - i}',
                           dz=-1, fontsize=14)

        if mode == 'hard' and i == np.argmax(activations):
            add_text_with_line(x_pos + depth / 2, width / 2, height + 0.5,
                               "WINNER",
                               dz=1.5, fontsize=16)

    # Draw competition lines in reverse order
    winner_idx = np.argmax(activations)
    for i in range(n_filters - 2, -1, -1):
        pos1 = [get_filter_position(i + 1) + 0.5,
                spatial_loc[1] + 0.5,
                spatial_loc[0] + 0.5]
        pos2 = [get_filter_position(i) + 0.5,
                spatial_loc[1] + 0.5,
                spatial_loc[0] + 0.5]

        if mode == 'hard':
            is_winner_connection = (i == winner_idx or i + 1 == winner_idx)
            draw_competition(pos1, pos2, 1.0 if is_winner_connection else 1.0)
        else:
            line_weight = (softmax_values[i] + softmax_values[i + 1]) / 2
            draw_competition(pos1, pos2, line_weight)

    # Title
    title = f"{'Hard' if mode == 'hard' else 'Soft'} Winner-Take-All"
    if mode == 'soft':
        title += f" (temperature={temperature})"
    subtitle = f"Competition at Spatial Location (h={spatial_loc[0]}, w={spatial_loc[1]})"
    if mode == 'soft':
        subtitle += "\nSoftmax values show relative strength of each filter's output"

    title_obj = ax.text(2 * (depth + spacing), -2, height + 4,
                        f"{title}\n{subtitle}",
                        color='black', fontsize=16, fontweight='bold',
                        horizontalalignment='center',
                        zorder=11,
                        bbox=dict(facecolor='white',
                                  edgecolor='none',
                                  alpha=0.9,
                                  pad=2))
    title_obj.set_path_effects([
        path_effects.withStroke(linewidth=2, foreground='white')
    ])

    # Legend
    legend_elements = [
        patches.Patch(color=colors[0], alpha=0.2, label='Filter Pixels (5x5)'),
        patches.Patch(color=colors[0], alpha=0.9, label='Competing Pixel'),
        patches.Patch(color='purple' if mode == 'soft' else 'red', alpha=0.8,
                      label='Competition Strength' if mode == 'soft' else 'Hard Competition'),
    ]
    if mode == 'hard':
        legend_elements.append(patches.Patch(color='gold', alpha=0.8, label='Winner'))

    legend = ax.legend(handles=legend_elements, loc='upper right',
                       bbox_to_anchor=(1.1, 1), fontsize=14)
    legend.set_zorder(12)

    # Optimize view for front-facing Filter 1
    ax.view_init(elev=20, azim=115)

    # Axis labels
    ax.set_xlabel('X (Depth)', labelpad=5, fontsize=14)
    ax.set_ylabel('Y (Width)', labelpad=5, fontsize=14)
    ax.set_zlabel('Z (Height)', labelpad=5, fontsize=14)

    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Axis limits
    max_x = (n_filters - 1) * (depth + spacing) + depth + 1
    ax.set_xlim(-1, max_x)
    ax.set_ylim(-2, width + 3)
    ax.set_zlim(-1, height + 4)

    # Remove grid and background
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    # Tight layout
    plt.tight_layout(pad=0.1)
    return fig


# Create and save both visualizations
fig_hard = create_simple_filter_viz(mode='hard')
plt.show()
plt.savefig('hard-wta.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

fig_soft = create_simple_filter_viz(mode='soft', temperature=0.5)
plt.show()
plt.savefig('soft-wta.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()