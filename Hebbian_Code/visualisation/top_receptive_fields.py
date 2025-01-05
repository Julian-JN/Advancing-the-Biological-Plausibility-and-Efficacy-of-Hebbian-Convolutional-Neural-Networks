import matplotlib.pyplot as plt
import torch
from torch import nn
from hebbian_layers.hebb import HebbianConv2d
from hebbian_layers.hebb_depthwise import HebbianDepthConv2d
import wandb
import matplotlib.patches as patches

# Global cache for fixed images
_fixed_images_cache = {}


def calculate_receptive_field(model, target_layer):
    """Calculate the receptive field size for a given layer"""
    current_rf = 1
    current_stride = 1

    for layer in model.children():
        if isinstance(layer, (nn.Conv2d, HebbianConv2d, HebbianDepthConv2d)):
            kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
            stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
            current_rf += (kernel_size - 1) * current_stride
            current_stride *= stride
        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            kernel_size = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            stride = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            current_rf += (kernel_size - 1) * current_stride
            current_stride *= stride

        if layer == target_layer:
            break

    return current_rf, current_stride


def get_layer_output(model, x, target_layer):
    """Forward pass through the model up to target layer"""
    for layer in model.children():
        x = layer(x)
        if layer == target_layer:
            return x
    raise ValueError(f"Target layer {target_layer} not found in the model.")


def get_fixed_images(dataloader, max_batch_images):
    """Get or create fixed set of images, using cache to ensure consistency"""
    cache_key = f"{id(dataloader)}_{max_batch_images}"

    if cache_key not in _fixed_images_cache:
        for batch, _ in dataloader:
            fixed_images = batch[:min(len(batch), max_batch_images)]
            _fixed_images_cache[cache_key] = fixed_images
            print(f"Collected {len(fixed_images)} new images for analysis")
            break
    else:
        print(f"Using {len(_fixed_images_cache[cache_key])} cached images")

    return _fixed_images_cache[cache_key]


def clear_image_cache():
    """Clear the cached images if needed"""
    global _fixed_images_cache
    _fixed_images_cache.clear()
    print("Image cache cleared")


def find_top_activations(model, dataloader, layer, num_top=25, max_batch_images=5):
    """
    Find the neurons with highest activations across all filters and their receptive fields.
    """
    model.eval()
    device = next(model.parameters()).device

    # Get fixed images (from cache if available)
    fixed_images = get_fixed_images(dataloader, max_batch_images)
    fixed_images = fixed_images.to(device)

    rf_size, stride = calculate_receptive_field(model, layer)
    print(f"Receptive field size: {rf_size}x{rf_size}, Stride: {stride}")

    if hasattr(layer, 'out_channels'):
        num_filters = layer.out_channels
    else:
        raise ValueError("The target layer does not have an 'out_channels' attribute.")

    with torch.no_grad():
        # Process all fixed images at once
        activations = get_layer_output(model, fixed_images, layer)

        # Reshape activations to find global maximum
        batch_size, channels, height, width = activations.shape
        activations_flat = activations.view(batch_size, channels, -1)  # [B, C, H*W]
        max_vals, spatial_indices = activations_flat.max(dim=2)  # Max across spatial dimensions
        max_vals_per_filter, batch_indices = max_vals.max(dim=0)  # Max across batch

        # Get top k filters by activation value
        top_k_values, top_k_filters = max_vals_per_filter.topk(num_top)

        # Store results
        best_images = []
        best_positions = []
        full_images = []
        filter_indices = []
        image_indices = []
        best_activations = []

        for idx, (filter_idx, batch_idx) in enumerate(zip(top_k_filters, batch_indices[top_k_filters])):
            # Get spatial position
            spatial_idx = spatial_indices[batch_idx, filter_idx]
            h_idx, w_idx = spatial_idx // width, spatial_idx % width

            # Calculate center position in input image
            input_h = h_idx * stride
            input_w = w_idx * stride

            # Calculate receptive field boundaries
            h_start = max(0, input_h - rf_size // 2)
            w_start = max(0, input_w - rf_size // 2)
            h_end = min(fixed_images.shape[2], h_start + rf_size)
            w_end = min(fixed_images.shape[3], w_start + rf_size)

            # Store information
            best_activations.append(top_k_values[idx].cpu().item())
            full_images.append(fixed_images[batch_idx].cpu())
            best_positions.append((int(h_start), int(h_end), int(w_start), int(w_end)))
            filter_indices.append(filter_idx.item())
            image_indices.append(batch_idx.item())

            print(
                f"Top {idx + 1}: Filter {filter_idx.item()} - Activation: {top_k_values[idx]:.4f} from image {batch_idx.item()}")

    return full_images, best_positions, best_activations, rf_size, filter_indices, image_indices


def visualize_top_activations(model, dataloader, layer, num_top=25, max_batch_images=5):
    """Visualize the receptive fields of neurons with highest activations"""
    full_images, positions, activations, rf_size, filter_indices, image_indices = find_top_activations(
        model, dataloader, layer, num_top, max_batch_images
    )

    # Create visualization grid
    grid_size = int(num_top ** 0.5) + (1 if num_top ** 0.5 % 1 > 0 else 0)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(num_top):
        # Display full image
        img = full_images[i].permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        img = img.numpy()
        axes[i].imshow(img)

        # Add receptive field rectangle
        h_start, h_end, w_start, w_end = positions[i]
        rect = patches.Rectangle(
            (w_start, h_start),
            w_end - w_start,
            h_end - h_start,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        axes[i].add_patch(rect)

        # Add title with rank, filter number, and activation
        axes[i].set_title(
            f'Rank {i + 1}\nFilter {filter_indices[i]}\nImage {image_indices[i]}\nAct: {activations[i]:.2f}')
        axes[i].axis('off')

    # Turn off unused subplots
    for j in range(num_top, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Top {num_top} Activations (RF Size: {rf_size}x{rf_size})', y=0.95)
    plt.tight_layout()
    wandb.log({"Top Activations": wandb.Image(fig)})
    plt.close(fig)