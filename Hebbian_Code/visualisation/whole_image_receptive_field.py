import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
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
    # Create a key for the cache based on dataloader properties
    cache_key = f"{id(dataloader)}_{max_batch_images}"

    if cache_key not in _fixed_images_cache:
        # If not in cache, collect the images
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


def find_receptive_field_activations(model, dataloader, layer, num_filters=25, max_batch_images=5):
    """
    Find which parts of input images maximally activate specific neurons,
    considering their receptive fields.
    """
    model.eval()
    device = next(model.parameters()).device

    # Get fixed images (from cache if available)
    fixed_images = get_fixed_images(dataloader, max_batch_images)
    fixed_images = fixed_images.to(device)

    rf_size, stride = calculate_receptive_field(model, layer)
    print(f"Receptive field size: {rf_size}x{rf_size}, Stride: {stride}")

    if hasattr(layer, 'out_channels'):
        out_channels = layer.out_channels
    else:
        raise ValueError("The target layer does not have an 'out_channels' attribute.")

    num_filters = min(num_filters, out_channels)

    best_activations = torch.full((num_filters,), float('-inf'), device=device)
    best_images = []
    best_positions = []
    full_images = []
    image_indices = []  # Track which image was used for each filter

    with torch.no_grad():
        # Process all fixed images at once
        activations = get_layer_output(model, fixed_images, layer)

        for filter_idx in range(num_filters):
            filter_activations = activations[:, filter_idx]

            # Find maximum activation across all spatial positions and images
            max_vals, _ = filter_activations.view(len(fixed_images), -1).max(dim=1)
            batch_max, batch_idx = max_vals.max(dim=0)

            # Find spatial position of maximum activation
            spatial_idx = filter_activations[batch_idx].flatten().argmax()
            h_idx, w_idx = spatial_idx // filter_activations.shape[2], spatial_idx % filter_activations.shape[2]

            # Calculate center position in input image
            input_h = h_idx * stride
            input_w = w_idx * stride

            # Calculate receptive field boundaries
            h_start = max(0, input_h - rf_size // 2)
            w_start = max(0, input_w - rf_size // 2)
            h_end = min(fixed_images.shape[2], h_start + rf_size)
            w_end = min(fixed_images.shape[3], w_start + rf_size)

            # Store information
            best_activations[filter_idx] = batch_max.cpu().item()
            full_image = fixed_images[batch_idx].cpu()
            image_indices.append(batch_idx.item())

            best_images.append((int(h_start), int(h_end), int(w_start), int(w_end)))
            best_positions.append((int(input_h), int(input_w)))
            full_images.append(full_image)

            print(f"Filter {filter_idx} - Best activation: {batch_max:.4f} from image {batch_idx}")

    return full_images, best_images, best_positions, best_activations, rf_size, image_indices


def visualize_receptive_fields_context(model, dataloader, layer, num_filters=25, max_batch_images=5):
    """Visualize the full images with highlighted receptive fields"""
    full_images, positions, centers, activations, rf_size, image_indices = find_receptive_field_activations(
        model, dataloader, layer, num_filters, max_batch_images
    )

    # Create visualization grid
    grid_size = int(num_filters ** 0.5) + (1 if num_filters ** 0.5 % 1 > 0 else 0)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(num_filters):
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

        # Add center point
        center_h, center_w = centers[i]
        axes[i].plot(center_w, center_h, 'r+', markersize=10)

        # Add image index to the title
        axes[i].set_title(f'Filter {i}\nImage {image_indices[i]}\nAct: {activations[i]:.2f}')
        axes[i].axis('off')

    # Turn off unused subplots
    for j in range(num_filters, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Receptive Field Size: {rf_size}x{rf_size}', y=0.95)
    plt.tight_layout()
    wandb.log({"Receptive Fields with Context": wandb.Image(fig)})
    plt.close(fig)