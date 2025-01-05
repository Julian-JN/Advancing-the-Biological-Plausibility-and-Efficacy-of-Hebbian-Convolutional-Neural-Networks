import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
import wandb
import numpy as np
from hebbian_layers.hebb import HebbianConv2d
from hebbian_layers.hebb_depthwise import HebbianDepthConv2d


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


def visualize_spatial_importance_input_space(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layer: nn.Module,
        max_batch_images: int = 5,
        aggregation_method: str = 'mean'
) -> None:
    """
    Visualizes the spatial importance mapped back to input image space by
    upscaling the activation heatmap to match original image dimensions.

    Args:
        model: Neural network model
        dataloader: DataLoader containing images
        layer: Target layer to analyze
        max_batch_images: Maximum number of images to process
        aggregation_method: How to aggregate filter activations ('mean' or 'max')
    """
    model.eval()
    device = next(model.parameters()).device

    # Get fixed images
    fixed_images = get_fixed_images(dataloader, max_batch_images)
    fixed_images = fixed_images.to(device)

    with torch.no_grad():
        # Get layer activations
        activations = get_layer_output(model, fixed_images, layer)

        # Aggregate across all filters
        if aggregation_method == 'mean':
            importance_maps = activations.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        else:  # max
            importance_maps = activations.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]

        # Normalize each importance map to [0, 1]
        importance_maps = (importance_maps - importance_maps.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]) / \
                          (importance_maps.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] -
                           importance_maps.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0])

        # Upscale importance maps to match input image size
        importance_maps = F.interpolate(
            importance_maps,
            size=(fixed_images.shape[2], fixed_images.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        # Create visualization grid
        batch_size = len(fixed_images)
        grid_size = int(np.ceil(np.sqrt(batch_size * 2)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        axes = axes.flatten()

        for idx in range(batch_size):
            # Original image
            img = fixed_images[idx].cpu().permute(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())

            # Plot original image
            axes[idx * 2].imshow(img)
            axes[idx * 2].set_title(f'Original Image {idx}')
            axes[idx * 2].axis('off')

            # Plot heatmap overlaid on original image
            heatmap = importance_maps[idx, 0].cpu()

            # Create a blended visualization
            axes[idx * 2 + 1].imshow(img)
            im = axes[idx * 2 + 1].imshow(heatmap, cmap='hot', alpha=0.6)
            axes[idx * 2 + 1].set_title(f'Importance Overlay {idx}')
            axes[idx * 2 + 1].axis('off')
            plt.colorbar(im, ax=axes[idx * 2 + 1])

        # Turn off unused subplots
        for j in range(batch_size * 2, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Spatial Importance in Input Space ({aggregation_method} aggregation)')
        plt.tight_layout()
        wandb.log({"Input Space Importance": wandb.Image(fig)})
        plt.close(fig)

def visualize_spatial_importance(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layer: nn.Module,
        max_batch_images: int = 5,
        aggregation_method: str = 'mean'
) -> None:
    """
    Visualizes the spatial importance of different regions in the input images
    by aggregating activations across all filters in a layer.

    Args:
        model: Neural network model
        dataloader: DataLoader containing images
        layer: Target layer to analyze
        max_batch_images: Maximum number of images to process
        aggregation_method: How to aggregate filter activations ('mean' or 'max')
    """
    model.eval()
    device = next(model.parameters()).device

    # Use existing function to get fixed images
    fixed_images = get_fixed_images(dataloader, max_batch_images)
    fixed_images = fixed_images.to(device)

    with torch.no_grad():
        # Use existing function to get layer activations
        activations = get_layer_output(model, fixed_images, layer)

        # Aggregate across all filters
        if aggregation_method == 'mean':
            importance_maps = activations.mean(dim=1)  # [B, H, W]
        else:  # max
            importance_maps = activations.max(dim=1)[0]  # [B, H, W]

        # Normalize each importance map to [0, 1]
        importance_maps = (importance_maps - importance_maps.min(dim=1, keepdim=True)[0]) / \
                          (importance_maps.max(dim=1, keepdim=True)[0] - importance_maps.min(dim=1, keepdim=True)[0])

        # Create visualization grid
        batch_size = len(fixed_images)
        grid_size = int(np.ceil(np.sqrt(batch_size * 2)))  # Space for original + heatmap
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        axes = axes.flatten()

        for idx in range(batch_size):
            # Original image
            img = fixed_images[idx].cpu().permute(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            axes[idx * 2].imshow(img)
            axes[idx * 2].set_title(f'Original Image {idx}')
            axes[idx * 2].axis('off')

            # Importance heatmap
            im = axes[idx * 2 + 1].imshow(importance_maps[idx].cpu(), cmap='hot')
            axes[idx * 2 + 1].set_title(f'Activation Heatmap {idx}')
            axes[idx * 2 + 1].axis('off')
            plt.colorbar(im, ax=axes[idx * 2 + 1])

        # Turn off unused subplots
        for j in range(batch_size * 2, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Spatial Importance Analysis ({aggregation_method} aggregation)')
        plt.tight_layout()
        wandb.log({"Spatial Importance": wandb.Image(fig)})
        plt.close(fig)


def visualize_mapped_importance(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layer: nn.Module,
        max_batch_images: int = 5,
        aggregation_method: str = 'mean'
) -> None:
    """
    Creates a heatmap by mapping layer activations back to their original spatial
    positions in the input image space.
    """
    model.eval()
    device = next(model.parameters()).device

    # Get fixed images and calculate receptive field parameters
    fixed_images = get_fixed_images(dataloader, max_batch_images)
    fixed_images = fixed_images.to(device)
    rf_size, stride = calculate_receptive_field(model, layer)

    with torch.no_grad():
        # Get layer activations
        activations = get_layer_output(model, fixed_images, layer)
        batch_size, channels, height, width = activations.shape

        # Create importance maps at original image resolution
        importance_maps = torch.zeros(batch_size, fixed_images.shape[2], fixed_images.shape[3], device=device)
        contribution_counts = torch.zeros_like(importance_maps)

        # For each activation, map it back to input space
        for h in range(height):
            for w in range(width):
                # Calculate center position in input image
                input_h = h * stride
                input_w = w * stride

                # Calculate region boundaries
                h_start = max(0, input_h - rf_size // 2)
                h_end = min(fixed_images.shape[2], h_start + rf_size)
                w_start = max(0, input_w - rf_size // 2)
                w_end = min(fixed_images.shape[3], w_start + rf_size)

                # Get activation values for this position
                if aggregation_method == 'mean':
                    position_importance = activations[..., h, w].mean(dim=1)  # Average across channels
                else:  # max
                    position_importance = activations[..., h, w].max(dim=1)[0]  # Max across channels

                # Add to importance map
                for b in range(batch_size):
                    importance_maps[b, h_start:h_end, w_start:w_end] += position_importance[b]
                    contribution_counts[b, h_start:h_end, w_start:w_end] += 1

        # Average by contribution count and normalize
        importance_maps = importance_maps / (contribution_counts + 1e-6)
        importance_maps = (importance_maps - importance_maps.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]) / \
                          (importance_maps.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] -
                           importance_maps.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0])

        # Create visualization grid
        grid_size = int(np.ceil(np.sqrt(batch_size * 2)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        axes = axes.flatten()

        for idx in range(batch_size):
            # Original image
            img = fixed_images[idx].cpu().permute(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())

            # Plot original image
            axes[idx * 2].imshow(img)
            axes[idx * 2].set_title(f'Original Image {idx}')
            axes[idx * 2].axis('off')

            # Plot heatmap overlaid on original image
            axes[idx * 2 + 1].imshow(img)
            im = axes[idx * 2 + 1].imshow(importance_maps[idx].cpu(), cmap='hot', alpha=0.6)
            axes[idx * 2 + 1].set_title(f'Mapped Importance {idx}')
            axes[idx * 2 + 1].axis('off')
            plt.colorbar(im, ax=axes[idx * 2 + 1])

        # Turn off unused subplots
        for j in range(batch_size * 2, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Mapped Spatial Importance (RF Size: {rf_size}x{rf_size}, Stride: {stride})')
        plt.tight_layout()
        wandb.log({"Mapped Importance": wandb.Image(fig)})
        plt.close(fig)