import matplotlib.pyplot as plt
import torch
from torch import nn
from hebbian_layers.hebb import HebbianConv2d
from hebbian_layers.hebb_depthwise import HebbianDepthConv2d
import wandb


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


def find_receptive_field_activations(model, dataloader, layer, num_filters=25):
    """
    Find which parts of input images maximally activate specific neurons,
    considering their receptive fields.
    """
    model.eval()
    device = next(model.parameters()).device

    # Calculate receptive field size and stride for this layer
    rf_size, stride = calculate_receptive_field(model, layer)
    print(f"Receptive field size: {rf_size}x{rf_size}, Stride: {stride}")

    if hasattr(layer, 'out_channels'):
        out_channels = layer.out_channels
    else:
        raise ValueError("The target layer does not have an 'out_channels' attribute.")

    num_filters = min(num_filters, out_channels)

    # Storage for best activations and corresponding image patches
    best_activations = torch.full((num_filters,), float('-inf'), device=device)
    best_images = []
    best_positions = []

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)
            batch_size = batch.shape[0]

            # Get activations for this batch
            activations = get_layer_output(model, batch, layer)  # Shape: [B, C, H, W]

            # For each filter
            for filter_idx in range(num_filters):
                filter_activations = activations[:, filter_idx]  # Shape: [B, H, W]

                # Find maximum activation for this filter across all spatial positions and batch items
                max_activation, _ = filter_activations.view(batch_size, -1).max(dim=1)
                batch_max, batch_idx = max_activation.max(dim=0)

                if batch_max > best_activations[filter_idx]:
                    # Find the spatial position of maximum activation
                    spatial_idx = filter_activations[batch_idx].flatten().argmax()
                    h_idx, w_idx = spatial_idx // filter_activations.shape[2], spatial_idx % filter_activations.shape[2]

                    # Calculate corresponding position in input image
                    input_h = h_idx * stride
                    input_w = w_idx * stride

                    # Extract the receptive field patch
                    h_start = max(0, input_h - rf_size // 2)
                    w_start = max(0, input_w - rf_size // 2)
                    h_end = min(batch.shape[2], h_start + rf_size)
                    w_end = min(batch.shape[3], w_start + rf_size)

                    patch = batch[batch_idx, :, h_start:h_end, w_start:w_end]

                    best_activations[filter_idx] = batch_max
                    if len(best_images) > filter_idx:
                        best_images[filter_idx] = patch
                        best_positions[filter_idx] = (h_start, h_end, w_start, w_end)
                    else:
                        best_images.append(patch)
                        best_positions.append((h_start, h_end, w_start, w_end))

                    print(f"New best activation for filter {filter_idx}: {batch_max:.4f}")

    return best_images, best_positions, best_activations


def visualize_receptive_fields(model, dataloader, layer, num_filters=25):
    """Visualize the receptive fields that maximally activate each filter"""
    best_patches, positions, activations = find_receptive_field_activations(model, dataloader, layer, num_filters)

    # Create visualization grid
    grid_size = int(num_filters ** 0.5) + (1 if num_filters ** 0.5 % 1 > 0 else 0)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(num_filters):
        img = best_patches[i].cpu().permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        axes[i].imshow(img)
        h_start, h_end, w_start, w_end = positions[i]
        axes[i].set_title(f'Filter {i}\nPos: ({h_start}:{h_end}, {w_start}:{w_end})\nAct: {activations[i]:.2f}')
        axes[i].axis('off')

    # Turn off unused subplots
    for j in range(num_filters, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    wandb.log({"Receptive Fields": wandb.Image(fig)})
    plt.close(fig)