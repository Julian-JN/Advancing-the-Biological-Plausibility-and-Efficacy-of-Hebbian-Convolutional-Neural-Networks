import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from hebbian_layers.hebb import HebbianConv2d
from hebbian_layers.hebb_depthwise import HebbianDepthConv2d
import wandb

# Code to visualise receptive fields

def get_partial_model(model, target_layer):
    layers = []
    for layer in model.children():
        layers.append(layer)
        if layer == target_layer:
            break
    return torch.nn.Sequential(*layers)


def calculate_receptive_field(model, target_layer):
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

    return current_rf

def get_layer_output(model, x, target_layer):
    """
    Forward pass through the model, stopping at the target custom Hebbian layer.
    """
    for layer in model.children():
        x = layer(x)  # Pass through each layer
        if layer == target_layer:
            return x  # Return the output of the target Hebbian layer
    raise ValueError(f"Target layer {target_layer} not found in the model.")

def remove_padding(model, target_layer):
    """Remove padding from layers before the target layer."""
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            layer.padding = (0, 0)
        elif isinstance(layer, (HebbianConv2d, HebbianDepthConv2d)):
            # For custom SoftHebbConv2d layers, we need to modify the padding directly
            layer.padding = 0
        if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            layer.padding = (0, 0)
        if layer == target_layer:
            break
    return model

def gaussian_blur(x, kernel_size=5, sigma=1.0):
    channels = x.shape[1]
    kernel = torch.tensor([
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ], device=x.device).unsqueeze(0).unsqueeze(0) / 256.0
    kernel = kernel.repeat(channels, 1, 1, 1)
    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding, groups=channels)

class SingleMax:
    def __init__(self, max_val: float, eps: float):
        self.max_val = max_val
        self.eps = eps

    def __call__(self, outputs):
        if self.max_val is None:
            return outputs > 0
        else:
            return (self.max_val - outputs) < self.eps

class L2ProjGradientDescent:
    def __init__(self, steps, random_start=True, rel_stepsize=0.1):
        self.steps = steps
        self.random_start = random_start
        self.rel_stepsize = rel_stepsize

    def get_random_start(self, x0, epsilon):
        batch_size, c, h, w = x0.shape
        r = torch.randn(batch_size, c * h * w, device=x0.device)
        r = r / r.norm(dim=1, keepdim=True)
        r = r.view_as(x0)
        return x0 + 0.00001 * epsilon * r

    def normalize_gradient(self, grad):
        return grad / (grad.view(grad.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8)

    def project(self, x, x0, epsilon):
        delta = x - x0
        delta = epsilon * delta / delta.view(delta.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=1e-12)
        return x0 + delta

    def run(self, model, x0, target_layer, filter_idx, epsilon, criterion):
        x = x0.clone()
        if self.random_start:
            x = self.get_random_start(x0, epsilon)
        for _ in range(self.steps):
            x.requires_grad_(True)
            x_smooth = gaussian_blur(x, sigma=1.0)  # Apply Gaussian smoothing
            activation = get_layer_output(model, x, target_layer)
            loss = -activation[0, filter_idx].sum()
            if criterion(loss.item()):
                break
            grad = torch.autograd.grad(loss, x)[0]
            grad = self.normalize_gradient(grad)
            with torch.no_grad():
                x = x - self.rel_stepsize * epsilon * grad
                x = self.project(x, x0, epsilon)
                x.clamp_(0, 1)
        return x

def visualize_filters(model, layer, num_filters=25, input_shape=(1, 3, 32, 32), step_size=0.001, iterations=500,
                      random_start=True, nb_start=1, l2_norm=True, max_val=1e10, eps=1e-05, epsilons=None):
    if epsilons is None:
        epsilons = torch.tensor([255.0], device='cuda') / 255.0
    else:
        epsilons = torch.tensor(epsilons, device='cuda')
    model.eval()
    model = remove_padding(model, layer)
    receptive_field_size = calculate_receptive_field(model, layer)
    print(f"Receptive field size: {receptive_field_size}x{receptive_field_size}")
    # Change depending on dataset
    input_shape = (1, 3, receptive_field_size, receptive_field_size)
    if hasattr(layer, 'out_channels'):
        out_channels = layer.out_channels
    else:
        raise ValueError("The target layer does not have an 'out_channels' attribute.")
    num_filters = min(num_filters, out_channels)
    filter_images = []
    criterion = SingleMax(max_val, eps)
    pgd = L2ProjGradientDescent(steps=iterations, random_start=random_start, rel_stepsize=step_size)

    for filter_idx in range(num_filters):
        best_image = None
        best_activation = float('-inf')
        for start in range(nb_start):
            x0 = torch.zeros(input_shape, device='cuda', requires_grad=True)
            optimized_image = pgd.run(model, x0, layer, filter_idx, epsilons, criterion)
            activation = -get_layer_output(model, optimized_image, layer)[0, filter_idx].sum().item()
            if activation > best_activation:
                print(f"New best Receptive Field found for filter {filter_idx} in Reboot {start}")
                best_activation = activation
                best_image = optimized_image
        optimized_image = best_image.cpu().squeeze(0).permute(1, 2, 0)
        optimized_image = (optimized_image - optimized_image.min()) / (optimized_image.max() - optimized_image.min())
        filter_images.append(optimized_image)
    # Plot the filter visualizations in a grid
    grid_size = int(num_filters ** 0.5) + (1 if num_filters ** 0.5 % 1 > 0 else 0)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()
    for i in range(num_filters):
        axes[i].imshow(filter_images[i].numpy())
        axes[i].set_title(f'Filter {i + 1}')
        axes[i].axis('off')
    # Turn off unused subplots
    for j in range(num_filters, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    wandb.log({"Rceptive Fields": wandb.Image(fig)})
    plt.close(fig)