import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from hebbian_layers.hebb import HebbianConv2d
from hebbian_layers.hebb_depthwise import HebbianDepthConv2d
from models.model_residual import HebbianResidualBlock
import wandb

# Code to visualise receptive fields. Modified version of receptive_fields.py to work with residual blocks


def remove_padding_except_conv2(model):
    for module in model.modules():
        if isinstance(module, HebbianResidualBlock):
            # Don't remove padding from conv1 if it's 1x1
            if module.conv1.kernel_size[0] > 1:
                module.conv1.padding = 0
            # Keep padding for conv2
            if module.conv3.kernel_size[0] > 1:
                module.conv3.padding = 0
            if isinstance(module.shortcut, nn.Sequential):
                for layer in module.shortcut:
                    if isinstance(layer, (nn.Conv2d, HebbianConv2d)):
                        if layer.kernel_size[0] > 1:
                            layer.padding = 0
        elif isinstance(module, (nn.Conv2d, HebbianConv2d, HebbianDepthConv2d)):
            if module.kernel_size[0] > 1:
                module.padding = 0
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            module.padding = 0

def get_partial_model(model, target_layer):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, HebbianConv2d, HebbianDepthConv2d, HebbianResidualBlock, nn.MaxPool2d, nn.BatchNorm2d)):
            if isinstance(module, HebbianResidualBlock):
                layers.append((f"{name}.bn1", module.bn1))
                layers.append((f"{name}.conv1", module.conv1))
                layers.append((f"{name}.bn2", module.bn2))
                layers.append((f"{name}.conv2", module.conv2))
                layers.append((f"{name}.bn3", module.bn3))
                layers.append((f"{name}.conv3", module.conv3))
                if isinstance(module.shortcut, nn.Sequential):
                    for i, shortcut_layer in enumerate(module.shortcut):
                        layers.append((f"{name}.shortcut.{i}", shortcut_layer))
            else:
                layers.append((name, module))
        if module == target_layer:
            break
    return layers


def calculate_receptive_field(model, target_layer):
    current_rf = 1
    current_stride = 1

    print(f"Calculating receptive field for {target_layer}")
    print(f"Initial RF: {current_rf}, Initial stride: {current_stride}")

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, HebbianConv2d, HebbianDepthConv2d)):
            kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            prev_rf = current_rf
            current_rf += (kernel_size - 1) * current_stride
            prev_stride = current_stride
            current_stride *= stride
            print(f"Layer: {name}, Type: Conv, Kernel: {kernel_size}, Stride: {stride}")
            print(f"  RF: {prev_rf} -> {current_rf}, Stride: {prev_stride} -> {current_stride}")
        elif isinstance(module, nn.MaxPool2d):
            kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            stride = module.stride if isinstance(module.stride, int) else module.stride[0]
            prev_rf = current_rf
            current_rf += (kernel_size - 1) * current_stride
            prev_stride = current_stride
            current_stride *= stride
            print(f"Layer: {name}, Type: MaxPool, Kernel: {kernel_size}, Stride: {stride}")
            print(f"  RF: {prev_rf} -> {current_rf}, Stride: {prev_stride} -> {current_stride}")
        else:
            print(f"Layer: {name}, Type: {type(module).__name__} (no effect on RF)")

        if module == target_layer:
            print(f"Reached target layer. Final receptive field: {current_rf}x{current_rf}")
            return current_rf

    print(f"Target layer not found. Final receptive field: {current_rf}x{current_rf}")
    return current_rf


def get_layer_output(model, x, target_layer, debug=False):
    if debug:
        print(f"Initial input shape: {x.shape}")

    for name, module in model.named_children():
        if debug:
            print(f"Processing module: {name}")

        if isinstance(module, HebbianResidualBlock):
            if debug:
                print(f"Entering {name}, input shape: {x.shape}")
            residual = x
            out = module.activ(module.conv1(module.bn1(x)))
            if debug:
                print(f"  After conv1: {out.shape}")
            out = module.activ(module.conv2(module.bn2(out)))
            if debug:
                print(f"  After conv2: {out.shape}")
            out = module.conv3(module.bn3(out))
            if debug:
                print(f"  After conv3: {out.shape}")
            shortcut_output = module.shortcut(residual)
            if debug:
                print(f"  Shortcut output: {shortcut_output.shape}")

            # Ensure out and shortcut_output have the same size
            if out.size() != shortcut_output.size():
                min_size = min(out.size(2), shortcut_output.size(2))
                out = out[:, :, :min_size, :min_size]
                shortcut_output = shortcut_output[:, :, :min_size, :min_size]

            x = module.activ(out + shortcut_output)
            if debug:
                print(f"Exiting {name}, output shape: {x.shape}")
        elif isinstance(module, (nn.Conv2d, HebbianConv2d, HebbianDepthConv2d, nn.MaxPool2d, nn.AvgPool2d)):
            if debug:
                print(f"Entering {name}, input shape: {x.shape}")
            x = module(x)
            if debug:
                print(f"Exiting {name}, output shape: {x.shape}")
        elif isinstance(module, nn.BatchNorm2d):
            if debug:
                print(
                    f"Entering {name} (BatchNorm2d), input shape: {x.shape}, expected features: {module.num_features}")
            x = module(x)
            if debug:
                print(f"Exiting {name} (BatchNorm2d), output shape: {x.shape}")

        if module == target_layer:
            if debug:
                print(f"Reached target layer {name}, output shape: {x.shape}")
            return x

        # If the target is inside a HebbianResidualBlock, we need to look inside it
        if isinstance(module, HebbianResidualBlock):
            for sub_name, sub_module in module.named_children():
                if sub_module == target_layer:
                    if debug:
                        print(f"Reached target layer {name}.{sub_name}, output shape: {x.shape}")
                    return x

    raise ValueError(f"Target layer {target_layer} not found in the model.")


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

    def run(self, model, x0, target_layer, filter_idx, epsilon, criterion, debug=False):
        x = x0.clone()
        if self.random_start:
            x = self.get_random_start(x0, epsilon)
        for _ in range(self.steps):
            x.requires_grad_(True)
            x_smooth = gaussian_blur(x, sigma=1.0)  # Apply Gaussian smoothing
            activation = get_layer_output(model, x, target_layer, debug=debug)
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


def visualize_filters(model, target_layer, num_filters=25, input_shape=(1, 3, 32, 32), step_size=0.001, iterations=500,
                      random_start=True, nb_start=1, l2_norm=True, max_val=1e10, eps=1e-05, epsilons=None, debug=False):
    if epsilons is None:
        epsilons = torch.tensor([255.0], device='cuda') / 255.0
    else:
        epsilons = torch.tensor(epsilons, device='cuda')
    model.eval()
    # remove_padding_except_conv2(model)
    receptive_field_size = calculate_receptive_field(model, target_layer)
    print(f"Receptive field size: {receptive_field_size}x{receptive_field_size}")
    input_shape = (1, 3, receptive_field_size, receptive_field_size)

    if isinstance(target_layer, HebbianResidualBlock):
        out_channels = target_layer.conv3.out_channels
    elif hasattr(target_layer, 'out_channels'):
        out_channels = target_layer.out_channels
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
            optimized_image = pgd.run(model, x0, target_layer, filter_idx, epsilons, criterion, debug=debug)
            activation = -get_layer_output(model, optimized_image, target_layer, debug=debug)[0, filter_idx].sum().item()
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