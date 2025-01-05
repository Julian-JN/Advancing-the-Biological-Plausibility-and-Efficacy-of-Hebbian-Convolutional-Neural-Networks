import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import wandb
import math

"""
Backpropagation model similar to model_hebb.py
TODO: make code more similar to model_hebb.py, with different architectures controlled by an argument
"""

torch.manual_seed(0)

class VisualizationMixin:
    def plot_grid(self, tensor, path, num_rows=5, num_cols=5, layer_name=""):
        # Ensure we're working with the first 25 filters (or less if there are fewer)
        tensor = tensor[:25]
        # Normalize the tensor
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        # Move to CPU and convert to numpy
        tensor = tensor.cpu().detach().numpy()

        if tensor.shape[2] == 1 and tensor.shape[3] == 1:  # 1x1 convolution case
            out_channels, in_channels = tensor.shape[:2]
            fig = plt.figure(figsize=(14, 10))

            # Create a gridspec for the layout
            gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 3],
                                left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.2)

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            cbar_ax = fig.add_subplot(gs[:, 1])

            # Bar plot for average weights per filter
            avg_weights = tensor.mean(axis=(1, 2, 3))
            norm = plt.Normalize(vmin=avg_weights.min(), vmax=avg_weights.max())
            im1 = ax1.bar(range(out_channels), avg_weights, color=plt.cm.RdYlGn(norm(avg_weights)))
            ax1.set_xlabel('Filter Index')
            ax1.set_ylabel('Average Weight')
            ax1.set_title(f'Average Weights for 1x1 Kernels in {layer_name}')
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

            # Heatmap for detailed weight distribution
            im2 = ax2.imshow(tensor.reshape(out_channels, in_channels), cmap='RdYlGn', aspect='auto', norm=norm)
            ax2.set_xlabel('Input Channel')
            ax2.set_ylabel('Output Channel (Filter)')
            ax2.set_title('Detailed Weight Distribution')

            # Add colorbar to the right of both subplots
            fig.colorbar(im2, cax=cbar_ax, label='Normalized Weight Value')

        else:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            fig.suptitle(f'First 25 Filters of {layer_name}')
            for i, ax in enumerate(axes.flat):
                if i < tensor.shape[0]:
                    filter_img = tensor[i]
                    # Handle different filter shapes
                    if filter_img.shape[0] == 3:  # RGB filter (3, H, W)
                        filter_img = np.transpose(filter_img, (1, 2, 0))
                    elif filter_img.shape[0] == 1:  # Grayscale filter (1, H, W)
                        filter_img = filter_img.squeeze()
                    else:  # Multi-channel filter (C, H, W), take mean across channels
                        filter_img = np.mean(filter_img, axis=0)
                    ax.imshow(filter_img, cmap='viridis' if filter_img.ndim == 2 else None)
                    ax.set_title(f'Filter {i + 1}')
                ax.axis('off')

        if path:
            fig.savefig(path, bbox_inches='tight')
        wandb.log({f'{layer_name} filters': wandb.Image(fig)})
        plt.close(fig)

    def visualize_filters(self, layer_name='conv1', save_path=None):
        weights = getattr(self, layer_name).weight.data
        self.plot_grid(weights, save_path, layer_name=layer_name)

class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power

def custom_depthwise_conv_init(conv_layer):
    in_channels = conv_layer.in_channels
    kernel_size = conv_layer.kernel_size[0]  # Assuming square kernels
    weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)
    # Depthwise separable weights
    conv_layer.weight.data = weight_range * torch.randn(in_channels, 1, *conv_layer.kernel_size)
    if conv_layer.bias is not None:
        conv_layer.bias.data.zero_()

def custom_pointwise_conv_init(conv_layer):
    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size[0]  # Should be 1 for pointwise
    groups = conv_layer.groups
    weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)
    conv_layer.weight.data = weight_range * torch.randn(out_channels, in_channels // groups, *conv_layer.kernel_size)
    if conv_layer.bias is not None:
        conv_layer.bias.data.zero_()

class Net_Backpropagation_depth(nn.Module, VisualizationMixin):
    def __init__(self):
        super(Net_Backpropagation_depth, self).__init__()

        # A single Depthwise convolutional layer
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(5,5), padding=2, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ1 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3), padding=1, bias=False, groups=96)
        self.bn_point2 = nn.BatchNorm2d(96, affine=False)
        self.conv_point2 = nn.Conv2d(in_channels=96, out_channels=384, kernel_size=(1,1), padding=0, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ2 = nn.ReLU()

        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), padding=1, bias=False, groups=384)
        self.bn_point3 = nn.BatchNorm2d(384, affine=False)
        self.conv_point3 = nn.Conv2d(in_channels=384, out_channels=1536, kernel_size=(1,1), padding=0, bias=False)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ3 = nn.ReLU()

        self.flatten = nn.Flatten()
        # Final fully-connected layer classifier
        self.fc1 = nn.Linear(24576, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 24576)
        self.dropout = nn.Dropout(0.5)

        # Apply custom initialization to depthwise convolutional layers
        custom_pointwise_conv_init(self.conv1)
        custom_depthwise_conv_init(self.conv2)
        custom_depthwise_conv_init(self.conv3)

        # Apply custom initialization to pointwise convolutional layers
        custom_pointwise_conv_init(self.conv_point2)
        custom_pointwise_conv_init(self.conv_point3)

    def forward_features(self, x):
        x = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        return x

    def features_extract(self, x):
        x = self.forward_features(x)
        x = self.pool2(self.activ2(self.conv_point2(self.bn_point2(self.conv2(self.bn2(x))))))
        x = self.pool3(self.activ3(self.conv_point3(self.bn_point3(self.conv3(self.bn3(x))))))
        return x

    def forward(self, x):
        x = self.features_extract(x)
        x = self.flatten(x)
        x = self.fc1(self.dropout(x))
        return x


class Net_Backpropagation(nn.Module, VisualizationMixin):
    def __init__(self):
        super(Net_Backpropagation, self).__init__()

        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(5,5), padding=2, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ1 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=384, kernel_size=(3,3), padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ2 = nn.ReLU()

        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=1536, kernel_size=(3,3), padding=1, bias=False)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ3 = nn.ReLU()

        self.flatten = nn.Flatten()
        # Final fully-connected layer classifier
        self.fc1 = nn.Linear(24576, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 24576)
        self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        x = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        return x

    def features_extract(self, x):
        x = self.forward_features(x)
        x = self.pool2(self.activ2(self.conv2(self.bn2(x))))
        x = self.pool3(self.activ3(self.conv3(self.bn3(x))))
        return x

    def forward(self, x):
        x = self.features_extract(x)
        x = self.flatten(x)
        x = self.fc1(self.dropout(x))
        # x = nn.ReLU()(x)  # Apply ReLU after the linear layer
        # x = self.fc2(self.dropout(x))

        return x

class Net_Backpropagation_Tumor(nn.Module, VisualizationMixin):
    def __init__(self):
        super(Net_Backpropagation_Tumor, self).__init__()

        # For 128x128 input:
        # Layer 1: 128x128x3 -> 126x126x16 -> 63x63x16 (after pooling)
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.activ1 = nn.ReLU()

        # Layer 2: 63x63x16 -> 61x61x32 -> 30x30x32 (after pooling)
        self.bn2 = nn.BatchNorm2d(16, affine=False)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.activ2 = nn.ReLU()

        # Layer 3: 30x30x32 -> 28x28x64 -> 14x14x64 (after pooling)
        self.bn3 = nn.BatchNorm2d(32, affine=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.activ3 = nn.ReLU()

        # self.bn4 = nn.BatchNorm2d(64, affine=False)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), bias=False)
        # self.pool4 = nn.MaxPool2d(kernel_size=2)
        # self.activ4 = nn.ReLU()

        self.flatten = nn.Flatten()

        # Input to first FC layer: 14 * 14 * 64 = 12544
        self.fc1 = nn.Linear(12544, 1024)
        self.fc2 = nn.Linear(1024, 4)  # 4 classes for tumor classification
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(1024, 12544)
        self.fc2.weight.data = 0.11048543456039805 * torch.rand(4, 1024)
        self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        x = self.pool1(self.activ1(self.conv1(x)))
        return x

    def features_extract(self, x):
        x = self.forward_features(x)
        x = self.pool2(self.activ2(self.conv2(x)))
        x = self.pool3(self.activ3(self.conv3(x)))
        # x = self.pool4(self.activ4(self.conv4(x)))
        return x

    def forward(self, x):
        x = self.features_extract(x)
        x = self.flatten(x)
        x = self.fc1(self.dropout(x))
        x = nn.ReLU()(x)
        x = self.fc2(self.dropout(x))
        return x

# class Net_Backpropagation_TumorLarge(nn.Module, VisualizationMixin):
#     def __init__(self):
#         super(Net_Backpropagation_TumorLarge, self).__init__()
#
#         # Layer 1: 128x128x3 -> 128x128x16 -> 64x64x16 (after pooling)
#         self.bn1 = nn.BatchNorm2d(3, affine=False)
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), padding=1, bias=False)
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#         self.activ1 = nn.ReLU()
#
#         # Layer 2: 64x64x16 -> 64x64x32 -> 32x32x32 (after pooling)
#         self.bn2 = nn.BatchNorm2d(16, affine=False)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         self.activ2 = nn.ReLU()
#
#         # Layer 3: 32x32x32 -> 32x32x64 -> 16x16x64 (after pooling)
#         self.bn3 = nn.BatchNorm2d(32, affine=False)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False)
#         self.pool3 = nn.MaxPool2d(kernel_size=2)
#         self.activ3 = nn.ReLU()
#
#         # Layer 4: 16x16x64 -> 16x16x128 -> 8x8x128 (after pooling)
#         self.bn4 = nn.BatchNorm2d(64, affine=False)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False)
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
#         self.activ4 = nn.ReLU()
#
#         self.flatten = nn.Flatten()
#
#         # Input to first FC layer: 8 * 8 * 128 = 8192
#         self.fc1 = nn.Linear(8192, 1024)
#         self.fc2 = nn.Linear(1024, 4)
#         self.fc1.weight.data = 0.11048543456039805 * torch.rand(1024, 8192)
#         self.fc2.weight.data = 0.11048543456039805 * torch.rand(4, 1024)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward_features(self, x):
#         x = self.pool1(self.activ1(self.conv1(self.bn1(x))))
#         return x
#
#     def features_extract(self, x):
#         x = self.forward_features(x)
#         x = self.pool2(self.activ2(self.conv2(self.bn2(x))))
#         x = self.pool3(self.activ3(self.conv3(self.bn3(x))))
#         x = self.pool4(self.activ4(self.conv4(self.bn4(x))))
#         return x
#
#     def forward(self, x):
#         x = self.features_extract(x)
#         x = self.flatten(x)
#         x = self.fc1(self.dropout(x))
#         x = nn.ReLU()(x)
#         x = self.fc2(self.dropout(x))
#         return x
#
class Net_Backpropagation_TumorLarge(nn.Module, VisualizationMixin):
    def __init__(self):
        super(Net_Backpropagation_TumorLarge, self).__init__()

        # Layer 1: 128x128x3 -> 128x128x16 -> 64x64x16 (after pooling)
        self.bn1 = nn.BatchNorm2d(1, affine=False)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.activ1 = nn.ReLU()

        # Layer 2: 64x64x16 -> 64x64x32 -> 32x32x32 (after pooling)
        self.bn2 = nn.BatchNorm2d(16, affine=False)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.activ2 = nn.ReLU()

        # Layer 3: 32x32x32 -> 32x32x64 -> 16x16x64 (after pooling)
        self.bn3 = nn.BatchNorm2d(32, affine=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.activ3 = nn.ReLU()

        # Layer 4: 16x16x64 -> 16x16x128 -> 8x8x128 (after pooling)
        self.bn4 = nn.BatchNorm2d(64, affine=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), bias=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.activ4 = nn.ReLU()

        # Layer 5:
        self.bn5 = nn.BatchNorm2d(128, affine=False)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), bias=False)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.activ5 = nn.ReLU()

        # Layer 6:
        self.bn6 = nn.BatchNorm2d(256, affine=False)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), bias=False)
        self.pool6 = nn.MaxPool2d(kernel_size=2)
        self.activ6 = nn.ReLU()

        self.flatten = nn.Flatten()

        # Input to first FC layer: 8 * 8 * 128 = 8192
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 4)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(1024, 2048)
        self.fc2.weight.data = 0.11048543456039805 * torch.rand(4, 1024)
        self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        x = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        return x

    def features_extract(self, x):
        x = self.forward_features(x)
        x = self.pool2(self.activ2(self.conv2(self.bn2(x))))
        x = self.pool3(self.activ3(self.conv3(self.bn3(x))))
        x = self.pool4(self.activ4(self.conv4(self.bn4(x))))
        x = self.pool5(self.activ5(self.conv5(self.bn5(x))))
        x = self.pool6(self.activ6(self.conv6(self.bn6(x))))
        return x

    def forward(self, x):
        x = self.features_extract(x)
        x = self.flatten(x)
        x = self.fc1(self.dropout(x))
        x = nn.ReLU()(x)
        x = self.fc2(self.dropout(x))
        return x

class Net_Backpropagation_STL10(nn.Module, VisualizationMixin):
    def __init__(self):
        super(Net_Backpropagation_STL10, self).__init__()

        # Layer 1: 96x96x3 -> 96x96x32 -> 48x48x32 (after pooling)
        # Using kernel=5 with padding=2 to maintain spatial dimensions
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), bias=False, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activ1 = nn.ReLU()

        # Layer 2: 48x48x32 -> 48x48x64 -> 24x24x64 (after pooling)
        self.bn2 = nn.BatchNorm2d(32, affine=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), bias=False, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activ2 = nn.ReLU()

        # Layer 3: 24x24x64 -> 24x24x128 -> 12x12x128 (after pooling)
        self.bn3 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), bias=False, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activ3 = nn.ReLU()

        # Layer 4: 12x12x128 -> 12x12x256 -> 6x6x256 (after pooling)
        self.bn4 = nn.BatchNorm2d(128, affine=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), bias=False, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activ4 = nn.ReLU()

        self.flatten = nn.Flatten()

        # Final feature map size: 6x6x256 = 9216
        self.fc1 = nn.Linear(9216, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 9216)
        self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        x = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        return x

    def features_extract(self, x):
        x = self.forward_features(x)
        x = self.pool2(self.activ2(self.conv2(self.bn2(x))))
        x = self.pool3(self.activ3(self.conv3(self.bn3(x))))
        x = self.pool4(self.activ4(self.conv4(self.bn4(x))))
        return x

    def forward(self, x):
        x = self.features_extract(x)
        x = self.flatten(x)
        x = self.fc1(self.dropout(x))
        return x