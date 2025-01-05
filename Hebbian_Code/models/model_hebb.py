import torch
import torch.nn as nn
import torch.nn.functional as F

from hebbian_layers.hebb import HebbianConv2d

import matplotlib.pyplot as plt
import numpy as np
import wandb

"""
File handles the creation of Hebbian models
"""

# torch.manual_seed(0)
DEFAULT_HEBB_PARAMS = {'mode': HebbianConv2d.MODE_SOFTWTA, 'w_nrm': True, 'k': 50, 'act': nn.Identity(), 'alpha': 1.}


# Activation function, as described in https://github.com/NeuromorphicComputing/SoftHebb
class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power


class Net_Hebbian(nn.Module):
    def __init__(self, hebb_params=None, version="softhebb"):
        super(Net_Hebbian, self).__init__()
        self.hebb_params = hebb_params or DEFAULT_HEBB_PARAMS
        self.version = version
        self._build_network()

    def _build_network(self):
        if self.version == "softhebb":
            self._build_softhebb_network()
        elif self.version == "mnist":
            self._build_softhebb_network_grey()
        elif self.version == "tumor":
            self._build_network_tumor()
        elif self.version == "hardhebb":
            self._build_hardhebb_network()
        elif self.version == "miconi":
            self._build_miconi_network()
        elif self.version == "lagani":
            self._build_lagani_network()
        elif self.version == "lagani_short":
            self._build_lagani_short_network()
        elif self.version == "stl_net":
            self._build_stl10_network()
        elif self.version == "stl10":
            self._build_network_stl10()
        else:
            raise ValueError(f"Unknown version: {self.version}")

    # Architecture equivalent to SoftHebb research
    # def _build_softhebb_network(self):
    #     print("Building SoftHebb model")
    #     # Layer 1
    #     self.bn1 = nn.BatchNorm2d(3, affine=False)
    #     self.conv1 = HebbianConv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, **self.hebb_params,
    #                                padding=2, t_invert=1)
    #     self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
    #     self.activ1 = Triangle(power=0.7)
    #
    #     # Layer 2
    #     self.bn2 = nn.BatchNorm2d(96, affine=False)
    #     self.conv2 = HebbianConv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, **self.hebb_params,
    #                                     t_invert=0.65, padding=1)
    #     self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
    #     self.activ2 = Triangle(power=1.4)
    #
    #     # Layer 3
    #     self.bn3 = nn.BatchNorm2d(128, affine=False)
    #     self.conv3 = HebbianConv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, **self.hebb_params,
    #                                     t_invert=0.25, padding=1)
    #     self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    #     self.activ3 = Triangle(power=1.)
    #
    #     # Output layers
    #     self.flatten = nn.Flatten()
    #     self.fc1 = nn.Linear(3072, 10)
    #     self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 3072)
    #     self.dropout = nn.Dropout(0.5)

    def _build_softhebb_network(self):
        print("Building SoftHebb model")
        # Layer 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = HebbianConv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, **self.hebb_params,
                                   padding=2, t_invert=1, bcm_theta=0.3, sigma_e=1.2, sigma_i=1.3, lateral_kernel=5,
                                   lr=0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ1 = Triangle(power=0.7)

        # Layer 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = HebbianConv2d(in_channels=96, out_channels=384, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.65, padding=1, bcm_theta=0.35, sigma_e=1.0, sigma_i=1.2, lateral_kernel=3,
                                   lr=0.08)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ2 = Triangle(power=1.4)

        # Layer 3
        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = HebbianConv2d(in_channels=384, out_channels=1536, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=1, bcm_theta=0.35, sigma_e=0.8, sigma_i=1.1, lateral_kernel=3,
                                   lr=0.05)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ3 = Triangle(power=1.)

        # Output layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(24576, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 24576)
        self.dropout = nn.Dropout(0.5)

    def _build_softhebb_network_grey(self):
        print("Building SoftHebb Greyscale model")
        # Layer 1
        self.bn1 = nn.BatchNorm2d(1, affine=False)
        self.conv1 = HebbianConv2d(in_channels=1, out_channels=96, kernel_size=5, stride=1, **self.hebb_params,
                                   padding=2, t_invert=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ1 = Triangle(power=0.7)

        # Layer 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = HebbianConv2d(in_channels=96, out_channels=384, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.65, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ2 = Triangle(power=1.4)

        # Layer 3
        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = HebbianConv2d(in_channels=384, out_channels=1536, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ3 = Triangle(power=1.)

        # Output layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(18816, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 18816)
        self.dropout = nn.Dropout(0.5)

    # For Brain_Tumor dataset
    def _build_network_tumor(self):
        print("Building Tumor model")
        # Layer 1
        self.bn1 = nn.BatchNorm2d(1, affine=False)
        self.conv1 = HebbianConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, **self.hebb_params,
                                   padding=0, t_invert=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.activ1 = Triangle(power=0.7)

        # Layer 2
        self.bn2 = nn.BatchNorm2d(16, affine=False)
        self.conv2 = HebbianConv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.65, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.activ2 = Triangle(power=1.4)

        # Layer 3
        self.bn3 = nn.BatchNorm2d(32, affine=False)
        self.conv3 = HebbianConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.activ3 = Triangle(power=1.)

        # Layer 4
        self.bn4 = nn.BatchNorm2d(64, affine=False)
        self.conv4 = HebbianConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.activ4 = Triangle(power=1.)

        # Layer 5
        self.bn5 = nn.BatchNorm2d(128, affine=False)
        self.conv5 = HebbianConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=0)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.activ5 = Triangle(power=1.)

        # Layer 6
        self.bn6 = nn.BatchNorm2d(256, affine=False)
        self.conv6 = HebbianConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=0)
        self.pool6 = nn.MaxPool2d(kernel_size=2)
        self.activ6 = Triangle(power=1.)

        # Output layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 4)  # 4 classes for tumor classification
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(1024, 2048)
        self.fc2.weight.data = 0.11048543456039805 * torch.rand(4, 1024)
        self.dropout = nn.Dropout(0.5)

    # For STL10 dataset
    def _build_network_stl10(self):
        print("Building STL-10 model")
        # Layer 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = HebbianConv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, **self.hebb_params,
                                   padding=2, t_invert=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activ1 = Triangle(power=0.7)

        # Layer 2
        self.bn2 = nn.BatchNorm2d(32, affine=False)
        self.conv2 = HebbianConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.65, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activ2 = Triangle(power=1.4)

        # Layer 3
        self.bn3 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = HebbianConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activ3 = Triangle(power=1.)

        # Layer 4
        self.bn4 = nn.BatchNorm2d(128, affine=False)
        self.conv4 = HebbianConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activ4 = Triangle(power=1.)

        # Output layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 9216)
        self.dropout = nn.Dropout(0.5)

    # Architecture equivalent to SoftHebb STL research
    def _build_stl10_network(self):
        print("Building STL SoftHebb model")
        # Layer 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = HebbianConv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, **self.hebb_params,
                                   padding=2, t_invert=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ1 = Triangle(power=0.7)

        # Layer 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = HebbianConv2d(in_channels=96, out_channels=384, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.65, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ2 = Triangle(power=1.4)

        # Layer 3
        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = HebbianConv2d(in_channels=384, out_channels=1536, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ3 = Triangle(power=1.)

        # Layer 4
        self.bn4 = nn.BatchNorm2d(1536, affine=False)
        self.conv4 = HebbianConv2d(in_channels=1536, out_channels=6144, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=1)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ4 = Triangle(power=1.)

        # Output layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(24576, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 24576)
        self.dropout = nn.Dropout(0.5)

    # Architecture equivalent to SoftHebb research, but without padding
    def _build_hardhebb_network(self):  # Hybrid: Similar to Lagani but with softhebb number of filters and layers
        # Layer 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = HebbianConv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, **self.hebb_params,
                                   padding=0, t_invert=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.activ1 = Triangle(power=0.7)

        # Layer 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = HebbianConv2d(in_channels=96, out_channels=384, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.65, padding=0)
        self.activ2 = Triangle(power=1.4)

        # Layer 3
        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = HebbianConv2d(in_channels=384, out_channels=1536, kernel_size=3, stride=1,
                                   **self.hebb_params, t_invert=0.25, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ3 = Triangle(power=1.)

        # Output layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(38400, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 38400)
        self.dropout = nn.Dropout(0.5)

    # Architecture equivalent to Miconi 3-layer research
    def _build_miconi_network(self):
        # Layer 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = HebbianConv2d(in_channels=3, out_channels=100, kernel_size=5, stride=1, **self.hebb_params,
                                   padding=0, t_invert=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ1 = Triangle(power=1.)

        # Layer 2
        self.bn2 = nn.BatchNorm2d(100, affine=False)
        self.conv2 = HebbianConv2d(in_channels=100, out_channels=196, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.65, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ2 = Triangle(power=1.)

        # Layer 3
        self.bn3 = nn.BatchNorm2d(196, affine=False)
        self.conv3 = HebbianConv2d(in_channels=196, out_channels=400, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ3 = Triangle(power=1.)

        # Output layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 1600)
        self.dropout = nn.Dropout(0.5)

    # Architecture equivalent to Lagani 3-layer research
    def _build_lagani_short_network(self):
        # Layer 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = HebbianConv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, **self.hebb_params,
                                   padding=0, t_invert=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.activ1 = Triangle(power=1.)

        # Layer 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = HebbianConv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.65, padding=0)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.activ2 = Triangle(power=1.)

        # Layer 3
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.conv3 = HebbianConv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ3 = Triangle(power=1.)

        # Output layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4800, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 4800)
        self.dropout = nn.Dropout(0.5)

    # Architecture equivalent to Lagani 4-layer research
    def _build_lagani_network(self):
        # Layer 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = HebbianConv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, **self.hebb_params,
                                   padding=0, t_invert=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.activ1 = Triangle(power=1.)

        # Layer 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = HebbianConv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.65, padding=0)
        self.activ2 = Triangle(power=1.)

        # Layer 3
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.conv3 = HebbianConv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ3 = Triangle(power=1.)

        # Layer 4
        self.bn4 = nn.BatchNorm2d(192, affine=False)
        self.conv4 = HebbianConv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, **self.hebb_params,
                                   t_invert=0.25, padding=0)
        self.activ4 = Triangle(power=1.)

        # Output layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2304, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 2304)
        self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        if not self.version == "tumor":
            x = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        else:
            x = self.pool1(self.activ1(self.conv1(self.bn1(x))))
            # x = self.pool1(self.activ1(self.conv1(x)))
        return x

    def features_extract(self, x):
        x = self.forward_features(x)
        if self.version == "lagani":
            x = self.activ2(self.conv2(self.bn2(x)))
            x = self.pool3(self.activ3(self.conv3(self.bn3(x))))
            x = self.activ4(self.conv4(self.bn4(x)))
        elif self.version == "hardhebb" or self.version == "lagani_short":
            x = self.activ2(self.conv2(self.bn2(x)))
            x = self.pool3(self.activ3(self.conv3(self.bn3(x))))
        elif self.version == "softhebb" or self.version == "mnist" or self.version == "miconi":
            x = self.pool2(self.activ2(self.conv2(self.bn2(x))))
            x = self.pool3(self.activ3(self.conv3(self.bn3(x))))
        elif self.version == "stl_net" or self.version == "stl10":
            x = self.pool2(self.activ2(self.conv2(self.bn2(x))))
            x = self.pool3(self.activ3(self.conv3(self.bn3(x))))
            x = self.pool4(self.activ4(self.conv4(self.bn4(x))))
        elif self.version == "tumor":
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
        if self.version == "tumor":
            x = nn.ReLU()(x)
            x = self.fc2(self.dropout(x))
        return x

    # Plot neurons/filter of a target layer
    def plot_grid(self, tensor, path, num_rows=5, num_cols=5, layer_name=""):
        # Ensure we're working with the first 25 filters (or less if there are fewer)
        excitatory = tensor[:20]
        inhibitory = tensor[-5:]
        # Symmetric normalization for excitatory weights
        max_abs_exc = torch.max(torch.abs(excitatory))
        norm_exc = excitatory / (max_abs_exc + 1e-8)
        # Symmetric normalization for inhibitory weights
        max_abs_inh = torch.max(torch.abs(inhibitory))
        norm_inh = inhibitory / (max_abs_inh + 1e-8)
        tensor = torch.cat((norm_exc, norm_inh))
        # Normalize the tensor
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
                        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
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
