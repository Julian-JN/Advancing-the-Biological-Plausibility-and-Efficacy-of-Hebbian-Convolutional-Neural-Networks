import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import matplotlib.pyplot as plt
import torch.nn.init as init


"""
Uses almost identical code to hebb.py. Please refer to hebb.py for additional explanations on code functionality
Only changes are the competition modes, which apply competition across spatial neurons in a filter for channel independence
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def normalize(x, dim=None):
    nrm = (x ** 2).sum(dim=dim, keepdim=True) ** 0.5
    nrm[nrm == 0] = 1.
    return x / nrm


def symmetric_pad(x, padding):
    if padding == 0:
        return x
    return F.pad(x, (padding,) * 4, mode='reflect')


def center_surround_init(out_channels, in_channels, kernel_size, groups=1):
    # Calculate weight range
    weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)
    # Calculate sigma based on kernel size (using equation 3 from the paper)
    gamma = torch.empty(out_channels).uniform_(0, 0.5)
    sigma = (kernel_size / 4) * torch.sqrt((1 - gamma ** 2) / (-torch.log(gamma)))
    # Create meshgrid for x and y coordinates
    x = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
    y = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    # Calculate center and surround Gaussians
    center = torch.exp(-(xx ** 2 + yy ** 2) / (2 * (gamma.view(-1, 1, 1) * sigma.view(-1, 1, 1)) ** 2))
    surround = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma.view(-1, 1, 1) ** 2))
    # Calculate DoG (Difference of Gaussians)
    dog = center - surround
    # Normalize DoG
    ac = torch.sum(torch.clamp(dog, min=0))
    as_ = torch.sum(-torch.clamp(dog, max=0))
    dog = weight_range * 0.5 * dog / (ac + as_)
    # Assign excitatory (positive) or inhibitory (negative) centers
    center_type = torch.cat([torch.ones(out_channels // 2), -torch.ones(out_channels - out_channels // 2)])
    center_type = center_type[torch.randperm(out_channels)].view(-1, 1, 1)
    dog = dog * center_type
    # Repeat for in_channels and reshape to match conv2d weight shape
    dog = dog.unsqueeze(1).repeat(1, in_channels // groups, 1, 1)
    dog = dog.reshape(out_channels, in_channels // groups, kernel_size, kernel_size)
    return nn.Parameter(dog)

def create_sm_kernel(kernel_size=5, sigma_e=1.2, sigma_i=1.4):
    center = kernel_size // 2
    x, y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
    x = x.float() - center
    y = y.float() - center
    gaussian_e = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_e ** 2))
    gaussian_i = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_i ** 2))
    dog = gaussian_e / (2 * math.pi * sigma_e ** 2) - gaussian_i / (2 * math.pi * sigma_i ** 2)
    sm_kernel = dog / dog[center, center]
    return sm_kernel.unsqueeze(0).unsqueeze(0).to(device)

# Doubts:
# Visualizing weights, as separated between channels and spatial
#

class HebbianDepthConv2d(nn.Module):
    """
	A 2d convolutional layer that learns through Hebbian plasticity
	"""

    MODE_HPCA = 'hpca'
    MODE_BASIC_HEBBIAN = 'basic'
    MODE_WTA = 'wta'
    MODE_SOFTWTA = 'soft'
    MODE_BCM = 'bcm'
    MODE_HARDWT = "hard"
    MODE_PRESYNAPTIC_COMPETITION = "pre"
    MODE_TEMPORAL_COMPETITION = "temp"
    MODE_ADAPTIVE_THRESHOLD = "thresh"
    MODE_ANTIHARDWT = "antihard"

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0,
                 w_nrm=False, act=nn.Identity(),
                 mode=MODE_SOFTWTA, patchwise=True,
                 contrast=1., uniformity=False, alpha=1., top_k=1, prune_rate=0, t_invert=1.,
                 # New configuration parameters
                 use_cosine_similarity=False,
                 use_lateral_inhibition=False,
                 init_method='kaiming_uniform',
                 use_presynaptic_competition=False,
                 presynaptic_competition_type='lp_norm',
                 use_homeostasis=False,
                 temporal_window = 500,
                 competition_k = 2,
                 competition_type = "hard",
                 use_structural_plasticity=False,
                 dale=False):
        """
        Extended initialization with new configuration parameters.

        Args:
            :param out_channels: output channels of the convolutional kernel
            :param in_channels: input channels of the convolutional kernel
            :param kernel_size: size of the convolutional kernel (int or tuple)
            :param stride: stride of the convolutional kernel (int or tuple
            :param w_nrm: whether to normalize the weight vectors before computing outputs
            :param act: the nonlinear activation function after convolution
            :param mode: the learning mode, either 'swta' or 'hpca'
            :param k: softmax inverse temperature parameter used for swta-type learning
            :param patchwise: whether updates for each convolutional patch should be computed separately,
            and then aggregated
            :param contrast: coefficient that rescales negative compared to positive updates in contrastive-type learning
            :param uniformity: whether to use uniformity weighting in contrastive-type learning.
            :param alpha: weighting coefficient between hebbian and backprop updates (0 means fully backprop, 1 means fully hebbian).
            use_cosine_similarity (bool): Whether to use cosine similarity instead of dot product
            use_lateral_inhibition (bool): Whether to apply surround modulation/lateral inhibition
            init_method (str): Weight initialization method ('kaiming_uniform', 'xavier_normal',
                             'orthogonal', 'softhebb')
            use_presynaptic_competition (bool): Whether to enable presynaptic competition
            presynaptic_competition_mode (str): Type of presynaptic competition ('lp_norm',
                                              'linear', 'softmax')
            use_homeostasis (bool): Whether to enable homeostatic plasticity
            use_structural_plasticity (bool): Whether to enable structural plasticity
        """
        super(HebbianDepthConv2d, self).__init__()

        # Store new configuration parameters
        self.use_cosine_similarity = use_cosine_similarity
        self.use_lateral_inhibition = use_lateral_inhibition
        self.init_method = init_method
        self.use_presynaptic_competition = use_presynaptic_competition
        self.presynaptic_competition_type = presynaptic_competition_type
        self.use_homeostasis = use_homeostasis
        self.use_structural_plasticity = use_structural_plasticity

        # Existing initialization code
        self.mode = mode
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel = kernel_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = padding
        self.F_padding = (padding, padding, padding, padding)
        self.groups = in_channels # in_channels for depthwise
        self.padding_mode = 'reflect'
        if mode == "hard":
            self.padding_mode = 'symmetric'

        if mode == "bcm":
            self.theta_decay = 0.5
            self.theta = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        # Initialize weights based on selected method
        self.dale = dale
        self.weight = nn.Parameter(torch.randn(in_channels, 1, *self.kernel_size))
        self.init_weights()

        self.register_buffer('delta_w', torch.zeros_like(self.weight))
        self.top_k = top_k
        self.patchwise = patchwise
        self.contrast = contrast
        self.uniformity = uniformity
        self.alpha = alpha
        self.lebesgue_p = 2
        self.prune_rate = prune_rate  # 99% of connections are pruned
        self.t_invert = torch.tensor(t_invert)
        self.activation_history = None
        self.temporal_window = temporal_window
        self.competition_k = competition_k
        self.competition_type = competition_type

        self.w_nrm = w_nrm
        self.act = act

        # Initialize surround modulation kernel only if needed
        if self.use_lateral_inhibition and self.kernel != 1:
            self.sm_kernel = create_sm_kernel()
            self.register_buffer('surround_kernel', self.sm_kernel)
            # self.visualize_surround_modulation_kernel()

        # Homeostasis parameters
        if self.use_homeostasis:
            self.target = top_k / out_channels
            self.gamma = 0.5
            self.register_buffer('H', torch.zeros(out_channels))
            self.register_buffer('avg_activity', torch.zeros(out_channels))

        # Structural plasticity parameters
        if self.use_structural_plasticity:
            self.growth_probability = 0.1
            self.new_synapse_strength = 1
            self.prune_threshold_percentile = 10

    def init_weights(self):
        # Experiment with different initializations
        init_method = self.init_method  # You can change this to try different methods

        if init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        elif init_method == 'xavier_normal':
            nn.init.xavier_normal_(self.weight)
        elif init_method == 'orthogonal':
            nn.init.orthogonal_(self.weight)
        elif init_method == 'softhebb':
            weight_range = 25 / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
            self.weight = nn.Parameter(
                weight_range * torch.randn((self.out_channels, self.in_channels // self.groups, *self.kernel_size)))

        if self.dale:
            self.weight.data = self.weight.data.abs()
        plt.figure(figsize=(8, 4))
        weights = self.weight.detach().cpu().numpy()
        plt.hist(weights.flatten(), bins=50, density=True)
        plt.title(f'Weight Distribution ({init_method})')
        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.show()

    def apply_lebesgue_norm(self, w):
        return torch.sign(w) * torch.abs(w) ** (self.lebesgue_p - 1)

    def cosine(self, x, w):
        w_normalized = F.normalize(w, p=2, dim=1)
        conv_output = F.conv2d(x, w_normalized, None, self.stride, 0, self.dilation, groups=self.groups)
        x_squared = x.pow(2)
        x_squared_sum = F.conv2d(x_squared, torch.ones_like(w), None, self.stride, 0, self.dilation,
                                 self.groups)
        x_norm = torch.sqrt(x_squared_sum + 1e-8)
        cosine_sim = conv_output / x_norm
        return cosine_sim

    def apply_weights(self, x, w):
        """
		This function provides the logic for combining input x and weight w
		"""
        # w = self.apply_lebesgue_norm(self.weight)
        return F.conv2d(x, w, None, self.stride, 0, self.dilation, groups=self.groups)

    def apply_surround_modulation(self, y):
        return F.conv2d(y, self.sm_kernel.repeat(self.out_channels, 1, 1, 1),
                        padding=self.sm_kernel.size(-1) // 2, groups=self.out_channels)

    def compute_activation(self, x):
        x = symmetric_pad(x, self.padding)
        w = self.weight
        if self.dale:
            w = w.abs()
        if self.w_nrm: w = normalize(w, dim=(1, 2, 3))
        if self.use_presynaptic_competition: w = self.compute_presynaptic_competition(w)
        # For cosine similarity activation if cosine is to be used for next layer
        if self.use_cosine_similarity:
            y_depthwise = self.cosine(x,w)
        else:
            y_depthwise = self.act(self.apply_weights(x, w))
        return x, y_depthwise, w

    def forward(self, x):
        x, y_depthwise, w = self.compute_activation(x)
        if self.kernel !=1 and self.use_lateral_inhibition:
            y_depthwise = self.apply_surround_modulation(y_depthwise)
        if self.training:
            self.compute_update(x, y_depthwise, w)
        return y_depthwise

    def compute_update(self, x, y, weight):
        if self.mode == self.MODE_BASIC_HEBBIAN:
            update = self.update_basic_hebbian(x, y, weight)
        elif self.mode == self.MODE_HARDWT:
            update = self.update_hardwt(x, y, weight)
        elif self.mode == self.MODE_SOFTWTA:
            update = self.update_softwta(x, y, weight)
        elif self.mode == self.MODE_BCM:
            update = self.update_bcm(x, y, weight)
        elif self.mode == self.MODE_TEMPORAL_COMPETITION:
            update = self.update_temporal_competition(x, y, weight)
        elif self.mode == self.MODE_ADAPTIVE_THRESHOLD:
            update = self.update_adaptive_threshold(x, y, weight)
        else:
            raise NotImplementedError(f"Learning mode {self.mode} unavailable for {self.__class__.__name__} layer")

        # Weight Normalization and added to weight change buffer
        update.div_(torch.abs(update).amax() + 1e-30)
        self.delta_w += update

    def update_basic_hebbian(self, x, y, weight):
        yx = self.compute_yx(x, y)
        y_sum = y.sum(dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
        yw = y_sum * weight
        return yx - yw

    def update_hardwt(self, x, y, weight):
        y_wta = y * self.compute_wta_mask(y)
        yx = self.compute_yx(x, y_wta)
        yu = torch.sum(y_wta, dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
        return yx - yu * weight

    def update_softwta(self, x, y, weight):
        softwta_activs = self.compute_softwta_activations(y)
        yx = self.compute_yx(x, softwta_activs)
        yu = torch.sum(torch.mul(softwta_activs, y), dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
        return yx - yu * weight

    def update_bcm(self, x, y, weight):
        batch_size, out_channels, height, width = y.shape
        y_wta = y * self.compute_wta_mask(y)
        # Compute squared activation for each spatial location and channel
        y_squared = y_wta.pow(2)
        # Reshape theta to match spatial dimensions
        theta_spatial = self.theta.view(1, -1, 1, 1).expand(1, out_channels, height, width)
        # Update theta for each spatial location
        theta_update = self.theta_decay * (y_squared - theta_spatial)
        self.theta.data += theta_update.mean(dim=(0, 2, 3))
        # Compute BCM updates for each spatial location
        y_minus_theta = y_wta - theta_spatial
        bcm_factor = y_wta * y_minus_theta
        yx = self.compute_yx(x, bcm_factor)
        update = yx.view(weight.shape)
        return update

    def update_temporal_competition(self, x, y, weight):
        batch_size, out_channels, height_out, width_out = y.shape
        self.update_activation_history(y)
        temporal_winners = self.compute_temporal_winners(y)
        y_winners = temporal_winners * y
        y_winners = self.apply_competition(y_winners, batch_size, out_channels)
        yx = self.compute_yx(x, y_winners)
        y_sum = y_winners.sum(dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
        update = yx - y_sum * weight
        return update

    def update_adaptive_threshold(self, x, y, weight):
        batch_size, out_channels, height_out, width_out = y.shape
        similarities = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups)
        similarities = similarities / (torch.norm(weight.view(out_channels, -1), dim=1).view(1, -1, 1, 1) + 1e-10)
        threshold = self.compute_adaptive_threshold(similarities)
        winners = (similarities > threshold).float()
        y_winners = winners * similarities
        y_winners = self.apply_competition(y_winners, batch_size, out_channels)
        yx = self.compute_yx(x, y_winners)
        y_sum = y_winners.sum(dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
        update = yx - y_sum * weight
        return update

    def compute_yx(self, x, y):
        yx = F.conv2d(x.transpose(0, 1), y.transpose(0, 1), padding=0,
                      stride=self.dilation, dilation=self.stride).transpose(0, 1)
        yx = yx.diagonal(dim1=0, dim2=1).permute(2, 0, 1).unsqueeze(1)
        return yx

    def compute_wta_mask(self, y):
        batch_size, out_channels, height_out, width_out = y.shape
        # WTA competition within each channel
        y_flat = y.view(batch_size, out_channels, -1)
        win_neurons = torch.argmax(y_flat, dim=2)
        wta_mask = F.one_hot(win_neurons, num_classes=height_out * width_out).float()
        return wta_mask.view(batch_size, out_channels, height_out, width_out)

    def compute_softwta_activations(self, y):
        # Competition and anti-Hebbian learning for y_depthwise
        batch_size, in_channels, height_depthwise, width_depthwise = y.shape
        # Reshape to apply softmax within each channel
        y_depthwise_reshaped = y.view(batch_size, in_channels, -1)
        # Apply softmax within each channel
        flat_softwta_activs_depthwise = torch.softmax(self.t_invert * y_depthwise_reshaped, dim=2)
        # Turn all postsynaptic activations into anti-Hebbian
        flat_softwta_activs_depthwise = -flat_softwta_activs_depthwise
        # Find winners within each channel
        win_neurons_depthwise = torch.argmax(y_depthwise_reshaped, dim=2)
        # Create a mask to flip the sign of winning neurons
        mask = torch.zeros_like(flat_softwta_activs_depthwise)
        mask.scatter_(2, win_neurons_depthwise.unsqueeze(2), 1)
        # Flip the sign of winning neurons
        flat_softwta_activs_depthwise = flat_softwta_activs_depthwise * (1 - 2 * mask)
        # Reshape back to original shape
        return flat_softwta_activs_depthwise.view(batch_size, in_channels, height_depthwise,
                                                                      width_depthwise)

    def update_activation_history(self, y):
        if self.activation_history is None:
            self.activation_history = y.detach().clone()
        else:
            self.activation_history = torch.cat([self.activation_history, y.detach()], dim=0)
            if self.activation_history.size(0) > self.temporal_window:
                self.activation_history = self.activation_history[-self.temporal_window:]

    def compute_temporal_winners(self, y):
        batch_size, out_channels, height_out, width_out = y.shape
        history_spatial = self.activation_history.view(-1, out_channels, height_out, width_out)
        median_activations = torch.median(history_spatial, dim=0)[0]
        temporal_threshold = torch.mean(median_activations, dim=(1, 2), keepdim=True)
        return (median_activations > temporal_threshold).float()

    def compute_adaptive_threshold(self, similarities):
        mean_sim = similarities.mean(dim=(2, 3), keepdim=True)
        std_sim = similarities.std(dim=(2, 3), keepdim=True)
        return mean_sim + self.competition_k * std_sim

    def apply_competition(self, y, batch_size, out_channels):
        if self.mode in [self.MODE_TEMPORAL_COMPETITION, self.MODE_ADAPTIVE_THRESHOLD]:
            if self.competition_type == 'hard':
                y = y.view(batch_size, out_channels, -1)
                top_k_indices = torch.topk(y, self.top_k, dim=2, largest=True, sorted=False).indices
                y_compete = torch.zeros_like(y)
                y_compete.scatter_(2, top_k_indices, y.gather(2, top_k_indices))
                return y_compete.view_as(y)
            elif self.competition_type == 'soft':
                return torch.softmax(self.t_invert * y.view(batch_size, out_channels, -1), dim=2).view_as(y)
        return y

    # @torch.no_grad()
    # def local_update(self):
    #     """
    #     This function transfers a previously computed weight update, stored in buffer self.delta_w, to the gradient
    #     self.weight.grad of the weight parameter.
    #
    #     This function should be called before optimizer.step(), so that the optimizer will use the locally computed
    #     update as optimization direction. Local updates can also be combined with end-to-end updates by calling this
    #     function between loss.backward() and optimizer.step(). loss.backward will store the end-to-end gradient in
    #     self.weight.grad, and this function combines this value with self.delta_w as
    #     self.weight.grad = (1 - alpha) * self.weight.grad - alpha * self.delta_w
    #     Parameter alpha determines the scale of the local update compared to the end-to-end gradient in the combination.
    #     """
    #     if self.weight.grad is None:
    #         self.weight.grad = -self.alpha * self.delta_w
    #     else:
    #         self.weight.grad = (1 - self.alpha) * self.weight.grad - self.alpha * self.delta_w
    #     self.delta_w.zero_()

    @torch.no_grad()
    # Weight Update
    def local_update(self):
        new_weight = self.weight + 0.1 * self.alpha * self.delta_w
        # Update weights
        if self.dale:
            self.weight.copy_(new_weight.abs())
        else:
            self.weight.copy_(new_weight)
        self.delta_w.zero_()
