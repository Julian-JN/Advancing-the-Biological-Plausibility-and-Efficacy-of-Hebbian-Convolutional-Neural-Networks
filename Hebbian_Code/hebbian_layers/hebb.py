import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import matplotlib.pyplot as plt

# Code uses elements from {https://github.com/GabrieleLagani/hebbdemo} and {
# https://github.com/NeuromorphicComputing/SoftHebb}

import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(0)


def normalize(x, dim=None):
    nrm = (x ** 2).sum(dim=dim, keepdim=True) ** 0.5
    nrm[nrm == 0] = 1.
    return x / nrm


def symmetric_pad(x, padding):
    if padding == 0:
        return x
    return F.pad(x, (padding,) * 4, mode='reflect')


def create_sm_kernel(kernel_size=5, sigma_e=1.2, sigma_i=1.4):
    """
    Create a surround modulation kernel.
    :param kernel_size: Size of the SM kernel.
    :param sigma_e: Standard deviation for the excitatory Gaussian.
    :param sigma_i: Standard deviation for the inhibitory Gaussian.
    :return: A normalized SM kernel.
    """
    center = kernel_size // 2
    x, y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing="ij")
    x = x.float() - center
    y = y.float() - center
    # Compute the excitatory and inhibitory Gaussians
    gaussian_e = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_e ** 2))
    gaussian_i = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_i ** 2))
    # Compute the Difference of Gaussians (DoG)
    dog = gaussian_e / (2 * math.pi * sigma_e ** 2) - gaussian_i / (2 * math.pi * sigma_i ** 2)
    # Normalize the DoG so that the center value is 1
    sm_kernel = dog / dog[center, center]
    return sm_kernel.unsqueeze(0).unsqueeze(0).to(device)


class HebbianConv2d(nn.Module):
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
                 temporal_window=100,
                 competition_k=2,
                 competition_type="hard",
                 use_structural_plasticity=False,
                 dale=False,
                 bcm_theta=0.5,
                 sigma_e=1.2,
                 sigma_i=1.4,
                 lateral_kernel=5,
                 lr=0.1):
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
        super(HebbianConv2d, self).__init__()

        self.lr = lr
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
        self.groups = 1
        self.padding_mode = 'reflect'
        if mode == "hard":
            self.padding_mode = 'symmetric'

        if mode == "bcm":
            self.theta_decay = bcm_theta
            self.theta = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        # Initialize weights based on selected method

        self.dale = dale
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // self.groups, *self.kernel_size))
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
            self.sm_kernel = create_sm_kernel(kernel_size=lateral_kernel, sigma_e=sigma_e, sigma_i=sigma_i)
            self.register_buffer('surround_kernel', self.sm_kernel)
            self.visualize_surround_modulation_kernel()

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

    def visualize_surround_modulation_kernel(self):
        """
        Visualizes the surround modulation kernel using matplotlib.
        """
        sm_kernel = self.sm_kernel.squeeze().cpu().detach().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(sm_kernel, cmap='jet')
        plt.colorbar()
        plt.title('Surround Modulation Kernel')
        plt.show()
        plt.close()

    def apply_lebesgue_norm(self, w):
        """Apply Lebesgue norm to weights."""
        return torch.sign(w) * torch.abs(w) ** (self.lebesgue_p - 1)

    def apply_weights(self, x, w):
        """
        Apply convolutional operation with weights to input.
		"""
        return F.conv2d(x, w, None, self.stride, 0, self.dilation, groups=self.groups)

    def update_average_activity(self, y):
        """Update the average activity of neurons."""
        current_activity = y.mean(dim=(0, 2, 3))
        self.average_activity = 0.9 * self.average_activity + 0.1 * current_activity

    def synaptic_scaling(self):
        """Apply synaptic scaling to maintain target activity."""
        scale_factor = self.target_activity / (self.average_activity + 1e-6)
        self.weight.data *= (1 + self.scaling_rate * (scale_factor - 1)).view(-1, 1, 1, 1)
        self.weight.data = F.normalize(self.weight.data, p=2, dim=(1, 2, 3))

    def structural_plasticity(self):
        """Prune sparse weights and create new ones."""
        with torch.no_grad():
            # Pruning step
            prune_threshold = torch.quantile(torch.abs(self.weight), self.prune_threshold_percentile / 100)
            weak_synapses = torch.abs(self.weight) < prune_threshold
            self.weight.data[weak_synapses] = 0
            # Growth step
            zero_weights = self.weight.data == 0
            new_synapses = torch.rand_like(self.weight) < self.growth_probability
            new_synapses &= zero_weights
            self.weight.data[new_synapses] = torch.randn_like(self.weight)[new_synapses] * self.new_synapse_strength

    def cosine(self, x, w):
        """Compute cosine similarity between input and weights."""
        w_normalized = F.normalize(w, p=2, dim=1)
        # conv_output = symmetric_pad(x, self.padding)
        conv_output = F.conv2d(x, w_normalized, None, self.stride, 0, self.dilation, groups=self.groups)
        x_squared = x.pow(2)
        x_squared_sum = F.conv2d(x_squared, torch.ones_like(w), None, self.stride, 0, self.dilation,
                                 self.groups)
        x_norm = torch.sqrt(x_squared_sum + 1e-8)
        cosine_sim = conv_output / x_norm
        return cosine_sim

    def apply_surround_modulation(self, y):
        """Apply surround modulation to the output."""
        return F.conv2d(y, self.sm_kernel.repeat(self.out_channels, 1, 1, 1),
                        padding=self.sm_kernel.size(-1) // 2, groups=self.out_channels)

    def compute_activation(self, x):
        """Modified activation computation based on configuration"""
        x = symmetric_pad(x, self.padding)
        w = self.weight
        if self.dale:
            w = w.abs()
        if self.w_nrm:
            w = normalize(w, dim=(1, 2, 3))
        if self.use_presynaptic_competition:
            w = self.compute_presynaptic_competition_spatial(w)
        if self.use_cosine_similarity:
            y = self.cosine(x, w)
        else:
            y = self.apply_weights(x, w)
            y = self.act(y)
        return x, y, w

    def forward(self, x):
        """Modified forward pass to use configured features"""
        x, y, w = self.compute_activation(x)
        # Apply lateral inhibition if enabled
        if self.use_lateral_inhibition and self.kernel != 1:
            y = self.apply_surround_modulation(y)
        if self.training:
            # if self.use_homeostasis:
            #     self.update_average_activity(y)
            if self.use_structural_plasticity:
                self.structural_plasticity()
            self.compute_update(x, y, w)
        return y

    def compute_update(self, x, y, weight):
        """Compute weight updates based on the chosen learning mode."""
        if self.mode == self.MODE_BASIC_HEBBIAN:
            update = self.update_basic_hebbian(x, y, weight)
        elif self.mode == self.MODE_HARDWT:
            update = self.update_hardwt(x, y, weight)
        elif self.mode == self.MODE_SOFTWTA:
            update = self.update_softwta(x, y, weight)
        elif self.mode == self.MODE_ANTIHARDWT:
            update = self.update_antihardwt(x, y, weight)
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
        """Implement basic Hebbian learning (Grossberg Instar rule)."""
        yx = self.compute_yx(x, y)
        y_sum = y.sum(dim=(0, 2, 3)).view(-1, 1, 1, 1)
        yw = y_sum * weight
        update = yx - yw
        return update

    def update_hardwt(self, x, y, weight):
        """Implement hard Winner-Take-All (WTA) Hebbian learning."""
        y_wta = y * self.compute_wta_mask(y)
        yx = self.compute_yx(x, y_wta)
        yu = torch.sum(y_wta, dim=(0, 2, 3))
        update = yx - yu.view(-1, 1, 1, 1) * weight
        return update

    def update_softwta(self, x, y, weight):
        """Implement soft Winner-Take-All Hebbian learning (SoftHebb)."""
        softwta_activs = self.compute_softwta_activations(y)
        yx = self.compute_yx(x, softwta_activs)
        yu = torch.sum(torch.mul(softwta_activs, y), dim=(0, 2, 3))
        update = yx - yu.view(-1, 1, 1, 1) * weight
        return update

    def update_antihardwt(self, x, y, weight):
        """Implement anti-Hebbian hard Winner-Take-All learning."""
        hardwta_activs = self.compute_antihardwta_activations(y)
        yx = self.compute_yx(x, hardwta_activs)
        yu = torch.sum(torch.mul(hardwta_activs, y), dim=(0, 2, 3))
        update = yx - yu.view(-1, 1, 1, 1) * weight
        return update

    def update_bcm(self, x, y, weight):
        """Implement BCM (Bienenstock-Cooper-Munro) learning rule."""
        y_wta = y * self.compute_wta_mask(y)
        y_squared = y_wta.pow(2).mean(dim=(0, 2, 3))
        self.theta.data = (1 - self.theta_decay) * self.theta + self.theta_decay * y_squared
        y_minus_theta = y_wta - self.theta.view(1, -1, 1, 1)
        bcm_factor = y_wta * y_minus_theta
        yx = self.compute_yx(x, bcm_factor)
        update = yx.view(weight.shape)
        return update

    def update_temporal_competition(self, x, y, weight):
        """Implement temporal competition-based learning."""
        self.update_activation_history(y)
        temporal_winners = self.compute_temporal_winners(y)
        y_winners = temporal_winners * y
        y_winners = y_winners * self.apply_competition(y_winners)
        yx = self.compute_yx(x, y_winners)
        y_sum = y_winners.sum(dim=(0, 2, 3)).view(-1, 1, 1, 1)
        update = yx - y_sum * weight
        return update

    def update_adaptive_threshold(self, x, y, weight):
        """Implement adaptive/statistics threshold-based learning."""
        batch_size, out_channels, height_out, width_out = y.shape
        similarities = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups)
        similarities = similarities / (torch.norm(weight.view(out_channels, -1), dim=1).view(1, -1, 1, 1) + 1e-10)
        threshold = self.compute_adaptive_threshold(similarities)
        winners = (similarities > threshold).float()
        y_winners = winners * similarities
        y_winners = y_winners * self.apply_competition(y_winners)
        yx = self.compute_yx(x, y_winners)
        y_sum = y_winners.sum(dim=(0, 2, 3)).view(-1, 1, 1, 1)
        update = yx - y_sum * weight
        return update

    def update_activation_history(self, y):
        """Update the activation history for temporal competition."""
        if self.activation_history is None:
            self.activation_history = y.detach().clone()
        else:
            self.activation_history = torch.cat([self.activation_history, y.detach()], dim=0)
            if self.activation_history.size(0) > self.temporal_window:
                self.activation_history = self.activation_history[-self.temporal_window:]

    def compute_temporal_winners(self, y):
        """Select winners of temporal competition."""
        batch_size, out_channels, height_out, width_out = y.shape
        history_spatial = self.activation_history.view(-1, out_channels, height_out, width_out)
        median_activations = torch.median(history_spatial, dim=0)[0]
        # Compute threshold for each spatial location
        temporal_threshold = torch.mean(median_activations, dim=0, keepdim=True)
        return (y > temporal_threshold).float()

    def compute_adaptive_threshold(self, similarities):
        """Create threshold for statistical thresholding."""
        mean_sim = similarities.mean(dim=1, keepdim=True)
        std_sim = similarities.std(dim=1, keepdim=True)
        return mean_sim + self.competition_k * std_sim

    def apply_competition(self, y):
        # Hard-WTA or Soft-WTA competition mask for temporal/statistical thresholding
        batch_size, out_channels, height, width = y.shape
        if self.mode in [self.MODE_TEMPORAL_COMPETITION, self.MODE_ADAPTIVE_THRESHOLD]:
            if self.competition_type == 'hard':
                y = y.view(batch_size, out_channels, -1)
                top_k_values, top_k_indices = torch.topk(y, self.top_k, dim=1, largest=True, sorted=False)
                y_compete = torch.zeros_like(y)
                y_compete.scatter_(1, top_k_indices, top_k_values)
                return y_compete.view(batch_size, out_channels, height, width)
            elif self.competition_type == 'soft':
                y_flat = y.view(batch_size, out_channels, -1)
                y_soft = torch.softmax(self.t_invert * y_flat, dim=1)
                return y_soft.view(batch_size, out_channels, height, width)
        return y

    def compute_yx(self, x, y):
        """Compute yx term from Grossberg Instar rule"""
        yx = F.conv2d(x.transpose(0, 1), y.transpose(0, 1), padding=0,
                      stride=self.dilation, dilation=self.stride).transpose(0, 1)
        if self.groups != 1:
            yx = yx.mean(dim=1, keepdim=True)
        return yx

    def compute_wta_mask(self, y):
        """Compute Hard-WTA mask."""
        batch_size, out_channels, height_out, width_out = y.shape
        y_flat = y.transpose(0, 1).reshape(out_channels, -1)
        win_neurons = torch.argmax(y_flat, dim=0)
        wta_mask = F.one_hot(win_neurons, num_classes=out_channels).float()
        mask = wta_mask.transpose(0, 1).view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
        return mask

    def compute_homeostatic_wta_mask(self, y):
        """Original WTA mask computation with homeostasis"""
        batch_size, out_channels, height_out, width_out = y.shape
        # Subtract homeostasis before WTA competition
        y_adjusted = y - self.H.view(1, -1, 1, 1)
        # Use the original working WTA logic
        y_flat = y_adjusted.transpose(0, 1).reshape(out_channels, -1)
        win_neurons = torch.argmax(y_flat, dim=0)
        wta_mask = F.one_hot(win_neurons, num_classes=out_channels).float()
        wta_mask = wta_mask.transpose(0, 1).view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
        # Update homeostasis statistics
        with torch.no_grad():
            current_activity = wta_mask.mean(dim=(0, 2, 3))
            self.avg_activity = 0.9 * self.avg_activity + 0.1 * current_activity
            self.H += self.gamma * (self.avg_activity - self.target)
        return wta_mask

    def compute_softwta_activations(self, y):
        """Computes SoftHebb mask and Hebb/AntiHebb implementation."""
        batch_size, out_channels, height_out, width_out = y.shape
        flat_weighted_inputs = y.transpose(0, 1).reshape(out_channels, -1)
        flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)
        flat_softwta_activs = -flat_softwta_activs
        win_neurons = torch.argmax(flat_weighted_inputs, dim=0)
        competing_idx = torch.arange(flat_weighted_inputs.size(1))
        flat_softwta_activs[win_neurons, competing_idx] = -flat_softwta_activs[win_neurons, competing_idx]
        return flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)

    def compute_antihardwta_activations(self, y):
        # Computes Hard-WTA mask and Hebb/AntiHebb implementation.
        batch_size, out_channels, height_out, width_out = y.shape
        flat_weighted_inputs = y.transpose(0, 1).reshape(out_channels, -1)
        win_neurons = torch.argmax(flat_weighted_inputs, dim=0)
        competing_idx = torch.arange(flat_weighted_inputs.size(1))
        anti_hebbian_mask = torch.ones_like(flat_weighted_inputs) * -1
        anti_hebbian_mask[win_neurons, competing_idx] = 1
        flat_hardwta_activs = flat_weighted_inputs * anti_hebbian_mask
        return flat_hardwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)

    def compute_presynaptic_competition(self, m):
        # This presynaptic competition promotes diversity among output channels.
        m = 1 / (torch.abs(m) + 1e-6)
        if self.presynaptic_competition_type == 'linear':
            return m / (m.sum(dim=0, keepdim=True) + 1e-6)
        elif self.presynaptic_competition_type == 'softmax':
            return F.softmax(m, dim=0)
        elif self.presynaptic_competition_type == 'lp_norm':
            return F.normalize(m, p=2, dim=0)
        else:
            raise ValueError(f"Unknown competition type: {self.competition_type}")

    def compute_presynaptic_competition_spatial(self, m):
        # The presynaptic spatial competition encourages each input-output channel pair.
        m = 1 / (torch.abs(m) + 1e-6)
        if self.presynaptic_competition_type == 'linear':
            # Sum across spatial dimensions (last two dimensions)
            return m / (m.sum(dim=(-2, -1), keepdim=True) + 1e-6)
        elif self.presynaptic_competition_type == 'softmax':
            # Reshape to combine spatial dimensions
            shape = m.shape
            m_flat = m.view(*shape[:-2], -1)
            # Apply softmax across spatial dimensions
            m_comp = F.softmax(m_flat, dim=-1)
            # Reshape back to original shape
            return m_comp.view(*shape)
        elif self.presynaptic_competition_type == 'lp_norm':
            # Normalize across spatial dimensions
            return F.normalize(m, p=2, dim=(-2, -1))
        else:
            raise ValueError(f"Unknown competition type: {self.presynaptic_competition_type}")

    def compute_presynaptic_competition_input_channels(self, m):
        # The presynaptic input competition promotes specialisation of each output channel across input features.
        m = 1 / (torch.abs(m) + 1e-6)
        if self.presynaptic_competition_type == 'linear':
            # Sum across input channel dimension
            return m / (m.sum(dim=1, keepdim=True) + 1e-6)
        elif self.presynaptic_competition_type == 'softmax':
            # Apply softmax across input channels
            return F.softmax(m, dim=1)
        elif self.presynaptic_competition_type == 'lp_norm':
            # Normalize across input channels
            return F.normalize(m, p=2, dim=1)
        else:
            raise ValueError(f"Unknown competition type: {self.presynaptic_competition_type}")

    def compute_presynaptic_competition_global(self, m):
        # The global competition creates a more intense competition where every weight competes with all others,
        # potentially leading to very sparse but highly specialised connections.
        m = 1 / (torch.abs(m) + 1e-6)
        if self.presynaptic_competition_type == 'linear':
            # Global sum across all dimensions
            return m / (m.sum() + 1e-6)
        elif self.presynaptic_competition_type == 'softmax':
            # Flatten and apply softmax globally
            m_flat = m.view(-1)
            return F.softmax(m_flat, dim=0).view(m.shape)
        elif self.presynaptic_competition_type == 'lp_norm':
            # Global normalization
            return F.normalize(m.view(-1), p=2).view(m.shape)
        else:
            raise ValueError(f"Unknown competition type: {self.presynaptic_competition_type}")

    #
    # @torch.no_grad()
    # def local_update(self):
    #     """
    # 	This function transfers a previously computed weight update, stored in buffer self.delta_w, to the gradient
    # 	self.weight.grad of the weight parameter.
    #
    # 	This function should be called before optimizer.step(), so that the optimizer will use the locally computed
    # 	update as optimization direction. Local updates can also be combined with end-to-end updates by calling this
    # 	function between loss.backward() and optimizer.step(). loss.backward will store the end-to-end gradient in
    # 	self.weight.grad, and this function combines this value with self.delta_w as
    # 	self.weight.grad = (1 - alpha) * self.weight.grad - alpha * self.delta_w
    # 	Parameter alpha determines the scale of the local update compared to the end-to-end gradient in the combination.
    # 	"""
    #     if self.weight.grad is None:
    #         self.weight.grad = -self.alpha * self.delta_w
    #     else:
    #         self.weight.grad = (1 - self.alpha) * self.weight.grad - self.alpha * self.delta_w
    #     self.delta_w.zero_()

    # Update method to modify weights without requiring an optimiser
    @torch.no_grad()
    # Weight Update
    def local_update(self):
        new_weight = self.weight + self.lr * self.alpha * self.delta_w
        # Update weights
        if self.dale:
            self.weight.copy_(new_weight.abs())
        else:
            self.weight.copy_(new_weight)
        self.delta_w.zero_()
