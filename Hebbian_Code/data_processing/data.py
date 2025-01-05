import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, STL10, MNIST
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# torch.manual_seed(0)

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
        self.images = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_path = self.root_dir / class_name
            if not class_path.exists():
                raise ValueError(f"Class directory {class_name} not found in {root_dir}")

            for img_path in class_path.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # grayscale
        # image = Image.open(img_path).convert('RGB')  # Convert to RGB
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class ZCAWhitening:
    def __init__(self, epsilon=1e-1):
        self.epsilon = epsilon
        self.zca_matrix = None
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor, transpose=True, dataset: str = "CIFAR10"):
        path = os.path.join("../zca_data", dataset, f"{dataset}_zca.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            saved_data = torch.load(path, map_location='cpu')
            self.zca_matrix = saved_data['zca']
            self.mean = saved_data['mean']
            self.std = saved_data['std']
            print(f"Loaded pre-computed ZCA matrix for {dataset}")
        except FileNotFoundError:
            print(f"Computing ZCA matrix for {dataset}")
            if transpose and x.dim() == 4:
                # Handle both RGB and grayscale images
                if x.shape[1] == 1 or x.shape[-1] == 1:  # Grayscale
                    x = x.squeeze(-1) if x.shape[-1] == 1 else x.squeeze(1)
                    x = x.unsqueeze(-1) if transpose else x.unsqueeze(1)
                x = x.permute(0, 3, 1, 2) if transpose else x

            x = x.reshape(x.shape[0], -1)
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True)
            x = (x - self.mean) / (self.std + self.epsilon)

            cov = torch.mm(x.T, x) / (x.shape[0] - 1)

            if torch.isnan(cov).any() or torch.isinf(cov).any():
                print("Warning: NaN or inf values detected in covariance matrix")
                cov = torch.where(torch.isnan(cov) | torch.isinf(cov), torch.zeros_like(cov), cov)

            # Use eigendecomposition for all datasets
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            inv_sqrt_eigenvalues = torch.diag(1.0 / torch.sqrt(eigenvalues + self.epsilon))
            self.zca_matrix = torch.mm(torch.mm(eigenvectors, inv_sqrt_eigenvalues), eigenvectors.T)

            torch.save({'zca': self.zca_matrix, 'mean': self.mean, 'std': self.std}, path)
            print(f"Saved computed ZCA matrix for {dataset}")

    def transform(self, x: torch.Tensor):
        if self.zca_matrix is None:
            raise ValueError("ZCA matrix not computed. Call fit() first.")

        original_shape = x.shape
        is_grayscale = original_shape[1] == 1  # Check if the input is grayscale

        x = x.reshape(x.shape[0], -1)
        x = (x - self.mean) / (self.std + self.epsilon)
        x_whitened = torch.mm(x, self.zca_matrix)

        # Reshape back to original dimensions
        transformed = x_whitened.reshape(original_shape)

        # Ensure the channel dimension is in the correct position
        if is_grayscale and len(original_shape) == 4:
            transformed = transformed.unsqueeze(1) if transformed.shape[1] != 1 else transformed
        return transformed


class ZCATransformation:
    def __init__(self, zca):
        self.zca = zca

    def __call__(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x_whitened = self.zca.transform(x)
        return x_whitened.squeeze(0) if x_whitened.shape[0] == 1 else x_whitened


class BatchZCAWhitening:
    def __init__(self, epsilon=1e-1):
        self.epsilon = epsilon
        self.zca_matrix = None
        self.mean = None
        self.std = None

    def fit(self, dataloader: DataLoader, transpose=True, dataset: str = "CIFAR10"):
        path = os.path.join("../zca_data", dataset, f"{dataset}_batch_zca.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            saved_data = torch.load(path, map_location='cpu')
            self.zca_matrix = saved_data['zca']
            self.mean = saved_data['mean']
            self.std = saved_data['std']
            print(f"Loaded pre-computed batch ZCA matrix for {dataset}")
            return
        except FileNotFoundError:
            print(f"Computing batch ZCA matrix for {dataset}")

        # First pass: compute mean and variance
        total_samples = 0
        sum_x = 0
        sum_x_sq = 0

        print(f'Processing Training batches: {len(dataloader)}')
        for i, batch in enumerate(dataloader):
            print(f'Batch: [{i}]')

            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            if transpose and batch.dim() == 4:
                if batch.shape[1] == 1 or batch.shape[-1] == 1:
                    batch = batch.squeeze(-1) if batch.shape[-1] == 1 else batch.squeeze(1)
                    batch = batch.unsqueeze(-1) if transpose else batch.unsqueeze(1)
                batch = batch.permute(0, 3, 1, 2) if transpose else batch

            # Match original normalization
            batch = batch.float() / 255.0
            batch = batch.reshape(batch.shape[0], -1)

            sum_x += batch.sum(dim=0)
            sum_x_sq += torch.mm(batch.t(), batch)
            total_samples += batch.shape[0]

        # Compute mean
        self.mean = sum_x / total_samples

        # Compute std using n-1 in denominator to match torch.std()
        mean_sq = torch.mm(self.mean.unsqueeze(1), self.mean.unsqueeze(0))
        self.std = torch.sqrt((sum_x_sq - total_samples * mean_sq) / (total_samples - 1))
        self.std = torch.diag(self.std)

        # Second pass: compute covariance on normalized data
        sum_normalized_sq = 0
        for i, batch in enumerate(dataloader):
            print(f'Batch: [{i}]')
            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            if transpose and batch.dim() == 4:
                if batch.shape[1] == 1 or batch.shape[-1] == 1:
                    batch = batch.squeeze(-1) if batch.shape[-1] == 1 else batch.squeeze(1)
                    batch = batch.unsqueeze(-1) if transpose else batch.unsqueeze(1)
                batch = batch.permute(0, 3, 1, 2) if transpose else batch

            # Match original normalization
            batch = batch.float() / 255.0
            batch = batch.reshape(batch.shape[0], -1)

            # Normalize exactly as in original
            normalized_batch = (batch - self.mean) / (self.std + self.epsilon)
            sum_normalized_sq += torch.mm(normalized_batch.t(), normalized_batch)

        # Compute covariance matrix using n-1
        covariance = sum_normalized_sq / (total_samples - 1)
        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        # Create ZCA matrix exactly as in original
        inv_sqrt_eigenvalues = torch.diag(1.0 / torch.sqrt(eigenvalues + self.epsilon))
        self.zca_matrix = torch.mm(torch.mm(eigenvectors, inv_sqrt_eigenvalues), eigenvectors.t())
        torch.save({
            'zca': self.zca_matrix,
            'mean': self.mean,
            'std': self.std
        }, path)
        print(f"Saved computed batch ZCA matrix for {dataset}")

    def transform(self, x: torch.Tensor):
        if self.zca_matrix is None:
            raise ValueError("ZCA matrix not computed. Call fit() first.")
        original_shape = x.shape
        is_grayscale = original_shape[1] == 1
        # Match original normalization
        x = x.float() / 255.0
        x = x.reshape(x.shape[0], -1)
        # Normalize and transform exactly as in original
        x = (x - self.mean) / (self.std + self.epsilon)
        x_whitened = torch.mm(x, self.zca_matrix)
        transformed = x_whitened.reshape(original_shape)
        if is_grayscale and len(original_shape) == 4:
            transformed = transformed.unsqueeze(1) if transformed.shape[1] != 1 else transformed
        return transformed


class BlockwiseZCA:
    def __init__(self, block_size=8, stride=None, epsilon=1e-5, overlap=True):
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size // 2 if overlap else block_size
        self.epsilon = epsilon
        self.zca_matrices = {}  # Store ZCA matrix for each unique block pattern
        self.mean = None
        self.std = None

    def _extract_blocks(self, x):
        """Extract overlapping or non-overlapping blocks from batch of images"""
        b, c, h, w = x.shape
        blocks = F.unfold(x, kernel_size=self.block_size, stride=self.stride)
        # Reshape to [batch * num_blocks, channels * block_size * block_size]
        blocks = blocks.transpose(1, 2).reshape(-1, c * self.block_size * self.block_size)
        return blocks

    def _reconstruct_from_blocks(self, blocks, original_size):
        """Reconstruct image from processed blocks"""
        b, c, h, w = original_size
        # Reshape blocks back to unfold format
        blocks = blocks.reshape(b, -1, c * self.block_size * self.block_size).transpose(1, 2)
        # Use fold operation to reconstruct image
        output = F.fold(
            blocks,
            output_size=(h, w),
            kernel_size=self.block_size,
            stride=self.stride
        )

        # Normalize for overlapping blocks
        if self.stride < self.block_size:
            divisor = F.fold(
                torch.ones_like(blocks),
                output_size=(h, w),
                kernel_size=self.block_size,
                stride=self.stride
            )
            output = output / (divisor + self.epsilon)

        return output

    def fit(self, dataloader: DataLoader, transpose=True, dataset: str = "CIFAR10"):
        """Compute ZCA statistics for image blocks"""
        path = os.path.join("../zca_data", dataset, f"{dataset}_block_zca_{self.block_size}_{self.stride}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            saved_data = torch.load(path, map_location='cpu')
            self.zca_matrices = saved_data['zca']
            self.mean = saved_data['mean']
            self.std = saved_data['std']
            print(f"Loaded pre-computed block ZCA matrices for {dataset}")
            return
        except FileNotFoundError:
            print(f"Computing block ZCA matrices for {dataset}")

        # Accumulate statistics for blocks
        total_blocks = 0
        sum_blocks = 0
        sum_blocks_sq = 0

        print(f'Processing training batches: {len(dataloader)}')
        for i, batch in enumerate(dataloader):
            print(f'Batch: [{i}]')
            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            # Extract blocks from batch
            blocks = self._extract_blocks(batch)

            # Accumulate statistics
            sum_blocks += blocks.sum(dim=0)
            sum_blocks_sq += (blocks ** 2).sum(dim=0)
            total_blocks += blocks.shape[0]

        # Compute mean and std of blocks
        self.mean = sum_blocks / total_blocks
        self.std = torch.sqrt((sum_blocks_sq / total_blocks) - self.mean ** 2)

        # Compute ZCA matrix for standardized blocks
        print("Computing ZCA matrix for blocks...")
        standardized_blocks = (blocks - self.mean) / (self.std + self.epsilon)
        cov = torch.mm(standardized_blocks.T, standardized_blocks) / (blocks.shape[0] - 1)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Create ZCA matrix
        inv_sqrt_eigenvalues = torch.diag(1.0 / torch.sqrt(eigenvalues + self.epsilon))
        self.zca_matrices['default'] = torch.mm(
            torch.mm(eigenvectors, inv_sqrt_eigenvalues),
            eigenvectors.T
        )

        # Save computed matrices
        torch.save({
            'zca': self.zca_matrices,
            'mean': self.mean,
            'std': self.std
        }, path)
        print(f"Saved computed block ZCA matrices for {dataset}")

    def transform(self, x: torch.Tensor):
        """Apply ZCA whitening to blocks and reconstruct image"""
        if not self.zca_matrices:
            raise ValueError("ZCA matrices not computed. Call fit() first.")

        original_shape = x.shape
        # Extract blocks
        blocks = self._extract_blocks(x)
        # Standardize blocks
        blocks = (blocks - self.mean) / (self.std + self.epsilon)
        # Apply ZCA whitening to blocks
        whitened_blocks = torch.mm(blocks, self.zca_matrices['default'])
        # Reconstruct image from whitened blocks
        transformed = self._reconstruct_from_blocks(whitened_blocks, original_shape)

        return transformed

class BatchZCATransformation:
    """Transformation wrapper for batch ZCA whitening"""

    def __init__(self, zca):
        self.zca = zca

    def __call__(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x_whitened = self.zca.transform(x)
        return x_whitened.squeeze(0) if x_whitened.shape[0] == 1 else x_whitened

class BioDecorrelation:
    def __init__(self, decay_rate=0.1, inhibition_strength=0.1, neighborhood_size=5):
        self.decay_rate = decay_rate
        self.inhibition_strength = inhibition_strength
        self.activity_trace = None
        self.inhibition_weights = None
        self.mean = None
        self.std = None
        self.neighborhood_size = neighborhood_size

    def _create_inhibition_weights(self, size):
        """Create sharper inhibition profile with controlled neighborhood"""
        positions = torch.arange(size)
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()

        # Create sharper inhibition profile
        weights = torch.zeros_like(distances)
        weights[distances <= self.neighborhood_size] = 1.0
        weights[distances == 0] = 0  # No self-inhibition

        # Normalize weights
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return weights * self.inhibition_strength

    def fit(self, dataloader: DataLoader, transpose=True, dataset: str = "CIFAR10"):
        """Initialize activity traces and inhibition patterns from dataset"""
        path = os.path.join("bio_data", dataset, f"{dataset}_bio_decorr.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            saved_data = torch.load(path, map_location='cpu')
            self.activity_trace = saved_data['activity']
            self.inhibition_weights = saved_data['inhibition']
            self.mean = saved_data['mean']
            self.std = saved_data['std']
            print(f"Loaded pre-computed bio-decorrelation data for {dataset}")
            return
        except FileNotFoundError:
            print(f"Computing bio-decorrelation data for {dataset}")

        # First pass: compute statistics and initialize traces
        total_samples = 0
        sum_x = 0
        sum_x_sq = 0
        activity_sum = 0

        print(f'Processing batches: {len(dataloader)}')
        for i, batch in enumerate(dataloader):
            print(f'Batch: [{i}]')
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.float() / 255.0
            batch = batch.reshape(batch.shape[0], -1)
            sum_x += batch.sum(dim=0)
            sum_x_sq += (batch ** 2).sum(dim=0)
            activity_sum += batch.mean(dim=0)
            total_samples += batch.shape[0]

        # Compute statistics
        self.mean = sum_x / total_samples
        self.std = torch.sqrt((sum_x_sq / total_samples) - self.mean ** 2)
        # Initialize activity trace
        self.activity_trace = activity_sum / len(dataloader)
        # Initialize inhibition weights with Mexican hat profile
        feature_dim = self.mean.shape[0]
        self.inhibition_weights = self._create_inhibition_weights(feature_dim)
        # Save computed data
        torch.save({
            'activity': self.activity_trace,
            'inhibition': self.inhibition_weights,
            'mean': self.mean,
            'std': self.std,
        }, path)
        print(f"Saved computed bio-decorrelation data for {dataset}")

    def transform(self, x: torch.Tensor):
        """Apply enhanced decorrelation with stronger local competition"""
        if self.activity_trace is None:
            raise ValueError("Decorrelation not initialized. Call fit() first.")
        original_shape = x.shape
        x = x.float() / 255.0
        x = x.reshape(x.shape[0], -1)
        # Center and scale
        x = (x - self.mean) / (self.std + 1e-3)
        # Apply stronger local inhibition
        inhibition = torch.mm(x, self.inhibition_weights)
        x_decorrelated = x - inhibition
        # Normalize response to maintain signal strength
        x_decorrelated = F.layer_norm(x_decorrelated, x_decorrelated.shape[1:])
        # Update activity trace with current batch
        with torch.no_grad():
            current_activity = x_decorrelated.mean(dim=0)
            self.activity_trace = (1 - self.decay_rate) * self.activity_trace + \
                                  self.decay_rate * current_activity
        # Reshape to original dimensions
        transformed = x_decorrelated.reshape(original_shape)
        return transformed

class BioDecorrelationTransform:
    def __init__(self, decorrelator):
        self.decorrelator = decorrelator

    def __call__(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x_decorrelated = self.decorrelator.transform(x)
        return x_decorrelated.squeeze(0) if x_decorrelated.shape[0] == 1 else x_decorrelated


def visualize_zca_effect(original_data, whitened_data, num_samples=5):
    # print(torch.isnan(whitened_data))
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    for i in range(num_samples):
        # Check if the image is grayscale (1 channel) or RGB (3 channels)
        is_grayscale = original_data[i].shape[0] == 1
        if is_grayscale:
            # For grayscale images, squeeze out the channel dimension
            orig = original_data[i].squeeze(0)
            whit = whitened_data[i].squeeze(0)
        else:
            # For RGB images, permute as before
            orig = original_data[i].permute(1, 2, 0)
            whit = whitened_data[i].permute(1, 2, 0)
        # Normalize to [0, 1] range
        orig = orig.cpu().numpy()
        whit = whit.cpu().numpy()
        orig = (orig - orig.min()) / (orig.max() - orig.min())
        whit = (whit - whit.min()) / (whit.max() - whit.min())
        # Use cmap='gray' for grayscale images
        axes[0, i].imshow(orig, cmap='gray' if is_grayscale else None)
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        axes[1, i].imshow(whit, cmap='gray' if is_grayscale else None)
        axes[1, i].axis('off')
        axes[1, i].set_title('Whitened')
    plt.tight_layout()
    plt.show()


def visualize_decorrelation_effect(original_data, decorrelated_data, num_samples=5):
    """Visualize the effect of bio-inspired decorrelation"""
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))

    for i in range(num_samples):
        # Original image
        is_grayscale = original_data[i].shape[0] == 1
        orig = original_data[i].squeeze() if is_grayscale else original_data[i].permute(1, 2, 0)
        orig = orig.cpu().numpy()

        # Decorrelated image
        decorr = decorrelated_data[i].squeeze() if is_grayscale else decorrelated_data[i].permute(1, 2, 0)
        decorr = decorr.cpu().numpy()

        # Difference map
        diff = np.abs(orig - decorr)

        # Normalize for visualization
        orig = (orig - orig.min()) / (orig.max() - orig.min())
        decorr = (decorr - decorr.min()) / (decorr.max() - decorr.min())
        diff = (diff - diff.min()) / (diff.max() - diff.min())

        # Plot
        axes[0, i].imshow(orig, cmap='gray' if is_grayscale else None)
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')

        axes[1, i].imshow(decorr, cmap='gray' if is_grayscale else None)
        axes[1, i].axis('off')
        axes[1, i].set_title('Decorrelated')

        axes[2, i].imshow(diff, cmap='hot')
        axes[2, i].axis('off')
        axes[2, i].set_title('Difference')

    plt.tight_layout()
    plt.show()

def check_covariance(data):
    data_flat = data.reshape(data.shape[0], -1)
    cov_matrix = torch.cov(data_flat.T)
    diag_mean = cov_matrix.diag().mean().item()
    off_diag_mean = (cov_matrix - torch.diag(cov_matrix.diag())).abs().mean().item()
    print(f"Mean of diagonal elements: {diag_mean:.6f}")
    print(f"Mean of off-diagonal elements: {off_diag_mean:.6f}")
    print(f"Ratio of off-diagonal to diagonal: {off_diag_mean / diag_mean:.6f}")

def check_normalization(data):
    data_flat = data.reshape(data.shape[0], -1)
    mean = data_flat.mean().item()
    std = data_flat.std().item()
    print(f"Mean: {mean:.6f}")
    print(f"Standard deviation: {std:.6f}")
    print(f"Min: {data_flat.min().item():.6f}")
    print(f"Max: {data_flat.max().item():.6f}")
    print()

def get_data(dataset='cifar10', root='datasets', batch_size=32, num_workers=0, whiten_lvl=None):
    trn_set, tst_set = None, None

    if dataset == 'cifar10':
        trn_set = CIFAR10(root=os.path.join(root, dataset), train=True, download=True, transform=T.ToTensor())
        tst_set = CIFAR10(root=os.path.join(root, dataset), train=False, download=True, transform=T.ToTensor())
        # all_data = torch.cat([torch.tensor(trn_set.data), torch.tensor(tst_set.data)], dim=0)
        # all_data = all_data.float() / 255.0
        img_size = 32

    elif dataset == 'stl10':
        trn_set = STL10(root=os.path.join(root, dataset), split='train', download=True, transform=T.ToTensor())
        tst_set = STL10(root=os.path.join(root, dataset), split='test', download=True, transform=T.ToTensor())
        # all_data = torch.cat([trn_sample[0].unsqueeze(0) for trn_sample in trn_set] +
        #                      [tst_sample[0].unsqueeze(0) for tst_sample in tst_set], dim=0)
        img_size = 96

    elif dataset == 'mnist':
        trn_set = MNIST(root=os.path.join(root, dataset), train=True, download=True, transform=T.ToTensor())
        tst_set = MNIST(root=os.path.join(root, dataset), train=False, download=True, transform=T.ToTensor())
        # all_data = torch.cat([trn_sample[0].unsqueeze(0) for trn_sample in trn_set] +
        #                      [tst_sample[0].unsqueeze(0) for tst_sample in tst_set], dim=0)
        img_size = 28

    elif dataset == 'brain_tumor':
        basic_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        # Assuming the Brain Tumor dataset is organized in folders by class
        trn_set = BrainTumorDataset(os.path.join(root, 'brain_tumor', 'train'), transform=basic_transform)
        tst_set = BrainTumorDataset(os.path.join(root, 'brain_tumor', 'test'), transform=basic_transform)
        # all_data = torch.stack([sample[0] for sample in trn_set] + [sample[0] for sample in tst_set])
        img_size = 256  # Standard size for medical images, adjust as needed

    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")

    zca = None
    if whiten_lvl is not None:
        # print("Data whitening")
        # zca = ZCAWhitening(epsilon=whiten_lvl)
        # zca.fit(all_data, transpose=False, dataset=dataset)

        print("Data Batch whitening")
        # Use BatchZCAWhitening instead of ZCAWhitening
        zca = ZCAWhitening(epsilon=whiten_lvl)
        # zca = BatchZCAWhitening(epsilon=1e-3)
        # zca = BioDecorrelation(decay_rate=0.5, inhibition_strength=0.5)
        # zca = BlockwiseZCA(block_size=5, stride=1, epsilon=whiten_lvl)
        # Create a temporary dataloader for fitting ZCA
        # Combine training and test sets for ZCA fitting
        combined_dataset = torch.utils.data.ConcatDataset([trn_set, tst_set])
        temp_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        # Fit ZCA using the batch-based approach
        zca.fit(temp_loader, transpose=False, dataset=dataset)

        is_grayscale = dataset in ['mnist', 'brain_tumor']
        if is_grayscale:
            basic_transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])
            full_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                # BatchRandomProjectionTransform(zca)
            ])
        else:
            # RGB transforms remain the same as in your original code
            basic_transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor()
            ])
            full_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                BatchZCATransformation(zca),
                # BioDecorrelationTransform(zca)
            ])

        temp_transform = T.Compose([T.Resize(img_size), T.ToTensor()])
        if dataset == "cifar10":
            temp_dataset = CIFAR10(root=os.path.join(root, dataset), train=True, download=False,
            transform=temp_transform)
        elif dataset == "stl10":
            temp_dataset = STL10(root=os.path.join(root, dataset), split='train', download=False,
                                 transform=temp_transform)
        elif dataset == "mnist":
            temp_dataset = MNIST(root=os.path.join(root, dataset), train=True, download=False,
            transform=temp_transform)
        elif dataset == "brain_tumor":
            temp_dataset = BrainTumorDataset(os.path.join(root, 'brain_tumor', 'train'), transform=temp_transform)
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # Get a batch for visualization
        original_batch, _ = next(iter(temp_loader))
        print(original_batch.shape)
        whitened_batch = zca.transform(original_batch)
        print(whitened_batch.shape)
        print("\nVisualization of ZCA effect:")
        visualize_zca_effect(original_batch, whitened_batch)
        visualize_decorrelation_effect(original_batch, whitened_batch)

        trn_set.transform = full_transform
        tst_set.transform = full_transform

    else:
        print("No ZCA")
        if dataset == 'cifar10':
            mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        elif dataset == 'stl10':
            mean, std = [0.4467, 0.4398, 0.4066], [0.2603, 0.2566, 0.2713]
        elif dataset == 'mnist':
            mean, std = [0.1307], [0.3081]
        elif dataset == 'brain_tumor':
            # mean, std = [0.1858, 0.1858, 0.1859], [0.1841, 0.1841, 0.1841]
            mean, std = [0.1858], [0.1841]


        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        trn_set.transform = transform
        tst_set.transform = transform

    trn_loader = DataLoader(trn_set, batch_size=batch_size, shuffle=True,
                            num_workers=8, persistent_workers=True)
    tst_loader = DataLoader(tst_set, batch_size=batch_size, shuffle=True,
                            num_workers=8, persistent_workers=True)

    return trn_loader, tst_loader, zca


# Example usage:
if __name__ == "__main__":
    # For CIFAR-10
    # cifar_trn_loader, cifar_tst_loader, cifar_zca = get_data(dataset='cifar10', batch_size=64, whiten_lvl=1e-3)

    brain_trn_loader, brain_tst_loader, brain_zca = get_data(dataset='brain_tumor', batch_size=64, whiten_lvl=1e-3)

    # # For MNIST
    # mnist_trn_loader, mnist_tst_loader, mnist_zca = get_data(dataset='mnist', batch_size=64, whiten_lvl=1e-3)
    # mnist_sample, _ = next(iter(mnist_trn_loader))
    # print(f"MNIST batch shape: {mnist_sample.shape}")
    # assert mnist_sample.shape == (64, 1, 28, 28), "MNIST whitening failed to produce correct shape."
