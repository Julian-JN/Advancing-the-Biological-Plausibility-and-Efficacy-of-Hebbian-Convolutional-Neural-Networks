import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import SpectralEmbedding, TSNE

# torch.manual_seed(0)

# Set global matplotlib parameters for LaTeX-style plots
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 20,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 20,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def print_weight_statistics(layer, layer_name):
    """
    Prints statistics of the weights from a given layer.
    Args:
    """
    weights = layer.weight.data

    print(f"Statistics for {layer_name}:")
    print(f"Mean: {weights.mean().item():.4f}")
    print(f"Max: {weights.max().item():.4f}")
    print(f"Min: {weights.min().item():.4f}")
    print(f"Standard Deviation: {weights.std().item():.4f}")
    print(f"Median: {weights.median().item():.4f}")
    print(f"25th Percentile: {weights.quantile(0.25).item():.4f}")
    print(f"75th Percentile: {weights.quantile(0.75).item():.4f}")
    print(f"Number of positive weights: {(weights >= 0).sum().item()}")
    print(f"Number of negative weights: {(weights < 0).sum().item()}")
    print(f"Total number of weights: {weights.numel()}")
    print()


# Visualise different information regarding weights and changes in weights
def plot_ltp_ltd_ex_in(layer, layer_name, num_filters=10, detailed_mode=False):
    weights = {
        'wee': layer.weight_ee.data,
        'wei': layer.weight_ei.data,
        'wie': layer.weight_ie.data
    }
    delta_w = {
        'wee': layer.delta_w_ee.data,
        'wei': layer.delta_w_ei.data,
        'wie': layer.delta_w_ie.data
    }

    if not detailed_mode:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for idx, (weight_type, weight) in enumerate(weights.items()):
            ax = axes[idx]
            delta = delta_w[weight_type]
            for i in range(min(num_filters, weight.shape[0])):
                weight_change = delta[i].sum().item()
                color = 'green' if weight_change > 0 else 'red'
                ax.bar(i, weight_change, color=color)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Filter Index')
            ax.set_ylabel('Total Weight Change')
            ax.set_title(f'Overall LTP/LTD for {weight_type} in {layer_name}')
        plt.tight_layout()
        wandb.log({f"{layer_name}_Overall_LTP_LTD": wandb.Image(fig)})
        plt.close(fig)

    else:
        for weight_type in weights.keys():
            weight = weights[weight_type]
            delta = delta_w[weight_type]

            # Plot 1: Overall weight change
            fig, ax1 = plt.subplots(figsize=(12, 6))
            for i in range(min(num_filters, weight.shape[0])):
                weight_change = delta[i].sum().item()
                color = 'green' if weight_change > 0 else 'red'
                ax1.bar(i, weight_change, color=color)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1.set_xlabel('Filter Index')
            ax1.set_ylabel('Total Weight Change')
            ax1.set_title(f'Overall LTP/LTD for {weight_type} in {layer_name}')
            wandb.log({f"{layer_name}_{weight_type}_Overall_LTP_LTD": wandb.Image(fig)})
            plt.close(fig)

            # Plot 2: Detailed weight changes within each filter
            fig, ax2 = plt.subplots(figsize=(12, 6))
            data = []
            filter_indices = []
            for i in range(min(num_filters, weight.shape[0])):
                changes = delta[i].view(-1).tolist()
                data.extend(changes)
                filter_indices.extend([i] * len(changes))
            sns.violinplot(x=filter_indices, y=data, ax=ax2)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Filter Index')
            ax2.set_ylabel('Weight Change')
            ax2.set_title(f'Detailed Weight Changes for {weight_type} in {layer_name}')
            wandb.log({f"{layer_name}_{weight_type}_Detailed_Weight_Changes": wandb.Image(fig)})
            plt.close(fig)

            # Plot 3: Detailed statistics for each filter and overall layer statistics
            fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})
            stats = []
            for i in range(min(num_filters, weight.shape[0])):
                filter_weights = weight[i].view(-1)
                stats.append({
                    'Mean': filter_weights.mean().item(),
                    'Median': filter_weights.median().item(),
                    'Std Dev': filter_weights.std().item(),
                    '% Positive': (filter_weights >= 0).float().mean().item() * 100,
                    '% Negative': (filter_weights < 0).float().mean().item() * 100
                })
            stat_df = pd.DataFrame(stats)
            sns.heatmap(stat_df.T, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_xlabel('Filter Index')
            ax3.set_title(f'Detailed Statistics for Each Filter in {weight_type}')
            all_weights = weight.view(-1)
            overall_stats = pd.DataFrame({
                'Layer Overall': {
                    'Mean': all_weights.mean().item(),
                    'Median': all_weights.median().item(),
                    'Std Dev': all_weights.std().item(),
                    '% Positive': (all_weights > 0).float().mean().item() * 100,
                    '% Negative': (all_weights < 0).float().mean().item() * 100
                }
            })
            sns.heatmap(overall_stats, annot=True, cmap='coolwarm', center=0, ax=ax4)
            ax4.set_title(f'Overall Layer Statistics for {weight_type}')
            plt.tight_layout()
            wandb.log({f"{layer_name}_{weight_type}_Weight_Statistics": wandb.Image(fig)})
            plt.close(fig)

            # Plot 4: LTP/LTD per Weight (mean across channels)
            num_filters_to_show = min(25, weight.shape[0])
            if weight.shape[2] == 1 and weight.shape[3] == 1:  # Check if kernels are 1x1
                fig, ax = plt.subplots(figsize=(20, 5))
                filter_changes = [delta[i].mean().item() for i in range(num_filters_to_show)]
                norm = plt.Normalize(vmin=min(filter_changes), vmax=max(filter_changes))
                colors = plt.cm.RdYlGn(norm(filter_changes))
                ax.bar(range(num_filters_to_show), filter_changes, color=colors)
                ax.set_xlabel('Filter Index')
                ax.set_ylabel('Weight Change')
                ax.set_title(f'LTP/LTD for 1x1 Kernels in {weight_type}')
                ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
                sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', aspect=30)
                cbar.set_label('Weight Change')
            else:
                rows = int(np.ceil(np.sqrt(num_filters_to_show)))
                cols = int(np.ceil(num_filters_to_show / rows))
                fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
                axes = axes.flatten()
                for i in range(num_filters_to_show):
                    filter_changes = delta[i].mean(dim=0).cpu().numpy()  # Mean across channels
                    im = axes[i].imshow(filter_changes, cmap='RdYlGn', interpolation='nearest')
                    axes[i].set_title(f'Filter {i}')
                    axes[i].axis('off')
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                for j in range(i + 1, rows * cols):
                    axes[j].axis('off')
                fig.suptitle(f'LTP/LTD per Weight (Mean across channels) for {weight_type}', fontsize=16)
            wandb.log({f"{layer_name}_{weight_type}_LTP_LTD_per_Weight": wandb.Image(fig)})
            plt.close(fig)

            # Plot 5: Detailed statistics for each filter and overall layer statistics: delta_w
            fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})
            stats = []
            for i in range(min(num_filters, weight.shape[0])):
                filter_weights = delta[i].view(-1)
                stats.append({
                    'Mean': filter_weights.mean().item(),
                    'Median': filter_weights.median().item(),
                    'Std Dev': filter_weights.std().item(),
                    '% Positive': (filter_weights > 0).float().mean().item() * 100,
                    '% Negative': (filter_weights < 0).float().mean().item() * 100
                })
            stat_df = pd.DataFrame(stats)
            sns.heatmap(stat_df.T, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_xlabel('Filter Index')
            ax3.set_title(f'Detailed Statistics for Each Filter in {weight_type}')
            all_weights = delta.view(-1)
            overall_stats = pd.DataFrame({
                'Layer Overall': {
                    'Mean': all_weights.mean().item(),
                    'Median': all_weights.median().item(),
                    'Std Dev': all_weights.std().item(),
                    '% Positive': (all_weights > 0).float().mean().item() * 100,
                    '% Negative': (all_weights < 0).float().mean().item() * 100
                }
            })
            sns.heatmap(overall_stats, annot=True, cmap='coolwarm', center=0, ax=ax4)
            ax4.set_title(f'Overall Layer Statistics for Delta_w in {weight_type}')
            plt.tight_layout()
            wandb.log({f"{layer_name}_{weight_type}_Delta_w_Statistics": wandb.Image(fig)})
            plt.close(fig)

            # Plot 6: Weight distribution plots
            fig, ax6 = plt.subplots(figsize=(12, 6))
            sns.histplot(weight.view(-1).cpu().numpy(), bins=50, kde=True, ax=ax6)
            ax6.set_xlabel('Weight Value')
            ax6.set_ylabel('Frequency')
            ax6.set_title(f'Weight Distribution in {layer_name} for {weight_type}')
            wandb.log({f"{layer_name}_{weight_type}_Weight_Distribution": wandb.Image(fig)})
            plt.close(fig)

def plot_ltp_ltd(layer, layer_name, num_filters=10, detailed_mode=False):
    weights = layer.weight.data
    delta_w = layer.delta_w.data
    # print(f"Layer {layer_name}")
    # print(delta_w.mean())
    # print(delta_w.max())

    if not detailed_mode:
        fig, ax = plt.subplots(figsize=(12, 6))

        for i in range(min(num_filters, weights.shape[0])):
            weight_change = delta_w[i].sum().item()
            color = 'green' if weight_change > 0 else 'red'
            ax.bar(i, weight_change, color=color)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Filter Index')
        ax.set_ylabel('Total Weight Change')
        ax.set_title(f'Overall LTP/LTD for first {num_filters} filters in {layer_name}')

        # Log the plot to wandb
        wandb.log({f"{layer_name}_Overall_LTP_LTD": wandb.Image(fig)})
        plt.close(fig)

    else:
        # Plot 1: Overall weight change
        fig, ax1 = plt.subplots(figsize=(12, 6))
        for i in range(min(num_filters, weights.shape[0])):
            weight_change = delta_w[i].sum().item()
            color = 'forestgreen' if weight_change > 0 else 'firebrick'
            ax1.bar(i, weight_change, color=color)

        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Filter Index')
        ax1.set_ylabel('Total Weight Change')
        ax1.set_title(f'Overall LTP/LTD for first {num_filters} filters')

        # Log the plot to wandb
        wandb.log({f"{layer_name}_Overall_LTP_LTD": wandb.Image(fig)})
        plt.close(fig)

        # Plot 2: Detailed weight changes within each filter
        fig, ax2 = plt.subplots(figsize=(12, 6))
        data = []
        filter_indices = []
        for i in range(min(num_filters, weights.shape[0])):
            changes = delta_w[i].view(-1).tolist()
            data.extend(changes)
            filter_indices.extend([i] * len(changes))

        sns.violinplot(x=filter_indices, y=data, ax=ax2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Filter Index')
        ax2.set_ylabel('Weight Change')
        ax2.set_title(f'Detailed Weight Changes within first {num_filters} filters')

        # Log the plot to wandb
        wandb.log({f"{layer_name}_Detailed_Weight_Changes": wandb.Image(fig)})
        plt.close(fig)

        # Plot 3: Detailed statistics for each filter and overall layer statistics
        fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})
        stats = []
        for i in range(min(num_filters, weights.shape[0])):
            filter_weights = weights[i].view(-1) #could be delta_w
            stats.append({
                'Mean': filter_weights.mean().item(),
                'Median': filter_weights.median().item(),
                'Std Dev': filter_weights.std().item(),
                '% Positive': (filter_weights > 0).float().mean().item() * 100,
                '% Negative': (filter_weights < 0).float().mean().item() * 100
            })
        # Per-filter statistics
        stat_df = pd.DataFrame(stats)
        sns.heatmap(stat_df.T, annot=True, cmap='coolwarm', center=0, ax=ax3)
        ax3.set_xlabel('Filter Index')
        ax3.set_title('Detailed Statistics for Each Filter')
        # Overall layer statistics
        all_weights = weights.view(-1)  # This includes ALL weights in the layer
        overall_stats = pd.DataFrame({
            'Layer Overall': {
                'Mean': all_weights.mean().item(),
                'Median': all_weights.median().item(),
                'Std Dev': all_weights.std().item(),
                '% Positive': (all_weights > 0).float().mean().item() * 100,
                '% Negative': (all_weights < 0).float().mean().item() * 100
            }
        })
        sns.heatmap(overall_stats, annot=True, cmap='coolwarm', center=0, ax=ax4)
        ax4.set_title('Overall Layer Statistics')
        plt.tight_layout()
        # Log the plot to wandb
        wandb.log({f"{layer_name}_Weight_Statistics": wandb.Image(fig)})
        plt.close(fig)

        # Plot 4: LTP/LTD per Weight (mean across channels)
        num_filters_to_show = min(25, weights.shape[0])
        if weights.shape[2] == 1 and weights.shape[3] == 1:  # Check if kernels are 1x1
            fig, ax = plt.subplots(figsize=(20, 5))
            filter_changes = [delta_w[i].mean().item() for i in range(num_filters_to_show)]
            norm = plt.Normalize(vmin=min(filter_changes), vmax=max(filter_changes))
            colors = plt.cm.RdYlGn(norm(filter_changes))
            ax.bar(range(num_filters_to_show), filter_changes, color=colors)
            ax.set_xlabel('Filter Index')
            ax.set_ylabel('Weight Change')
            ax.set_title('LTP/LTD for 1x1 Kernels')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', aspect=30)
            cbar.set_label('Weight Change')

        else:
            rows = int(np.ceil(np.sqrt(num_filters_to_show)))
            cols = int(np.ceil(num_filters_to_show / rows))
            fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
            axes = axes.flatten()

            for i in range(num_filters_to_show):
                filter_changes = delta_w[i].mean(dim=0).cpu().numpy()  # Mean across channels
                im = axes[i].imshow(filter_changes, cmap='RdYlGn', interpolation='nearest')
                axes[i].set_title(f'Filter {i}')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

            for j in range(i + 1, rows * cols):
                axes[j].axis('off')

            fig.suptitle('LTP/LTD per Weight (Mean across channels)', fontsize=16)
        # Log the plot to wandb
        wandb.log({f"{layer_name}_LTP_LTD_per_Weight": wandb.Image(fig)})
        plt.close(fig)

        # Plot 5: Detailed statistics for each filter and overall layer statistics: delta_w
        fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})
        stats = []
        for i in range(min(num_filters, weights.shape[0])):
            filter_weights = delta_w[i].view(-1)  # could be delta_w
            stats.append({
                'Mean': filter_weights.mean().item(),
                'Median': filter_weights.median().item(),
                'Std Dev': filter_weights.std().item(),
                '% Positive': (filter_weights > 0).float().mean().item() * 100,
                '% Negative': (filter_weights < 0).float().mean().item() * 100
            })
        # Per-filter statistics
        stat_df = pd.DataFrame(stats)
        sns.heatmap(stat_df.T, annot=True, cmap='coolwarm', center=0, ax=ax3)
        ax3.set_xlabel('Filter Index')
        ax3.set_title('Detailed Statistics for Each Filter')
        # Overall layer statistics
        all_weights = delta_w.view(-1)  # This includes ALL weights in the layer
        overall_stats = pd.DataFrame({
            'Layer Overall': {
                'Mean': all_weights.mean().item(),
                'Median': all_weights.median().item(),
                'Std Dev': all_weights.std().item(),
                '% Positive': (all_weights > 0).float().mean().item() * 100,
                '% Negative': (all_weights < 0).float().mean().item() * 100
            }
        })
        sns.heatmap(overall_stats, annot=True, cmap='coolwarm', center=0, ax=ax4)
        ax4.set_title('Overall Layer Statistics for Delta_w')
        plt.tight_layout()
        # Log the plot to wandb
        wandb.log({f"{layer_name}_Delta_w_Statistics": wandb.Image(fig)})
        plt.close(fig)

        # Plot 6: Weight distribution plots to understand connectivity of weights better
        fig, ax6 = plt.subplots(figsize=(12, 6))
        sns.histplot(weights.view(-1).cpu().numpy(), bins=50, kde=True, ax=ax6)
        ax6.set_xlabel('Weight Value')
        ax6.set_ylabel('Frequency')
        ax6.set_title(f'Weight Distribution in {layer_name}')
        # Log the plot to wandb
        wandb.log({f"{layer_name}_Weight_Distribution": wandb.Image(fig)})
        plt.close(fig)


def visualize_data_clusters(dataloader, model=None, method='tsne', dim=2, perplexity=30, n_neighbors=15, min_dist=0.1,
                            n_components=2, random_state=42):
    # Full CIFAR-10 class names for clear identification
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

    # class_names = ['0', '1', '2', '3', '4',
    #                '5', '6', '7', '8', '9']

    # class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_list = []
    labels_list = []

    # Extract features using the model if provided
    if model is not None:
        model.eval()
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            if model is not None:
                if hasattr(model, 'features_extract'):
                    features = model.features_extract(data)
                else:
                    features = model.conv1(data)
            else:
                features = data
            features = features.view(features.size(0), -1).cpu().numpy()
            features_list.append(features)
            labels_list.append(labels.numpy())

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Apply dimensionality reduction with original parameters
    if method == 'tsne':
        reducer = TSNE(n_components=dim, perplexity=perplexity, n_iter=1000, random_state=random_state)
    elif method == 'umap':
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=dim, random_state=random_state)
    elif method == 'lda':
        reducer = LinearDiscriminantAnalysis(n_components=2)
    elif method == 'spectral':
        reducer = SpectralEmbedding(n_components=2)
    else:
        raise ValueError("Method must be either 'tsne' or 'umap'")

    if method == "lda":
        projected_data = reducer.fit_transform(features, labels)
    else:
        projected_data = reducer.fit_transform(features_normalized)

    # Create visualization with original plot sizes and improved text
    if dim == 2:
        fig = plt.figure(figsize=(12, 10))
        scatter = plt.scatter(projected_data[:, 0], projected_data[:, 1],
                              c=labels, alpha=0.5, cmap='tab10')

        # Colorbar with class names
        cbar = plt.colorbar(scatter, ticks=np.arange(len(class_names)))
        cbar.set_label('Classes', labelpad=len(class_names))
        cbar.set_ticklabels(class_names)

        plt.title(f'CIFAR-10 Class Separation using {method.upper()}')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')

    elif dim == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(projected_data[:, 0], projected_data[:, 1],
                             projected_data[:, 2], c=labels, alpha=0.5, cmap='tab10')

        # Colorbar with class names
        cbar = fig.colorbar(scatter, ticks=np.arange(len(class_names)))
        cbar.set_label('Classes', labelpad=len(class_names))
        cbar.set_ticklabels(class_names)

        ax.set_title(f'CIFAR-10 Class Separation using {method.upper()}')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_zlabel(f'{method.upper()} Component 3')

    # Ensure proper spacing and save to wandb
    plt.tight_layout()
    wandb.log({"Class_Separation": wandb.Image(fig)})
    plt.close(fig)