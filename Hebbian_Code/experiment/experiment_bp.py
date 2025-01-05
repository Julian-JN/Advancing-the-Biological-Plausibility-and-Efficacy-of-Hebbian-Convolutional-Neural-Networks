from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

from data_processing.data import get_data

from models.Model_BackProp import Net_Backpropagation
import numpy as np
import matplotlib.pyplot as plt
import umap

from utils.logger import Logger
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import seaborn as sns
import wandb
import pandas as pd

from visualisation.receptive_fields import visualize_filters
from visualisation.whole_image_receptive_field import visualize_receptive_fields_context, clear_image_cache
from visualisation.top_receptive_fields import visualize_top_activations
from visualisation.heatmap_activations import visualize_mapped_importance

torch.manual_seed(0)

def print_weight_statistics(layer, layer_name):
    """
    Prints statistics of the weights from a given layer.

    Args:
    layer (nn.Module): The layer to analyze
    layer_name (str): Name of the layer for printing purposes
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
    print(f"Number of positive weights: {(weights > 0).sum().item()}")
    print(f"Number of negative weights: {(weights < 0).sum().item()}")
    print(f"Total number of weights: {weights.numel()}")
    print()


def plot_ltp_ltd(layer, layer_name, num_filters=10, detailed_mode=False):
    weights = layer.weight.data
    delta_w = layer.weight.grad

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
            color = 'green' if weight_change > 0 else 'red'
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

        # Plot 6: Weight distribution plots
        fig, ax6 = plt.subplots(figsize=(12, 6))
        sns.histplot(weights.view(-1).cpu().numpy(), bins=50, kde=True, ax=ax6)
        ax6.set_xlabel('Weight Value')
        ax6.set_ylabel('Frequency')
        ax6.set_title(f'Weight Distribution in {layer_name}')
        wandb.log({f"{layer_name}_Weight_Distribution": wandb.Image(fig)})
        plt.close(fig)


def calculate_metrics(preds, labels, num_classes):
    if num_classes == 2:
        accuracy = Accuracy(task='binary', num_classes=num_classes).to(device)
        precision = Precision(task='binary', average='weighted', num_classes=num_classes).to(device)
        recall = Recall(task='binary', average='weighted', num_classes=num_classes).to(device)
        f1 = F1Score(task='binary', average='weighted', num_classes=num_classes).to(device)
        confusion_matrix = ConfusionMatrix(task='binary', num_classes=num_classes).to(device)
    else:
        accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        precision = Precision(task='multiclass', average='macro', num_classes=num_classes).to(device)
        recall = Recall(task='multiclass', average='macro', num_classes=num_classes).to(device)
        f1 = F1Score(task='multiclass', average='macro', num_classes=num_classes).to(device)
        confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)

    acc = accuracy(preds, labels)
    prec = precision(preds, labels)
    rec = recall(preds, labels)
    f1_score = f1(preds, labels)
    conf_matrix = confusion_matrix(preds, labels)

    return acc, prec, rec, f1_score, conf_matrix

def visualize_data_clusters(dataloader, model=None, method='tsne', dim=2, perplexity=30, n_neighbors=15, min_dist=0.1,
                            n_components=2, random_state=42):
    # Full CIFAR-10 class names for clear identification
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

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
        cbar.set_label('Classes', labelpad=10)
        cbar.set_ticklabels(class_names)

        ax.set_title(f'CIFAR-10 Class Separation using {method.upper()}')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_zlabel(f'{method.upper()} Component 3')

    # Ensure proper spacing and save to wandb
    plt.tight_layout()
    wandb.log({"Class_Separation": wandb.Image(fig)})
    plt.close(fig)

# Main file to train backpropagation network
# TODO: Same visualisation code as Hebbian is not fully implemented

if __name__ == "__main__":

    device = torch.device('cuda:0')
    model = Net_Backpropagation()
    model.to(device)

    wandb_logger = Logger(
        f"SoftHebb-BackPropagation",
        project='CIFAR10_Dataset', model=model)
    logger = wandb_logger.get_logger()
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter Count Total: {num_parameters}")

    sup_optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trn_set, tst_set, zca = get_data(dataset='cifar10', root='datasets', batch_size=64,
                                          whiten_lvl=None)

    # # Dataset visualization
    # trainimages, trainlabels = next(iter(trn_set))
    # print(trainimages.shape)
    # print(f"Feature batch shape: {trainimages.size()}")
    # print(f"Labels batch shape: {trainlabels.size()}")
    # for i in range(3):
    #     img = trainimages[i].squeeze().cpu()
    #     label = trainlabels[i]
    #     plt.imshow(img.T, cmap="gray")
    #     plt.show()
    #     print(f"Label: {label}")
    #     plt.close()
    print("Initial Filters")
    model.visualize_filters('conv1')
    model.visualize_filters('conv2')
    model.visualize_filters('conv3')

    print(f'Processing Training batches: {len(trn_set)}')
    for epoch in range(20):
        print(f"Training BP epoch {epoch}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_preds = []
        train_labels = []
        for i, data in enumerate(trn_set, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            sup_optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # if i % 200 == 0:  # Every 100 datapoint
            #     print(f'Saving details after batch {i}')
            #     plot_ltp_ltd(model.conv1, 'conv1', num_filters=10, detailed_mode=True)
            #     plot_ltp_ltd(model.conv2, 'conv2', num_filters=10, detailed_mode=True)
            #     # plot_ltp_ltd(model.conv_point2, 'conv_point2', num_filters=10, detailed_mode=True)
            #     model.visualize_filters('conv1')
            #     model.visualize_filters('conv2')
            #     model.visualize_filters('conv3')

            sup_optimizer.step()
            # compute training statistics
            running_loss += loss.item()
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            # For wandb logs
            preds = torch.argmax(outputs, dim=1)
            train_preds.append(preds)
            train_labels.append(labels)

        print(f'Saving details after Epoch {epoch}')
        plot_ltp_ltd(model.conv1, 'conv1', num_filters=10, detailed_mode=True)
        plot_ltp_ltd(model.conv2, 'conv2', num_filters=10, detailed_mode=True)
        plot_ltp_ltd(model.conv2, 'conv3', num_filters=10, detailed_mode=True)

        # plot_ltp_ltd(model.conv_point2, 'conv_point2', num_filters=10, detailed_mode=True)
        model.visualize_filters('conv1')
        model.visualize_filters('conv2')
        model.visualize_filters('conv3')


        print(f'Accuracy of the network on the train images: {100 * correct // total} %')
        print(f'[{epoch + 1}] loss: {running_loss / total:.3f}')

        train_preds = torch.cat(train_preds, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        acc, prec, rec, f1_score, conf_matrix = calculate_metrics(train_preds, train_labels, 10)
        logger.log({'train_accuracy': acc, 'train_precision': prec, 'train_recall': rec, 'train_f1_score': f1_score})
        f, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(conf_matrix.clone().detach().cpu().numpy(), annot=True, ax=ax)
        logger.log({"train_confusion_matrix": wandb.Image(f)})
        plt.close(f)

        # Evaluation on test set
        model.eval()
        running_loss = 0.
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for data in tst_set:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                # For wandb logs
                preds = torch.argmax(outputs, dim=1)
                test_preds.append(preds)
                test_labels.append(labels)

        print(f'Accuracy of the network on the test images: {100 * correct / total} %')
        print(f'test loss: {running_loss / total:.3f}')

        test_preds = torch.cat(test_preds, dim=0)
        test_labels = torch.cat(test_labels, dim=0)
        acc, prec, rec, f1_score, conf_matrix = calculate_metrics(test_preds, test_labels, 10)
        logger.log({'test_accuracy': acc, 'test_precision': prec, 'test_recall': rec, 'test_f1_score': f1_score})
        f, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(conf_matrix.clone().detach().cpu().numpy(), annot=True, ax=ax)
        logger.log({"test_confusion_matrix": wandb.Image(f)})
        plt.close(f)

    print("Visualizing Filters")
    model.visualize_filters('conv1', f'results/{"demo"}/demo_conv1_filters_epoch_{1}.png')
    model.visualize_filters('conv2', f'results/{"demo"}/demo_conv2_filters_epoch_{1}.png')
    model.visualize_filters('conv3', f'results/{"demo"}/demo_conv3_filters_epoch_{1}.png')
    # model.visualize_filters('conv_point2', f'results/{"demo"}/demo_conv_point2_filters_epoch_{1}.png')
    print("Weight statistics")
    print_weight_statistics(model.conv1, 'conv1')
    print_weight_statistics(model.conv2, 'conv2')
    print_weight_statistics(model.conv3, 'conv3')
    # print_weight_statistics(model.conv_point2, 'conv3')

    print("Visualizing Class separation")

    visualize_data_clusters(tst_set, model=model, method='lda', n_components=2)
    visualize_data_clusters(tst_set, model=model, method='spectral', n_components=2, n_neighbors=30)
    visualize_data_clusters(tst_set, model=model, method='umap', dim=2)
    visualize_data_clusters(tst_set, model, method='tsne', perplexity=40)
    visualize_data_clusters(tst_set, model=model, method='umap', dim=2, n_neighbors=25)


    visualize_mapped_importance(model, tst_set, model.conv1, max_batch_images=8)
    visualize_mapped_importance(model, tst_set, model.conv2, max_batch_images=8)
    visualize_mapped_importance(model, tst_set, model.conv3, max_batch_images=8)
    clear_image_cache()

    visualize_top_activations(model, dataloader=tst_set, layer=model.conv1, num_top=25, max_batch_images=8)
    visualize_top_activations(model, tst_set, model.conv2, num_top=25, max_batch_images=8)
    visualize_top_activations(model, tst_set, model.conv3, num_top=25, max_batch_images=8)
    clear_image_cache()

    visualize_receptive_fields_context(model, dataloader=tst_set, layer=model.conv1, num_filters=25, max_batch_images=8)
    visualize_receptive_fields_context(model, tst_set, model.conv2, num_filters=25, max_batch_images=8)
    visualize_receptive_fields_context(model, tst_set, model.conv3, num_filters=25, max_batch_images=8)
    clear_image_cache()

    print("Visualizing Receptive fields")
    visualize_filters(model, model.conv1, num_filters=25)
    visualize_filters(model, model.conv2, num_filters=25)
    visualize_filters(model, model.conv3, num_filters=25)