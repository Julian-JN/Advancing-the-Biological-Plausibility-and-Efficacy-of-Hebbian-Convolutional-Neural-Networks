import torch
import torch.nn as nn
import torch.optim as optim

from hebbian_config import get_hebbian_config

from data_processing.data import get_data
from models.model_hebb import Net_Hebbian
import matplotlib.pyplot as plt
import warnings

from utils.logger import Logger
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import seaborn as sns
import wandb
from visualisation.visualizer import plot_ltp_ltd, print_weight_statistics, visualize_data_clusters
from visualisation.receptive_fields import visualize_filters
from visualisation.whole_image_receptive_field import visualize_receptive_fields_context, clear_image_cache
from visualisation.top_receptive_fields import visualize_top_activations
from visualisation.heatmap_activations import visualize_mapped_importance

"""
Code structure inspired and modified from https://github.com/NeuromorphicComputing/SoftHebb
Similar training and testing loop, with some identical function for the custom learning rate schedule
"""

torch.manual_seed(0)

# Calculate evaluation metrics
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

class WeightNormDependentLR(optim.lr_scheduler._LRScheduler):
    """
    Custom Learning Rate Scheduler for unsupervised training of SoftHebb Convolutional blocks.
    Difference between current neuron norm and theoretical converged norm (=1) scales the initial lr.
    """

    def __init__(self, optimizer, power_lr, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.initial_lr_groups = [group['lr'] for group in self.optimizer.param_groups]  # store initial lrs
        self.power_lr = power_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        new_lr = []
        for i, group in enumerate(self.optimizer.param_groups):
            for param in group['params']:
                # difference between current neuron norm and theoretical converged norm (=1) scales the initial lr
                # initial_lr * |neuron_norm - 1| ** 0.5
                norm_diff = torch.abs(torch.linalg.norm(param.view(param.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                new_lr.append(self.initial_lr_groups[i] * (norm_diff ** self.power_lr)[:, None, None, None])
        return new_lr

class TensorLRSGD(optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step, using a non-scalar (tensor) learning rate.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(-group['lr'] * d_p)
        return loss

# Main training and testing loop
# Experiment is configured to replicate SoftHebb-Surr/HardWTA/Cos-Instar configuration results
if __name__ == "__main__":

    hebb_param = get_hebbian_config(version="optimal")
    device = torch.device('cuda:0')
    print(device)
    model = Net_Hebbian(hebb_params=hebb_param, version="softhebb")
    model.to(device)

    wandb_logger = Logger(
        f"SoftHebb_Optimal",project='CIFAR10_Dataset', model=model)
    logger = wandb_logger.get_logger()
    print(hebb_param)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter Count Total: {num_parameters}")

    # Custom learning rate and scheduler for only SoftHebb implementation
    # unsup_optimizer = TensorLRSGD([
    #     {"params": model.conv1.parameters(), "lr": 0.08, },
    #     {"params": model.conv2.parameters(), "lr": 0.005, },
    #     {"params": model.conv3.parameters(), "lr": 0.01, },
    # ], lr=0)
    # unsup_lr_scheduler = WeightNormDependentLR(unsup_optimizer, power_lr=0.5)

    # Learning rate for all other hebbian experiments
    hebb_params = [
        {'params': model.conv1.parameters(), 'lr': 0.1},
        {'params': model.conv2.parameters(), 'lr': 0.1},
        {'params': model.conv3.parameters(), 'lr': 0.1},
        # {'params': model.conv4.parameters(), 'lr': 0.1},
    ]
    # unsup_optimizer = optim.SGD(hebb_params, lr=0)  # The lr here will be overridden by the individual lrs

    # sup_optimizer = optim.Adam(model.fc1.parameters(), lr=0.001)
    sup_optimizer = optim.Adam([
        {'params': model.fc1.parameters()},
        # {'params': model.fc2.parameters()}
    ], lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        sup_optimizer, lr_lambda=lambda epoch: 0.5 ** max(0, (epoch - 10) // 2))
    criterion = nn.CrossEntropyLoss()

    mode = hebb_param['mode']
    if mode == "soft":
        ZCA_White = None
    else:
        ZCA_White = 1e-3
    trn_set, tst_set, zca = get_data(dataset='cifar10', root='../datasets', batch_size=64,
                                          whiten_lvl=ZCA_White)
    print(f'Processing Testing batches: {len(trn_set)}')

    print("Initial Weight statistics")
    print_weight_statistics(model.conv1, 'conv1')
    print_weight_statistics(model.conv2, 'conv2')
    print_weight_statistics(model.conv3, 'conv3')

    print("Visualizing Initial Filters")
    model.visualize_filters('conv1')
    model.visualize_filters('conv2')
    model.visualize_filters('conv3')
    # model.visualize_filters('conv4', f'results/{"demo"}/demo_conv3_filters_epoch_{1}.png')
    # model.visualize_filters('conv6', f'results/{"demo"}/demo_conv3_filters_epoch_{1}.png')

    running_loss = 0.0
    # Hebbian feature extraction
    for epoch in range(1):
        print(f"Training Hebbian epoch {epoch}")
        for i, data in enumerate(trn_set, 0):
            print(f"Training Hebbian Batch {i}")
            inputs, _ = data
            inputs = inputs.to(device)
            # print(inputs.shape)
            # zero the parameter gradients
            # unsup_optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(inputs)
            # Visualize changes before updating
            # if (i % 200) == 0: # Every 100 datapoint
            #     print(f'Saving details after batch {i}')
            #     plot_ltp_ltd(model.conv1, 'conv1', num_filters=10, detailed_mode=True)
            #     plot_ltp_ltd(model.conv2, 'conv2', num_filters=10, detailed_mode=True)
            #     plot_ltp_ltd(model.conv3, 'conv3', num_filters=10, detailed_mode=True)
            #     model.visualize_filters('conv1')
            #     model.visualize_filters('conv2')
            #     model.visualize_filters('conv3')
            for layer in [model.conv1, model.conv2, model.conv3]:
                if hasattr(layer, 'local_update'):
                    layer.local_update()
            # unsup_optimizer.step()
            # Scheduler only for SoftHebb
            # unsup_lr_scheduler.step()
    print("Visualizing Filters")
    plot_ltp_ltd(model.conv1, 'conv1', num_filters=10, detailed_mode=True)
    plot_ltp_ltd(model.conv2, 'conv2', num_filters=10, detailed_mode=True)
    plot_ltp_ltd(model.conv3, 'conv3', num_filters=10, detailed_mode=True)
    # plot_ltp_ltd(model.conv4, 'conv4', num_filters=10, detailed_mode=True)
    model.visualize_filters('conv1')
    model.visualize_filters('conv2')
    model.visualize_filters('conv3')
    # model.visualize_filters('conv4', f'results/{"demo"}/demo_conv3_filters_epoch_{1}.png')

    # set requires grad false and eval mode for all modules but classifier
    # unsup_optimizer.zero_grad()

    for name, module in model.named_modules():
        if any(x in name for x in ['conv', 'bn']):
            # Freeze the layer's parameters
            for param in module.parameters():
                param.requires_grad = False
            # Set to eval mode
            module.eval()

    visualize_mapped_importance(model, tst_set, model.conv1, max_batch_images=8)
    visualize_mapped_importance(model, tst_set, model.conv2, max_batch_images=8)
    visualize_mapped_importance(model, tst_set, model.conv3, max_batch_images=8)
    # visualize_mapped_importance(model, tst_set, model.conv4, max_batch_images=8)
    # visualize_mapped_importance(model, tst_set, model.conv5, max_batch_images=8)
    # visualize_mapped_importance(model, tst_set, model.conv6, max_batch_images=8)
    clear_image_cache()

    visualize_top_activations(model, dataloader=tst_set, layer=model.conv1, num_top=25, max_batch_images=8)
    visualize_top_activations(model, tst_set, model.conv2, num_top=25, max_batch_images=8)
    visualize_top_activations(model, tst_set, model.conv3, num_top=25, max_batch_images=8)
    # visualize_top_activations(model, tst_set, model.conv4, num_top=25, max_batch_images=8)
    # visualize_top_activations(model, tst_set, model.conv5, num_top=25, max_batch_images=8)
    # visualize_top_activations(model, tst_set, model.conv6, num_top=25, max_batch_images=8)
    clear_image_cache()

    visualize_receptive_fields_context(model, dataloader=tst_set, layer=model.conv1, num_filters=25, max_batch_images=8)
    visualize_receptive_fields_context(model, tst_set, model.conv2, num_filters=25, max_batch_images=8)
    visualize_receptive_fields_context(model, tst_set, model.conv3, num_filters=25, max_batch_images=8)
    # visualize_receptive_fields_context(model, tst_set, model.conv4, num_filters=25, max_batch_images=8)
    # visualize_receptive_fields_context(model, tst_set, model.conv5, num_filters=25, max_batch_images=8)
    # visualize_receptive_fields_context(model, tst_set, model.conv6, num_filters=25, max_batch_images=8)
    clear_image_cache()


    print("Visualizing Test Class separation")
    # visualize_data_clusters(tst_set, model=model, method='lda', n_components=2)
    # visualize_data_clusters(tst_set, model=model, method='spectral', n_components=2, n_neighbors=30)
    visualize_data_clusters(tst_set, model=model, method='umap', dim=2)
    # visualize_data_clusters(tst_set, model, method='tsne', perplexity=40)
    # visualize_data_clusters(tst_set, model=model, method='umap', dim=2, n_neighbors=25)

    # Supervised training of classifier
    # Train classifier with backpropagation
    print("Training Classifier")
    for epoch in range(20):
        model.fc1.train()
        # model.fc2.train()
        model.dropout.train()
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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
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
        test_preds = []
        test_labels = []
        # since we're not training, we don't need to calculate the gradients for our outputs/model
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

        # Step the scheduler after each epoch
        scheduler.step()
    # Visualise receptive fields at end, as this code alters the model architecture (padding removal)
    print("Visualizing Receptive fields")
    visualize_filters(model, model.conv1, num_filters=25)
    visualize_filters(model, model.conv2, num_filters=25)
    visualize_filters(model, model.conv3, num_filters=25)
    # visualize_filters(model, model.conv4, num_filters=25)
    # visualize_filters(model, model.conv5, num_filters=25)
    # visualize_filters(model, model.conv6, num_filters=25)