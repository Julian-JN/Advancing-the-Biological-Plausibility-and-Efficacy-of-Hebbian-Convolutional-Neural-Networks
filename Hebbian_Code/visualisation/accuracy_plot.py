import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_comparison(exp1_results, exp2_results, exp3_results, exp4_results, experiment_names=None,
                             title_suffix=""):
    """
    Plot accuracy results for four different experiments across epochs.
    """
    n_epochs = len(exp1_results)
    if not all(len(results) == n_epochs for results in [exp1_results, exp2_results, exp3_results, exp4_results]):
        raise ValueError("All experiment results must contain the same number of values")

    epochs = np.arange(1, n_epochs + 1)

    if experiment_names is None:
        experiment_names = ['Experiment 1', 'Experiment 2', 'Experiment 3', 'Experiment 4']

    plt.figure(figsize=(12, 7), dpi=150)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 2

    marker_every = max(1, n_epochs // 10)
    plt.plot(epochs, exp1_results, 'o-', label=experiment_names[0], linewidth=2.5,
             markersize=10, markevery=marker_every)
    plt.plot(epochs, exp2_results, 's--', label=experiment_names[1], linewidth=2.5,
             markersize=10, markevery=marker_every)
    plt.plot(epochs, exp3_results, '^:', label=experiment_names[2], linewidth=2.5,
             markersize=10, markevery=marker_every)
    plt.plot(epochs, exp4_results, 'D-.', label=experiment_names[3], linewidth=2.5,
             markersize=10, markevery=marker_every)

    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
    plt.title(f'Accuracy Comparison Across Experiments ({title_suffix})',
              fontsize=18, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1)

    plt.legend(fontsize=12, frameon=True, fancybox=False, edgecolor='black',
               bbox_to_anchor=(1.02, 1), loc='upper left')

    all_results = exp1_results + exp2_results + exp3_results + exp4_results
    plt.ylim(min(all_results) * 0.95, max(all_results) * 1.05)

    plt.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
    plt.tick_params(axis='both', which='minor', width=1, length=3)

    plt.xticks(np.arange(0, n_epochs + 1, 2))

    plt.tight_layout()
    plt.show()


# Test data
test_exp1 = [0.6703, 0.6922, 0.7115, 0.7214, 0.7214, 0.7293, 0.7347, 0.7344, 0.7398,
             0.7465, 0.7436, 0.7434, 0.7559, 0.7557, 0.7548, 0.7583, 0.7600, 0.7580,
             0.7594, 0.7592]  # Hard-BCM test

test_exp2 = [0.5962, 0.6247, 0.6535, 0.6795, 0.6868, 0.7245, 0.7366, 0.7310, 0.7477,
             0.7443, 0.7382, 0.7386, 0.7570, 0.7526, 0.7301, 0.7475, 0.7582, 0.7707,
             0.7637, 0.7774]  # BackProp test

test_exp3 = [0.6703, 0.6896, 0.7078, 0.7108, 0.7225, 0.7413, 0.7246, 0.7321, 0.7310,
             0.7274, 0.7206, 0.7222, 0.7676, 0.7561, 0.7763, 0.7874, 0.7886, 0.7834,
             0.7919, 0.7923]  # Soft-Instar test

test_exp4 = [0.4970, 0.5296, 0.5399, 0.5437, 0.5522, 0.5623, 0.5619, 0.5690, 0.5730,
             0.5780, 0.5772, 0.5787, 0.5927, 0.5919, 0.5912, 0.5887, 0.5952, 0.5969,
             0.5945, 0.5950]  # Hard-Instar test

# Training data
train_exp1 = [0.5876, 0.6793, 0.7000, 0.7157, 0.7235, 0.7319, 0.7412, 0.7443, 0.7429,
              0.7498, 0.7527, 0.7518, 0.7637, 0.7646, 0.7708, 0.7726, 0.7750, 0.7779,
              0.7788, 0.7772]  # Hard-BCM train

train_exp2 = [0.4558, 0.5684, 0.6161, 0.6477, 0.6701, 0.6912, 0.7071, 0.7206, 0.7317,
              0.7417, 0.7391, 0.7396, 0.7422, 0.7424, 0.7542, 0.7482, 0.7615, 0.7625,
              0.7680, 0.7688]  # BackProp train

train_exp3 = [0.5439, 0.6317, 0.6590, 0.6723, 0.6861, 0.6948, 0.6977, 0.7014, 0.7037,
              0.7082, 0.7156, 0.7154, 0.7353, 0.7406, 0.7493, 0.7550, 0.7584, 0.7613,
              0.7648, 0.7613]  # Soft-Instar train

train_exp4 = [0.3986, 0.4718, 0.4866, 0.4968, 0.5027, 0.5090, 0.5129, 0.5151, 0.5180,
              0.5146, 0.5186, 0.5185, 0.5276, 0.5296, 0.5360, 0.5375, 0.5387, 0.5387,
              0.5418, 0.5436]  # Hard-Instar train

# Plot with custom names
custom_names = ['SoftHebb-Optimal', 'SoftHebb-BackPropagation', 'SoftHebb-SoftWTA-Instar', 'Lagani-Hard/Cos-Instar']

# Create separate plots for test and training accuracies
plot_accuracy_comparison(test_exp1, test_exp2, test_exp3, test_exp4, custom_names, "Test")
plot_accuracy_comparison(train_exp1, train_exp2, train_exp3, train_exp4, custom_names, "Training")