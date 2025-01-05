import torch.nn as nn


def get_hebbian_config(version="softhebb"):
    """
    Returns Hebbian learning configuration parameters based on the specified version.
    This centralized configuration makes it easier to manage different experimental setups.

    Args:
        version (str): The version of Hebbian learning to use
                      Options: "softhebb", "basic", "temporal", "adaptive"

    Returns:
        dict: Configuration parameters for Hebbian learning
    """
    # Base configuration shared across all versions
    base_config = {
        'mode': 'hard',  # Learning mode
        # 'alpha': 1.0,  # Learning rate for Hebbian updates
        'w_nrm': False,  # Weight normalization flag
        'act': nn.Identity(),  # Activation function

        # Competition and similarity parameters
        'use_cosine_similarity': True,  # Use cosine similarity instead of dot product
        'use_lateral_inhibition': False,  # Apply surround modulation
        'top_k': 1,  # Number of winners in competition
        # 't_invert': 1.0,  # Temperature for soft competition

        # Weight initialization and structure
        'init_method': 'softhebb',  # Weight initialization method
        'patchwise': True,  # Patch-wise update computation
        'contrast': 1.0,  # Negative vs positive update scaling
        'uniformity': False,  # Uniformity weighting flag

        # Presynaptic competition parameters
        'use_presynaptic_competition': False,
        'presynaptic_competition_type': 'lp_norm',

        # # Plasticity and homeostasis
        # 'use_homeostasis': False,  # Homeostatic plasticity flag
        # 'use_structural_plasticity': False,  # Structural plasticity flag
        # 'prune_rate': 0.1,  # Connection pruning rate
        #
        # Competition parameters
        'competition_type': 'hard',  # Competition type
        'temporal_window': 500,  # Temporal competition window
        'competition_k': 2,  # Competition strength

        'dale': False,

        # # Structural plasticity parameters
        # 'growth_probability': 0.1,  # New synapse formation probability
        # 'new_synapse_strength': 1.0,  # New synapse initial strength
        # 'prune_threshold_percentile': 10,  # Pruning threshold
        #
        # # Homeostasis parameters
        # 'gamma': 0.5,  # Homeostasis strength
        # 'theta_decay': 0.5,  # BCM threshold decay rate
    }

    # Version-specific configurations
    version_configs = {
        "soft": {
            'mode': 'soft',
            'use_cosine_similarity': False,
            'use_lateral_inhibition': False,
            'init_method': 'softhebb',
        },
        "hard": {
            'mode': 'hard',
            'use_cosine_similarity': True,
            'use_lateral_inhibition': True,
            'init_method': 'kaiming_uniform'

        },
        "hard_basic": {
            'mode': 'hard',
            'use_cosine_similarity': True,
            'use_lateral_inhibition': False,
            'init_method': 'kaiming_uniform',
            'dale': False,
        },
        "optimal": {
            'mode': 'bcm',
            'use_cosine_similarity': True,
            'use_lateral_inhibition': True,
            'init_method': 'kaiming_uniform',
            'dale': False,
            'use_homeostasis': False
        },
        "dale": {
            'mode': 'bcm',
            'use_cosine_similarity': True,
            'use_lateral_inhibition': True,
            'init_method': 'kaiming_uniform',
            'dale': True,
        },
        "bcm": {
            'mode': 'bcm',
            'use_cosine_similarity': True,
            'use_lateral_inhibition': False,
            'init_method': 'kaiming_uniform',
        },
        "temporal": {
            'mode': 'temp',
            'use_cosine_similarity': True,
            'use_lateral_inhibition': False,
            'competition_type': 'hard',
            'temporal_window': 500,
            'init_method': 'kaiming_uniform'
        },
        "adaptive": {
            'mode': 'thresh',
            'use_cosine_similarity': True,
            'use_lateral_inhibition': False,
            'competition_type': 'hard',
            'init_method': 'kaiming_uniform'

        },
        "presynaptic": {
            'mode': 'hard',
            'use_cosine_similarity': True,
            'use_lateral_inhibition': False,
            'init_method': 'kaiming_uniform',
            'use_presynaptic_competition': True,
    },
        "none": {
            'mode': 'basic',
            'use_cosine_similarity': True,
            'use_lateral_inhibition': False,
            'init_method': 'kaiming_uniform',
            'use_presynaptic_competition': False,

        },
    }

    # Update base configuration with version-specific settings
    if version in version_configs:
        base_config.update(version_configs[version])

    return base_config