# ----------------------------
# imports for configuration
# ----------------------------
import os
from tensorflow.keras import optimizers  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# hyperparameter configurations
optimizer_configs = [{'type': optimizers.Adam, 'learning_rate': 0.001}]
activations = ['elu']

# batch size configurations
'''
these are selected based on previously best-performing configurations.
'''
batch_sizes = {
    'AE': [32, 128, 512], 
    'DAE': [128, 512, 1024],
    'VAE': [ 2048, 4096]
}

'''
layer configuration Flag:
- set to False to use predefined best manual layer configurations.
'''
USE_AUTOMATED_LAYERS = True 

# manual layer configurations
'''
manual layer configurations:
- these are pre-tuned and best-performing AE/DAE and VAE layer structures.
- example layer: [input_dim, bottleneck, input_dim]
'''

layers_configurations = [
    [95, 90, 95],
    # [95, 85, 95],
    # [95, 80, 95],
    # [95, 75, 95],
    # [95, 70, 95],
    # [95, 85, 95],
    # [95, 75, 95],
    # [95, 65, 95],
    # [95, 55, 95],
    # [95, 45, 95],

    # [95, 90, 85, 90, 95],
    # [95, 85, 75, 85, 95],
    # [95, 80, 65, 80, 95],
    # [95, 75, 60, 75, 95],
    # [95, 70, 55, 70, 95],
    # [95, 85, 75, 85, 95],
    # [95, 75, 55, 75, 95],
    # [95, 65, 45, 65, 95],
    # [95, 55, 35, 55, 95],
    # [95, 45, 25, 45, 95],

    [128, 95, 128],
    # [128, 256, 95, 256, 128],
    
    # [256, 128, 95, 128, 256],
    [256, 95, 256],
    
    # [1024, 95, 1024],
    # [1024, 512, 1024],
    # [1024, 512, 95, 512, 1024],
    # [1024, 512, 256, 128, 95, 128, 256, 512, 1024],
    # [2048, 1024, 2048],
    # [2048, 1024, 512, 1024, 2048],

    # [4096, 2048, 4096],
    # [4096, 2048, 1024, 2048, 4096]

]

vae_layers_configuration = [
    # [95, 90],
    # [95, 85],
    # [95, 80],
    # [95, 75],
    # [95, 70],
    # [95, 85],
    # [95, 75],
    # [95, 65],
    # [95, 55],
    # [95, 45],

    # [95, 90, 85],
    # [95, 85, 75],
    # [95, 80, 65],
    # [95, 75, 60],
    # [95, 70, 55],
    # [95, 85, 75],
    # [95, 75, 55],
    # [95, 65, 45],
    # [95, 55, 35],
    # [95, 45, 25],

    [128, 95],
    # [128, 256, 95],
    
    # [256, 128, 95],
    # [256, 95],
    
    [1024, 95],
    # [1024, 512],
    # [1024, 512, 95],
    # [1024, 512, 256, 128, 95],
    [2048, 1024],
    # [2048, 1024, 512],

    [4096, 2048],
    [4096, 2048, 1024]
]

## automated customized layers ###
def generate_layer_configurations(input_columns_amount, increments=10, z_increments=10, max_z=50, max_configs=1):
    layer_configs = []
    for z in range(0, max_z + z_increments, z_increments):
        z_value = input_columns_amount 
        for x in range(increments, z_value, increments):
            for y in range(increments, z_value, increments):
                hidden_2 = z_value - x
                hidden_3 = hidden_2 - y
                if hidden_3 > 0:
                    config = [z_value, hidden_2, hidden_3, hidden_2, z_value]
                    if config not in layer_configs:
                        layer_configs.append(config)
                    if len(layer_configs) >= max_configs:
                        break
            if len(layer_configs) >= max_configs:
                break
        if len(layer_configs) >= max_configs:
            break

    return layer_configs

def generate_vae_layer_configurations(input_columns_amount, increments=10, z_increments=10, max_z=50, max_configs=1):
    layer_configs = []
    for z in range(0, max_z + z_increments, z_increments):
        z_value = input_columns_amount 

        for x in range(increments, z_value, increments):
            for y in range(increments, z_value, increments):
                hidden_2 = z_value - x
                hidden_3 = hidden_2 - y
                if hidden_3 > 0:
                    config = [z_value, hidden_2, hidden_3]
                    if config not in layer_configs:
                        layer_configs.append(config)
                    if len(layer_configs) >= max_configs:
                        break
            if len(layer_configs) >= max_configs:
                break
        if len(layer_configs) >= max_configs:
            break

    return layer_configs

def parse_layer_config(folder_name):
    try:
        return list(map(int, folder_name.split('_')))
    except ValueError:
        return None

def get_layer_configurations(config, input_columns_amount=95, use_automated_layers=True, combine_layers=True, max_configs=10):
    """
    Returns AE/DAE and VAE layer configurations based on:
    - User-set flags for using manual/automated layers.
    - Existing folders in with_masking / without_masking directories.
    - Auto-generated layer configurations.

    Parameters:
        config (dict): Contains 'base_dir' and masking info.
        input_columns_amount (int): Input dimensionality.
        use_automated_layers (bool): Whether to generate automated layers.
        combine_layers (bool): Whether to combine manual + automated layers.
        max_configs (int): Limit for automated configurations.

    Returns:
        Tuple: (AE/DAE layers, VAE layers)
    """
    base_dir = config['base_dir']
    selected_masking_dir = os.path.join(
        base_dir, 'with_masking' if config.get('masking', True) else 'without_masking'
    )

    ae_dae_folder = os.path.join(selected_masking_dir, 'ae_dae_layers')
    vae_folder = os.path.join(selected_masking_dir, 'vae_layers')

    layers_configurations = []
    vae_layers_configuration = []

    # loading manual layer configs from saved folders
    if os.path.exists(ae_dae_folder):
        ae_dae_layer_folders = [f for f in os.listdir(ae_dae_folder) if os.path.isdir(os.path.join(ae_dae_folder, f))]
        layers_configurations += [parse_layer_config(f) for f in ae_dae_layer_folders if parse_layer_config(f)]

    if os.path.exists(vae_folder):
        vae_layer_folders = [f for f in os.listdir(vae_folder) if os.path.isdir(os.path.join(vae_folder, f))]
        vae_layers_configuration += [parse_layer_config(f) for f in vae_layer_folders if parse_layer_config(f)]

    # generating and optionally combine automated layers
    if use_automated_layers:
        auto_ae_dae = generate_layer_configurations(input_columns_amount, max_configs=max_configs)
        auto_vae = generate_vae_layer_configurations(input_columns_amount, max_configs=max_configs)

        if combine_layers:
            layers_configurations += auto_ae_dae
            vae_layers_configuration += auto_vae
        else:
            layers_configurations = auto_ae_dae
            vae_layers_configuration = auto_vae

    # removing duplicates (some might overlap)
    layers_configurations = [list(x) for x in set(tuple(x) for x in layers_configurations)]
    vae_layers_configuration = [list(x) for x in set(tuple(x) for x in vae_layers_configuration)]

    return layers_configurations, vae_layers_configuration
