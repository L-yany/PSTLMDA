import torch
import argparse

config_args = {
    'training_config': {
        'seed': (2022, int, 'random seed'),
        'lr': (0.001, float, 'learning rate'),
        'wd': (5e-3, float, 'weight decaying'),
        'bs': (128, int, 'batch size'),
        'epochs': (201, int, 'training epochs'),
        'cuda': (torch.cuda.is_available(), bool, 'cuda'),
        'patience': (30, int, 'early stop patience')
    },
    'data_config': {
        'dataset': ('mda', str, 'which dataset to use'),
        'mask_ratio': ('0.7', float, 'the mask ratio in NFM'),
        'dropout_ratio': ('0.7', float, 'the dropout ratio in DropNode'),
        'hidden_channel_t': ('32', int, 'the hidden dim in topology-learning channel'),
        'hidden_channel_s': ('256', int, 'the hidden dim in semantic-learning channel'),
        'out_channel': ('32', int, 'the out dim in two channels'),
        'num_layers_t': ('3', int, 'the number of layers in topology-learning channel'),
        'num_layers_s': ('2', int, 'the number of layers in semantic-learning channel')
    }
}

parser = argparse.ArgumentParser('PSTLMDA')

for _, config_dict in config_args.items():
    for param in config_dict:
        default, type, description = config_dict[param]
        parser.add_argument(f"--{param}", default=default, type=type, help=description)
