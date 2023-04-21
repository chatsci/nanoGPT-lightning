"""
Note: split this file into different sub files later.
"""
import os
import sys
import random
import torch
import yaml
import numpy as np
from contextlib import nullcontext
import pickle
from torch.distributed import init_process_group
from torch.cuda import device_count


def get_num_workers(use_ddp: bool = False):
    """
    Determine the appropriate number of DataLoader workers based on available resources.
    
    Args:
        use_ddp (bool, optional): Set to True if using distributed data parallel (DDP) training. 
                                  Defaults to False.
    
    Returns:
        int: The recommended number of workers for DataLoader.
    """
    num_workers = 0
    num_cores = os.cpu_count()

    # Check if GPU is available
    if torch.cuda.is_available():
        # If DDP is used, divide the num_cores by the number of GPUs
        if use_ddp:
            num_gpus = device_count()
            num_workers = max(1, num_cores // num_gpus)
        else:
            num_workers = max(1, num_cores // 2)
    else:
        num_workers = max(1, num_cores // 2)

    return num_workers


def clean_state_dict(state_dict, unwanted_prefix='_orig_mod.'):
    """
    Remove the unwanted prefix from the state dictionary keys.
    
    Sometimes during saving and loading model checkpoints, the state dictionary keys
    might get an unwanted prefix due to the specific way certain modules or wrappers
    are implemented. In particular, the `_orig_mod` prefix might appear when using
    DistributedDataParallel (DDP) with the "find_unused_parameters" flag set to True.

    Args:
        state_dict (OrderedDict): Model's state dictionary.
        unwanted_prefix (str): The unwanted prefix to remove.

    Returns:
        OrderedDict: Cleaned state dictionary without the unwanted prefix.
    """
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_key = k[len(unwanted_prefix):] if k.startswith(unwanted_prefix) else k
        cleaned_state_dict[cleaned_key] = v
    return cleaned_state_dict



def configure_ddp_and_device(config):
    """
    Configure the DistributedDataParallel (DDP) and device settings.

    Args:
        config: A configuration object with the properties 'backend', 'device', 'batch_size', 
                'block_size', 'gradient_accumulation_steps', and other project-specific settings.

    Returns:
        config: The updated configuration object with the device, master_process, and seed_offset properties.
    """
    # Check if this is a DistributedDataParallel (DDP) run
    ddp = int(os.environ.get('RANK', -1)) != -1
    config.ddp = ddp
    if ddp:
        init_process_group(backend=config.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        config.device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(config.device)
        # This process will do logging, checkpointing, etc.
        config.master_process = ddp_rank == 0
        config.seed_offset = ddp_rank  # Each process gets a different seed
    else:
        # If not DDP, we are running on a single GPU and one process
        config.master_process = True
        config.seed_offset = 0
        config.gradient_accumulation_steps *= 8  # Simulate 8 GPUs

    print("total number of tokens per iteration:", config.batch_size *
          config.block_size * config.gradient_accumulation_steps)

    return config



def get_vocab_size_from_meta(meta_path: str):
    """
    Get the vocab_size from the meta.pkl file in the data directory.
    
    Args:
        data_dir (str): The directory containing the dataset and meta.pkl file.

    Returns:
        meta_vocab_size (int): The vocab size from the meta.pkl file or None if the file does not exist.
    """
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    return meta_vocab_size



def set_mixed_precision(config):
    """
    Enable mixed precision training by setting the appropriate flags and contexts in PyTorch.

    Args:
        config (argparse.Namespace): Configuration options, containing the following fields:
            - device (str): Device to use for training ('cuda' or 'cpu').
            - dtype (str): Data type to use for training ('float32', 'bfloat16', or 'float16').

    Returns:
        ctx (contextlib.AbstractContextManager): PyTorch context manager for mixed precision training.
    """
    # Allow tf32 on matmul and cudnn
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Determine device type based on config
    config.device_type = 'cuda' if 'cuda' in config.device else 'cpu'

    # Determine PyTorch data type based on config
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]

    # Create context manager for mixed precision training
    ctx = nullcontext() if config.device_type == 'cpu' else torch.amp.autocast(device_type=config.device_type, dtype=ptdtype)

    return config, ctx


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device(no_cuda=False):
    use_cuda = torch.cuda.is_available() and not no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    if use_cuda:
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu")
    return device, use_cuda, n_gpu


def set_logger(log_file):
    logger = sys.stdout
    if log_file is not None:
        logger = open(log_file, "a")
    return logger


def summarize_model(model):
    total_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, ":", p.size(), np.prod(p.size()))
            total_params += np.prod(p.size())
    print('Trainable trainable parameters:', total_params)

class dotdict(dict):
    """
    A dictionary supporting dot notation.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = dotdict(v)

    def lookup(self, dotkey):
        """
        Lookup value in a nested structure with a single key, e.g. "a.b.c"
        """
        path = list(reversed(dotkey.split(".")))
        v = self
        while path:
            key = path.pop()
            if isinstance(v, dict):
                v = v[key]
            elif isinstance(v, list):
                v = v[int(key)]
            else:
                raise KeyError(key)
        return v

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, dotdict) else v for k, v in self.items()}


def load_config(config):
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                # config = yaml.safe_load(f)
                config.update(yaml.safe_load(f)) # this aims to load several different config files.
                # print(yaml.dump(config))
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, val = arg.split('=')
            key = key[2:]
            if key in config:
                try:
                    # attempt to eval it it (e.g. if bool, number, or etc)
                    attempt = yaml.safe_load(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok
                assert type(attempt) == type(config[key])
                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                config[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")
    return config

