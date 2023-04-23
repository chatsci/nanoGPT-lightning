"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python main.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 main.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 main.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 main.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import tiktoken

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import GPTConfig as ModelConfig
from model import GPTLightning as ModelLightning
from model import GPTCallback
from data import LanguageModelDataModule
from util import dotdict, load_config, get_vocab_size_from_meta, clean_state_dict, get_num_workers


def init_model_from_scratch(config):
    # create the model
    print("initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    # if there is a meta file, we attempt to derive vocab_size from the dataset
    meta_path = os.path.join(config.data_dir, 'meta.pkl')
    config.meta_path = meta_path
    meta_vocab_size = get_vocab_size_from_meta(meta_path)
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    config.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304

    model = ModelLightning(config)

    return model, config


def init_model_from_ckpt(config, ckpt_path=None):
    print(f"resuming training from checkpoint: {config.out_dir}")
    if ckpt_path is None:
        ckpt_path = os.path.join(config.out_dir, 'best_ckpt.pt') # !!! NOTE: how the best check point is named.
    config.ckpt_path = ckpt_path

    model = ModelLightning.load_from_checkpoint(ckpt_path)

    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    configs_to_sync = [field.name for field in fields(ModelConfig)].remove('dropout')
    checkpoint_model_args = model.hparams
    for k in configs_to_sync:
        config[k] = checkpoint_model_args[k]

    return model, config


def init_model_from_gpt2(config):
    print(f"initializing from OpenAI GPT-2 weights: {config.init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=config.dropout)
    model = ModelLightning.from_pretrained(config.init_from, override_args)

    # crop down the model block size if desired, using model surgery
    if config.block_size < model.config.block_size:
        model.crop_block_size(block_size)
        # so that the checkpoint will have the right value
        # model_args['block_size'] = config.block_size
    return model, config


def run_sample(model, max_new_tokens=500, temperature=0.8, top_k=200, load_meta=True):
    """
    Generate text samples from a trained GPT model.

    Args:
        model (pl.LightningModule): The trained GPT model.
        max_new_tokens (int): The maximum number of tokens to generate for each sample.
        temperature (float): Controls the degree of randomness in the generated samples.
        top_k (int): Controls the number of candidate tokens to consider at each step.
        load_meta (bool): Whether or not to load the tokenizer metadata.

    Returns:
        None.
    """
    model.eval()

    # Load tokenizer metadata
    if load_meta:
        print(f"Loading meta from {model.config.meta_path}...")
        with open(model.config.meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # Generate text samples
    while True:
        # Get start text from user input
        start = input("Enter starting text: ")
        start_ids = encode(start)
        x = torch.tensor(start_ids, dtype=torch.long, device=model.device)[None, ...]
        with torch.no_grad():
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('-' * 80)
        
        # Ask user if they want to generate more samples
        more_samples = input("Generate more samples? [y/n]: ")
        if more_samples.lower() in ["n", "no", "nope"]:
            break


def train():
    # set random seeds for reproducibility
    pl.seed_everything(1337)

    # load configure settings
    config = {}
    config = dotdict(load_config(config))
    print(config)

    os.makedirs(config.out_dir, exist_ok=True)
    config.data_dir = os.path.join('data', config.dataset)

    # create the model
    if config.init_from == 'scratch':
        model, config = init_model_from_scratch(config)
    elif config.init_from in ['resume_ckpt', 'load_ckpt']:
        model, config = init_model_from_ckpt(config)
    elif config.init_from.startswith('gpt2'):
        model, config = init_model_from_gpt2(config)

    # compile the model
    if config.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0


    # create the dataset and dataloaders
    config.ddp = int(os.environ.get('RANK', -1)) != -1
    num_workers = get_num_workers(use_ddp=config.ddp)
    leave_cpu_cores_free = 2
    print("Num_wokers: ", num_workers)
    data_module = LanguageModelDataModule(
        config.data_dir,
        config.batch_size,
        config.block_size,
        max(1, num_workers - leave_cpu_cores_free)
    )
    data_module.setup(stage='fit')

    # set up the WandbLogger
    wandb_logger = WandbLogger(project=config.project, log_model=True)

    # set up the ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.out_dir,
        filename='ckpt-{epoch:02d}-{val_loss:.2f}.pt',  # Save models with epoch and val_loss
        save_top_k=-1 if config.always_save_checkpoint else 1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_last=True  # Save the latest model, regardless of its performance. The name is last.ckpt
    )

    model_callback = GPTCallback(config.log_interval, config.gradient_accumulation_steps)

    # set up the Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, model_callback],
        max_steps=config.max_iters,
        accelerator='ddp' if config.ddp else 'auto',
        precision=16 if config.mixed_precision else 32,
        deterministic=True,
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        log_every_n_steps=config.log_every_n_steps if config.log_every_n_steps else config.eval_interval
    )
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=config.ckpt_path if config.init_from == 'resume_ckpt' else None
    )

    # After training, rename the best model to 'best_ckpt.pt'
    print("checkpoint_callback.best_model_path: ", checkpoint_callback.best_model_path)
    if checkpoint_callback.best_model_path != "": # if the first epoch is not complete, then the file won't exist.
        os.rename(checkpoint_callback.best_model_path, os.path.join(config.out_dir, 'best.ckpt'))

    print("trainer is done")

    # # run test set
    # data_module.setup(stage='test')
    # result = trainer.test()
    # print(result)
    # print("test is done")
    return model, config


def sample():
    # set random seeds for reproducibility
    pl.seed_everything(1337)

    # load configure settings
    config = {}
    config = dotdict(load_config(config))
    print(config)

    os.makedirs(config.out_dir, exist_ok=True)
    config.data_dir = os.path.join('data', config.dataset)

    # create the model
    if config.init_from == 'scratch':
        model, config = init_model_from_scratch(config)
    elif config.init_from in ['resume_ckpt', 'load_ckpt']:
        model, config = init_model_from_ckpt(config)
    elif config.init_from.startswith('gpt2'):
        model, config = init_model_from_gpt2(config)

    # compile the model
    if config.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # Run the sample step
    run_sample(model)


if __name__ == '__main__':
    model, config = train()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main_process = local_rank == 0
    if is_main_process:
        run_sample(model)


