
# nanoGPT-lightning

This is a rewrite of [nanoGPT](https://github.com/karpathy/nanoGPT) based on PyTorch Lightning. Currently the file `main.py` can train a model and sample text from it.


## quick start

First try train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```
$ python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. 

```
$ python main.py config/train_shakespeare_char.yaml --device=cpu --compile=False --limit_val_batches=20 --log_every_n_steps=100 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```


## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```
$ python train.py config/eval_gpt2.yaml --init_from=gpt2
$ python train.py config/eval_gpt2.yaml --init_from=gpt2-medium
$ python train.py config/eval_gpt2.yaml --init_from=gpt2-large
$ python train.py config/eval_gpt2.yaml --init_from=gpt2-xl
```

TODO: complete the readme.


