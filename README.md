
# nanoGPT-lightning

This is a rewrite of [nanoGPT](https://github.com/karpathy/nanoGPT) based on PyTorch Lightning. Currently the file `main.py` can train a model and sample text from it.


## quick start

First try train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```
$ python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. 

```
$ python main.py config/train_gpt2_default.yaml config/train_shakespeare_char.yaml --device=mps --compile=False --eval_iters=2 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

TODO: complete the readme.

