1. Install UV - `pip install uv`
2. Clone the repo - `git clone https://github.com/nicknochnack/ViTCandlesticks .`
3. Install all the dependencies `uv sync`

To Do
- Load pretrained weights: 
    https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html or 
    
<!-- Note torchvision parameters. -->
torchrun --nproc_per_node=8 train.py\
    --model vit_b_16 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema
