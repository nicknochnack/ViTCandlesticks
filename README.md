# Using Vision Transformers to Predict Candlestick Patterns
Need a reason to go and learn vision transformers? Well...here ya go, one kicka--politically correct project to get you started. 

## See it live and in action 📺 - Click the image!
<!-- <a href="https://youtu.be/D3pXSkGceY0"><img src="https://i.imgur.com/nEfrhIQ.png"/></a> -->
TBD - vid coming in hot.

# Setup 🪛
1. Install UV - `pip install uv`
2. Clone the repo - `git clone https://github.com/nicknochnack/ViTCandlesticks .`
3. Install all the dependencies `uv sync`

# Training 🦾
1. Create a checkpoints folder `mkdir checkpoints`
2. Run the training pipeline `uv run src/train.py`

# Running  🚀 
1. In one terminal, startup the candlesticks pipeline from yahoo finance `uv run src/utils/candlesticks.py`
2. In a separate terminal, run `uv run candlestick_prediction.py`
<p><strong>P.s.</strong> you might need to tweak the dimensions of the screen capture based on your monitor size in `candlestick_prediction.py`. You can do that in this line via albumentations: `A.Crop(x_min=0, y_min=170, x_max=3840, y_max=2160)`</p>


# Who, When, Why?
👨🏾‍💻 Author: Nick Renotte <br />
📅 Version: 1.x<br />
📜 License: This project is licensed under the MIT License </br>
