import ccxt
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

from src.model import ViT

# Binance client (ccxt)
api_key = 'Ds9XkUhUzW5dXIkwdpUsNq05esaCZ7yCN4s5xgV0MBqQi0qC7ggXH119kx0pEflt'
api_secret = 'uDoI6aATw0LbvwroJhPt2tkIWr9OR88a9ed0ZURCsz4PBBWQcnLH7rKZ77gfKsl4'
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})

# Model yükle
model = ViT()
model.load_state_dict(torch.load("checkpoints/25_model.pt", map_location="cpu"))
model.eval()

# Label dictionary
classes = {
    0: "doji",
    1: "bullish_engulfing",
    2: "bearish_engulfing",
    3: "morning_star",
    4: "evening_star",
}

# Transform pipeline
transforms = A.Compose([
    A.Resize(104, 72),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Son n candle al (ccxt)
def get_candles(symbol="BTC/USDT", timeframe="1m", n=20):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=n)
    df = pd.DataFrame(ohlcv, columns=["Open time","Open","High","Low","Close","Volume"])
    df["Open time"] = pd.to_datetime(df["Open time"], unit='ms')
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    return df

# Candle grafiği -> PIL Image
def candles_to_image(df: pd.DataFrame) -> Image.Image:
    fig, ax = plt.subplots(figsize=(1.04, 0.72))
    for i, row in df.iterrows():
        color = "green" if row["Close"] >= row["Open"] else "red"
        ax.plot([i, i], [row["Low"], row["High"]], color=color)
        ax.plot([i, i], [row["Open"], row["Close"]], color=color, linewidth=5)
    ax.axis("off")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf).convert("RGB")

# Tahmin
def predict_from_candles(df):
    img = candles_to_image(df)
    img_np = np.array(img)
    res = transforms(image=img_np)
    tensor = res["image"].unsqueeze(0)
    preds = model(tensor)
    probs = torch.nn.functional.softmax(preds, dim=1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    return classes[pred_idx], float(probs[0][pred_idx])

# Main loop
if __name__ == "__main__":
    symbol = "BTC/USDT"
    last_time = None
    while True:
        df = get_candles(symbol)
        newest_candle_time = df["Open time"].iloc[-1]
        if newest_candle_time != last_time:  # yeni candle varsa
            pred, prob = predict_from_candles(df)
            print(f"{pd.Timestamp.now()} - Tahmin: {pred} (prob={prob:.3f})")
            last_time = newest_candle_time
        else:
            print(f"{pd.Timestamp.now()} - Yeni veri yok")
        time.sleep(10)  # 10 saniyede bir kontrol et
