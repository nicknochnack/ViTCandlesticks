import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import io
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import ViT  # senin kendi modelin

# Model yükle
model = ViT()
model.load_state_dict(torch.load("checkpoints/25_model.pt", map_location="cpu"))
model.eval()
print(model)

# Label dictionary
classes = {
    0: "doji",
    1: "bullish_engulfing",
    2: "bearish_engulfing",
    3: "morning_star",
    4: "evening_star",
}

# Transform pipeline (train ile aynı!)
transforms = A.Compose([
    A.Resize(104, 72),   # eğitimde kullandığın size
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Veri kaynağı
def get_candles(stock="AMZN", n=20):
    dat = yf.Ticker(stock)
    df = dat.history(period="1d", interval="1m").reset_index()
    return df.tail(n)  # son 20 candle

# Candle chart'ı resme çevir
def candles_to_image(df: pd.DataFrame) -> Image.Image:
    fig, ax = plt.subplots(figsize=(3,3))  # küçük plot
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

# Prediction
def predict_from_candles(df, show_img=True):
    img = candles_to_image(df)   # PIL image
    if show_img:
        img.show()  # default image viewer'da açar (Windows/Mac/Linux)
        img.save("image.png")
        # alternatif: matplotlib ile inline göstermek
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    img_np = np.array(img)
    res = transforms(image=img_np)
    tensor = res["image"].unsqueeze(0)  # (1,3,104,72)
    preds = model(tensor)
    probs = torch.nn.functional.softmax(preds, dim=1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    return classes[pred_idx], float(probs[0][pred_idx])

# Test
if __name__ == "__main__":
    df = get_candles("AMZN")
    pred, prob = predict_from_candles(df)
    print(f"Tahmin: {pred} (prob={prob:.3f})")
