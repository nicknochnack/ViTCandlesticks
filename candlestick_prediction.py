from PIL import Image
from mss import mss
from model import ViT
from torch import load 
import torch 
import time
import numpy as np
from colorama import Fore
import cv2 

model = ViT()
model.load_state_dict(torch.load('checkpoints/20_model.pt', weights_only=True, map_location=torch.device('cpu')))
model.eval() 

classes = {
    '0': 'nothing',
    '1': 'doji', 
    '2': 'bullish_engulfing',
    '3': 'bearish_engulfing',
    '4': 'morning_star',
    '5': 'evening_star',
}
    

with mss() as sct: 
    for x in range(500): 
        sct_image = sct.grab(sct.monitors[0])
        raw_image = Image.frombytes("RGB", sct_image.size, sct_image.bgra, "raw", "BGRX")
        img = raw_image.crop((2280,420,3650,1120)).resize((72,120))
        img_tensor = torch.tensor(np.array(img)).permute(2,0,1) / 255.0
        softy = torch.nn.Softmax(dim=1)
        preds = model(torch.unsqueeze(img_tensor, dim=0))
        probs = softy(preds)
        prediction = torch.argmax(probs, dim=-1)[0]
        probability = probs[0][int(prediction)]
        print(Fore.LIGHTYELLOW_EX +  str(prediction) + ' ' + str(probability) + Fore.RESET)

        # Showing BB
        render_image = cv2.cvtColor(np.array(raw_image.crop((0,0,3840,2160))), cv2.COLOR_BGR2RGB)
        # Big bb
        render_image = cv2.rectangle(render_image, (2300,400), (3640,1100), (0,255,255), 10)
        # Label bb
        render_image = cv2.rectangle(render_image, (2300,260), (3500,400), (0,255,255), -1)
        # Label text
        label = str(classes[str(int(prediction))]) +' - '+ str(round(float(probability),3))

        render_image = cv2.putText(render_image, label, (2300,350), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 8, cv2.LINE_AA)
        time.sleep(1) 
        cv2.imshow('Frame', render_image)
        if cv2.waitKey(1) and 0xFF == ord('q'): 
            break