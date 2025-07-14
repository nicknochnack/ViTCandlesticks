from PIL import Image
from mss import mss
from model import ViT
from torch import load 
import torch 
import time
import numpy as np
from colorama import Fore
import cv2 
import uuid
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import os 

model = ViT()
model.load_state_dict(torch.load('checkpoints/100_model.pt', weights_only=True, map_location=torch.device('cpu')))
model.eval() 

classes = {
    '0': 'doji', 
    '1': 'bullish_engulfing',
    '2': 'bearish_engulfing',
    '3': 'morning_star',
    '4': 'evening_star',
}
transforms = A.Compose(
    [
        A.Crop(x_min=0, y_min=170, x_max=3840, y_max=2160),
        A.Resize(700,500),
        A.Resize(224,224), 
        A.Crop(x_min=128, y_min=38, x_max=200, y_max=158),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ]
)

with mss() as sct: 
    for x in range(500): 
        sct_image = sct.grab(sct.monitors[0])
        raw_image = Image.frombytes("RGB", sct_image.size, sct_image.rgb)
        # raw_image = raw_image.crop((0,170,3840, 2160))
        # raw_image = raw_image.resize((700,500))
        res = transforms(image=np.array(raw_image)[:,:,:3])
        img = res['image']

        # IMPORTANT 
        # raw_image = Image.open('test_cap/5b55ebb2-6054-11f0-aac3-ab00b4bd9398.png')
        # raw_image = raw_image.resize((224,224))
        # raw_image = raw_image.crop((128,38,200,158))
        # raw_image.save('rawscreencap.png') 
        
  
        softy = torch.nn.Softmax(dim=1)
        preds = model(torch.unsqueeze(img, dim=0))
        probs = softy(preds)
        prediction = torch.argmax(probs, dim=-1)[0]
        probability = probs[0][int(prediction)]
        print(Fore.LIGHTYELLOW_EX +  str(prediction) + ' ' + str(probability) + Fore.RESET)

        
        img_np = img.permute(1,2,0).numpy()
        img_min, img_max = img_np.min(), img_np.max()
        img_np_scaled = (img_np - img_min) / (img_max - img_min)
        plt.imsave(f"test_live_preds_cached/{uuid.uuid1()}---{prediction}_{probability}.png", img_np_scaled)
            

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