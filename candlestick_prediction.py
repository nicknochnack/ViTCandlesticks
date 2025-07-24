from mss import mss
from PIL import Image
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
model.load_state_dict(torch.load('checkpoints/25_model.pt', weights_only=True, map_location=torch.device('cpu')))
model.eval() 

classes = {
    '0': 'doji', 
    '1': 'bullish_engulfing',
    '2': 'bearish_engulfing',
    '3': 'morning_star',
    '4': 'evening_star',
}
bar_colors = [
    (156, 220, 235),(166, 207, 140),(236, 171, 193),(202, 163, 232),(255, 128, 128)
]
transforms = A.Compose(
    [
        A.Crop(x_min=0, y_min=170, x_max=3840, y_max=2160),
        A.Resize(700,500),
        A.Resize(224,224), 
        # A.Crop(x_min=172, y_min=43, x_max=204, y_max=147), # three candles
        A.Crop(x_min=130, y_min=43, x_max=202, y_max=147), # eight candles
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ]
)

with mss() as sct: 
    while True:
    # for x in range(500): 
        sct_image = sct.grab(sct.monitors[0])
        raw_image = Image.frombytes("RGB", sct_image.size, sct_image.rgb)
        # raw_image = raw_image.crop((0,170,3840, 2160))
        # raw_image = raw_image.resize((700,500))
        # raw_image.save(f'zz_hand_labelled/{uuid.uuid1()}.png') 
        # time.sleep(1)
        res = transforms(image=np.array(raw_image)[:,:,:3])
        img = res['image']

        # IMPORTANT 
        # raw_image = Image.open('data/test_data/screengrabs/8a9cda74-60fa-11f0-92df-2b583dcc4c21.png')
        # raw_image = raw_image.resize((224,224))
        # raw_image = raw_image.crop((128,38,200,158)) #three candles
        # raw_image = raw_image.crop((130,43,202,147)) #eight candles
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
        # plt.imsave(f"test_live_preds_cached/{uuid.uuid1()}---{prediction}_{probability}.png", img_np_scaled)

        raw_image.save('example_ss.png') 
        # Showing BB
        render_image = cv2.cvtColor(np.array(raw_image.crop((0,0,3840,2160))), cv2.COLOR_BGR2RGB)
        
        overlay = render_image.copy()
        cv2.rectangle(overlay, (1350,590), (2224, 1076), (128, 128, 128), -1) 
        render_image = cv2.addWeighted(overlay, 0.7, render_image, 1 - 0.7, 0)

        # Bar chart rolling
        for x in range(len(classes.keys())):
            class_name = str(classes[str(x)])
            class_prob = float(probs[0][x])
            label = class_name +' - '+ str(round(class_prob,3))
            y_min = (x+1)*100+490
            y_max = y_min + 100 
            x_max = 1350+int(800*class_prob)
            render_image = cv2.rectangle(render_image, (1350,y_min), (x_max, y_max), bar_colors[x], -1) 
            render_image = cv2.putText(render_image, label, (1350,y_min+75), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 3, cv2.LINE_AA)        

        # Big bb
        render_image = cv2.rectangle(render_image,(2224,590), (3420,1455), bar_colors[int(prediction)], 10)
        # render_image = cv2.rectangle(render_image, (2300,400), (3640,1100), (0,255,255), 10)
        # Label bb
        render_image = cv2.rectangle(render_image, (1350,400), (3420,590), bar_colors[int(prediction)], -1)
        # Label text
        label = str(classes[str(int(prediction))]) +' - '+ str(round(float(probability),3))

        render_image = cv2.putText(render_image, label, (1350,550), cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,0), 8, cv2.LINE_AA)
        
        cv2.imshow('Frame', render_image)
        time.sleep(0.1) 
        if cv2.waitKey(1) and 0xFF == ord('q'): 
            cv2.destroyAllWindows() 
            break
        