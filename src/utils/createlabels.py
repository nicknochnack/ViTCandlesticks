import os 
import pandas as pd 
import numpy as np 

images = os.listdir('data2')
df = pd.DataFrame(images)
df.columns = ['Image']
df['Label'] = np.zeros(len(images))
df.to_csv('data2/labels.csv')
    