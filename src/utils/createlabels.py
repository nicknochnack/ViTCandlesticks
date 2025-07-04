import os 
import pandas as pd 
import numpy as np 

images = os.listdir('test_data')
df = pd.DataFrame(images)
df.columns = ['Image']
df['Label'] = np.zeros(len(images))
df.to_csv('test_data/labels.csv')
    