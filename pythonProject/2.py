# 加油
# 时间：2023/10/8 10:28

from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path='模型测试集输出.xlsx'

model=keras.models.load_model('test1.keras')


test_data_list1=[[[7454.702179536,0,0,0,10150.542]],[[7454.702179536,1295.04129,157.81125,193.17875,10150.542]],[[7454.702179536,6500,2500,600,10150.542]]]
test_data1=np.array(test_data_list1)


y1=model.predict(test_data1)

df1=pd.DataFrame(y1)
df1.to_excel(path,sheet_name='Sheet2',index=False)