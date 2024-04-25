# 加油
# 时间：2023/10/8 10:28

from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path='模型测试集输出.xlsx'

model=keras.models.load_model('test1.keras')


test_data_list1=[[[30734.87,2000,3000,900,9986.924]]]
test_data1=np.array(test_data_list1)

test_data=np.array([[[34.31 ],[34.29],[34.25],[34.59],[33.72],[33.08],[33.73],[34.28],[34.83],[34.44],[33.62],[33.57]]])
y1=model.predict([test_data1,test_data])

df1=pd.DataFrame(y1[0])
df1.to_excel(path,sheet_name='Sheet2',index=False)