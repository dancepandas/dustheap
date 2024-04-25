# 加油
# 时间：2023/10/8 9:37

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import PReLU
from keras.losses import MeanSquaredLogarithmicError

path=r'模型训练输出数据.xlsx'


df=pd.read_excel(path,sheet_name='训练输入数据')
value=df.values


df1 = pd.read_excel(path, sheet_name='训练输出数据')
value1 = df1.values

input_data_list=[]
result_data_list=[]
test_data_list=[]
ture_data_list=[]
for i in range(value.shape[0]):
    if (i+1)%12==0 and i!=0 and i<value.shape[0]-12:
        input_data_list.append(value[i-11:i+1])
        result_data_list.append(value1[i-11:i+1])
    elif (i+1)%12==0  and i>value.shape[0]-12:
        test_data_list.append(value[i-11:])
        ture_data_list.append(value1[i-11:])


input_data=np.array(input_data_list)
result_data=np.array(result_data_list)
test_data=np.array(test_data_list)
ture_data=np.array(ture_data_list)


model = keras.models.Sequential([
    keras.layers.Normalization(),
    keras.layers.Bidirectional(keras.layers.LSTM(120, return_sequences=True), input_shape=(12, 5)),
    keras.layers.Bidirectional(keras.layers.LSTM(60)),
    keras.layers.Dense(12,activation=PReLU())
])

# 设置学习率
learning_rate=0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
print(input_data.shape,result_data.shape)
model.fit(input_data,result_data,epochs=1000,batch_size=12)

model.save('undergroundwater.keras')

y=model.predict(test_data)



df2=pd.DataFrame(y)

df2.to_excel(r'模型测试集输出.xlsx',sheet_name='Sheet1')



x=[n+1 for n in range(len(y[0]))]
fig=plt.figure()
l1,=plt.plot(x,y[0],c='r',label='predict')
l2,=plt.plot(x,ture_data[0],c='blue',label='ture')
plt.legend(loc='best')
plt.show()
