from tensorflow import keras
import pandas as pd
import numpy as np
from keras.losses import MeanSquaredLogarithmicError
import matplotlib.pyplot as plt

path=r'训练数据.xlsx'
df=pd.read_excel(path,sheet_name='Sheet1')
value=df.values
df1 = pd.read_excel(path, sheet_name='Sheet3')
value1 = df1.values.tolist()

df2 = pd.read_excel(path, sheet_name='Sheet2')
value2 = df2.values.tolist()

train_data_list1=[]
train_data_list2=[]
train_result_list=[]

for i in range(value.shape[0]):
    train_data_list1.append([value[i]])
for j in range(len(value1)):
    if (j+1)%12==0:
        mid2=value1[j-11:j+1]
        mid1=value2[j-11:j+1]
        train_result_list.append(mid2)
        train_data_list2.append(mid1)

input_data1=np.array(train_data_list1)
input_data2=np.array(train_data_list2)
result_data=np.array(train_result_list)
test_data1=np.array([[[27537.02,1348,178,677,15461.92]]])
test_data2=np.array([[[28.56 ],[28.22],[28.61],[27.21],[25.5],[26.91],[27.21],[30.46],[31.95],[33.43],[33.37],[33.86]]])


#输入层
input_1 = keras.layers.Input(shape=(1, 5))
input_2 = keras.layers.Input(shape=(12, 1))

# 处理 input_1 的网络层
output_1_layer = keras.layers.Bidirectional(keras.layers.LSTM(120, return_sequences=True))(input_1)
output_1_layer = keras.layers.Bidirectional(keras.layers.LSTM(120))(output_1_layer)
# 处理 input_2 的网络层
output_2_layer = keras.layers.Bidirectional(keras.layers.LSTM(120, return_sequences=True))(input_2)
output_2_layer = keras.layers.Bidirectional(keras.layers.LSTM(120))(output_2_layer)

# 将两个输出层合并
merged_layer = keras.layers.concatenate([output_1_layer, output_2_layer])

# 共享的 Dense 层
dense_layer = keras.layers.Dense(12)(merged_layer)

# 输出层调整形状以适应标签数据
output_layer = keras.layers.Reshape((12, 1))(dense_layer)

# 构建多输入模型
model = keras.models.Model(inputs=[input_1, input_2], outputs=output_layer)


# 使用模型进行训练
learning_rate=0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='MSE')
model.fit([input_data1, input_data2], result_data, epochs=40,batch_size=1)
model.save('test1.keras')
y=model.predict([test_data1,test_data2])

or_level=33.86
test_result=[34.31,34.29,34.25,34.59,33.72,33.08,33.73,34.28,34.83,34.44,33.62,33.57]
pre_result=[]
for  m in range(y.shape[0]):
    for n in range(y.shape[1]):
        if n==0:
            k=or_level+y[m,n,0]
        else:
            k=pre_result[n-1]+y[m,n,0]
        pre_result.append(k)

x=[p+1 for p in range(len(test_result))]

plt.plot(x,test_result,label='ture')
plt.plot(x,pre_result,label='predict')
plt.legend(loc='best')
plt.show()