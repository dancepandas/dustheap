# 加油
# 时间：2023/10/8 9:37
#导入库
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import PReLU
from keras.losses import MeanSquaredLogarithmicError

#利用pandas调用处理Excel数据
path=r'模型训练输出数据.xlsx'   #定义变量
df=pd.read_excel(path,sheet_name='训练输入数据')    #df，即DataFrame是pandas中一个重要的数据结构，类似于一张表格，可以存储和操作二维数据。
value=df.values
#将DataFrame对象的值转化为一个列表，方便后续处理。
df1 = pd.read_excel(path, sheet_name='训练输出数据')
value1 = df1.values.tolist()

#建立一个空的列表，用来存储
train_data_list =[]
train_result_list=[]
test_data_list=[[[5257.415208,2003.93709,3.45,296.16,8482.542]]]
test_result_list=[[[-0.52,0.46,0.03,-0.22	,-0.31,0.45,-0.04,1.15,0.35,-0.06,0.04,0.08]]]

for i in range(value.shape[0]):
    train_data_list.append([value[i]])
for j in range(len(value1)):
    if (j+1)%12==0:
        mid2=value1[j-11:j+1]
        train_result_list.append(mid2)

#np.array：将列表转为数组（列表没有维度，数组有维度）
input_data=np.array(train_data_list) #输入数据集
print(input_data.shape)   #.shape形状信息描述了数组的维度和每个维度中的元素数量
result_data=np.array(train_result_list)  #结果数据集
print(result_data.shape)
test_data=np.array(test_data_list)   #测试数据集
print(test_data.shape)


#构建模型？
model = keras.models.Sequential([
    keras.layers.Bidirectional(keras.layers.LSTM(120, return_sequences=True), input_shape=(1, 5)), #输入层
    keras.layers.Bidirectional(keras.layers.LSTM(120)), #反向层
    keras.layers.Dense(12)  #输出层   PReLU为激活函数
])

# 设置学习率
learning_rate=0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=MeanSquaredLogarithmicError())
model.fit(input_data,result_data,epochs=1000,batch_size=1)

model.save('undergroundwater.keras')

y=model.predict(test_data)  #预测结果
print(y)
#数组转化为列表

'''#数据输出保存
df2=pd.DataFrame(pr)
df2.to_excel(r'模型测试集输出',sheet_name='Sheet1')'''

#可视化
x=[n+1 for n in range(12)]
fig=plt.figure()
l1,=plt.plot(x,y[0],c='r',label='predict')
l2,=plt.plot(x,test_result_list[0][0],c='blue',label='ture')
plt.legend(loc='best')
plt.show()