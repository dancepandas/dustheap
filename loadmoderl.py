# 加油
# 时间：2023/10/8 10:28

from tensorflow import keras
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimSun']# 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

model=keras.models.load_model('undergroundwater.keras')


test_data1=np.array([[[5257.415208,2003.93709,3.45,296.16,8482.542]]])
test_data2=np.array([[[28.44 ],[28.30 ],[28.28],[27.64],[26.75],[26.25],[28.21],[30.85],[33.65],[35.16],[35.56],[35.79]]])

y=model.predict([test_data1,test_data2])

or_level=35.79
test_result=[35.27,35.73,35.76,35.54,35.23,35.68,35.64,36.79,37.14,37.08,37.12,37.20]
pre_result=[]
for  m in range(y.shape[0]):
    for n in range(y.shape[1]):
        if n==0:
            k=or_level+y[m,n,0]
        else:
            k=pre_result[n-1]+y[m,n,0]
        pre_result.append(k)

x=[p+1 for p in range(len(test_result))]

fig=plt.figure(figsize=(16,16))
plt.scatter(x,test_result,label='实测水位',s=65)
plt.scatter(x,pre_result,label='预测水位',s=65)
plt.ylim(30,50)
plt.ylabel('水位（m）',fontsize=35)
plt.xlabel('时间（月）',fontsize=35)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='best',fontsize=25)
plt.show()