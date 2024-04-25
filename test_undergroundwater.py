
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
path=r'C:\Users\程帅\Desktop\地下水1.xlsx'
df = pd.read_excel(path, sheet_name='Sheet1')
value = df.values.tolist()

df1 = pd.read_excel(path, sheet_name='Sheet2')
value1 = df1.values.tolist()

train_data_list = []
train_result_list = []
test_data_list = []
test_result_list = []

for i in range(len(value)):
    if (i + 1) % 12 == 0 and i <= 191:
        mid1 = value[i - 11:i + 1]
        mid2 = value1[i - 11:i + 1]
        train_data_list.append(mid1)
        train_result_list.append(mid2)
    elif (i + 1) % 12 == 0 and i > 191:
        mid3 = value[i - 11:i + 1]
        mid4 = value1[i - 11:i + 1]
        test_data_list.append(mid3)
        test_result_list.append(mid4)

input_data = np.array(train_data_list)
target_data = np.array(train_result_list)
test_data = np.array(test_data_list)



tu=[]
for j in range(len(test_result_list)):
    for k in range(len(test_result_list[j])):
        tu.append(test_result_list[j][k][0])



model = keras.models.Sequential([
    keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True), input_shape=(15, 3)),
    keras.layers.Bidirectional(keras.layers.LSTM(100)),
    keras.layers.Dense(15)
])

learning_rate=0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
model.fit(input_data,result_data,epochs=100,batch_size=1)

model.save('undergroundwater.keras')





y=model.predict(test_data)

pr=[]
for m in range(y.shape[0]):
    for f in range(y.shape[1]):
        pr.append(y[m,f])


pc=[]
ture=[]
for pl in range(len(tu)):
    if (pl+1)%15==0:
        pc.append(pr[pl-14])
        pc.append(pr[pl - 13])
        pc.append(pr[pl - 12])
        ture.append(tu[pl-14])
        ture.append(tu[pl - 13])
        ture.append(tu[pl - 12])


df2=pd.DataFrame(pc)

df2.to_excel(r'C:\Users\程帅\Desktop\11.xlsx',sheet_name='Sheet1')



x=[n+1 for n in range(len(ture))]
fig=plt.figure()
l1,=plt.plot(x,pc,c='r',label='predict')
l2,=plt.plot(x,ture,c='blue',label='ture')
plt.legend(loc='best')
plt.show()