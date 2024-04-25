import copy
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

def creat_pop(tz,dim):
    popw=[]
    for i in range(dim):
        w=[0 for i in range(tz)]
        for n in range(tz):
            if n==3:
                w[n]=random.uniform(-1,0)
            else:
                w[n]=random.random()
        popw.append(w)
    return np.array(popw)

def fun(input_data,result_data,popw):
    mse_list=[]
    for i in range(popw.shape[0]):
        loss_total=0
        for j in range(input_data.shape[0]):
            a=copy.copy(input_data[j])
            dot_data=np.dot(a,popw[i])
            loss=np.sqrt(np.mean(np.square(result_data[j].T[0]-dot_data)))
            loss_total+=loss
        mse_list.append(loss_total/input_data.shape[0])
    return mse_list

def crossover(popw,mse_list,c_precent):
    popw_new_list=list(copy.copy(popw))
    mse_ar=np.array(mse_list)
    arg=np.argsort(mse_ar)
    popw_rank=popw[arg]
    for i in range(popw_rank.shape[0]):
        k=random.random()
        if k>c_precent:
            index=random.randint(0,popw_rank.shape[0]-1)
            crossover_one=popw_rank[i]
            crossover_two=popw_rank[index]
            m=random.randint(2,popw_rank.shape[1]-1)
            crossover_one_new=np.concatenate((crossover_one[:m],crossover_two[m:]),axis=0)
            crossover_two_new =np.concatenate((crossover_two[:m],crossover_one[m:]),axis=0)
            popw_new_list.append(crossover_one_new)
            popw_new_list.append(crossover_two_new)
    popw_new=np.array(popw_new_list)
    return popw_new

def mutation(popw,m_precent):
    popw_new_list=list(copy.copy(popw))
    for i in range(popw.shape[0]):
        k=random.random()
        if k>m_precent:
            mutation_one=popw[i]
            mutation_one_new =np.zeros((mutation_one.shape[0]))
            index=random.randint(0,popw.shape[1])
            for n in range(mutation_one.shape[0]):
                if n!=index:
                    mutation_one_new[n]=mutation_one[n]
                elif index==3:
                    mutation_one_new[index]=random.uniform(-1000,0)
                elif index!=3:
                    mutation_one_new[index]=random.uniform(0,1000)
            popw_new_list.append(mutation_one_new)
    popw_new=np.array(popw_new_list)
    return popw_new

def popcon(popw_new1,popw_new2):
    popw_new=list(copy.copy(popw_new1))
    for i in range(len(popw_new)):
        for j in range(len(popw_new2)):
            if np.any(popw_new2[j]!=popw_new[i]):
                popw_new.append(popw_new2[j])
    popw=np.array(popw_new)
    return popw

def main(tz_number,dim,input_data,result_data,c_percent,m_percent,max_M):
    record_list=[]
    popw=creat_pop(tz_number,dim)
    mse_list=fun(input_data,result_data,popw)
    record_list.append(min(mse_list))
    print('M//max_M',0,'//',max_M)
    for M in range(max_M):
        print('M//max_M',f'{M+1}//{max_M}')
        pop_w_new1=crossover(popw,mse_list,c_percent)
        pop_w_new2=mutation(popw,m_percent)
        popw_new=popcon(pop_w_new1,pop_w_new2)
        mse_list_new=fun(input_data,result_data,popw_new)
        record_list.append(min(mse_list_new))
        mse=np.array(mse_list_new)
        arg=np.argsort(mse)
        popw=popw_new[arg][:dim+1]
    return popw,record_list

def plot_r(record_list):
    fig = plt.figure()
    x = [i + 1 for i in range(len(record_list))]
    plt.plot(x, record_list)
    plt.show()


def softmax(input_data):
    input_data_a = []
    for i in range(input_data.shape[0]):
        b = copy.copy(input_data[i])
        b = np.exp(b)
        sum_b = np.sum(b, axis=0)
        soft_b = b / sum_b
        input_data_a.append(soft_b)
    input_data_ = np.array(input_data_a)
    return input_data_

def norm(input_data):
    targ=1e-6
    input_data_a=[]
    for i in range(input_data.shape[0]):
        a=copy.copy(input_data[i])
        for j in range(a.shape[1]):
            if np.max(a[:,j])-np.min(a[:,j])!=0:
                a[:,j]=(a[:,j]-np.min(a[:,j]))/(np.max(a[:,j])-np.min(a[:,j]))
            else:
                a[:,j]=targ
        input_data_a.append(a)
    input_data_=np.array(input_data_a)
    return input_data_




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

input_data=norm(input_data)
input_data=softmax(input_data)

'''w,record_list=main(5,100,input_data,result_data,0.5,0.5,10)

mse=fun(input_data,result_data,w)
index=mse.index(min(mse))
print(w[index],mse[index])
test_data=norm(test_data)
test_data=softmax(test_data)
plot_r(record_list)
print(np.dot(test_data,np.array(w[index])))'''



