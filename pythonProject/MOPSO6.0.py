# 加油
# 时间：2023/11/15 19:01
import copy
import random
import pandas as pd
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def creat_pop(fill_limit,exploit_limit,dim,years,rain):
    result=[]
    for m in range(dim):
        pop=[]
        for n in range(years):
            pop_ = [rain[n]]
            for i in range(len(fill_limit)):
                pop_.append(np.random.uniform(fill_limit[i][0], fill_limit[i][1]))
            for j in range(len(exploit_limit)):
                pop_.append(np.random.uniform(exploit_limit[j][0], exploit_limit[j][1]))
            pop.append([pop_])
        result.append(pop)
    return np.array(result)

def creat_v(fill_limit,exploit_limit,dim,years):
    result = []
    for m in range(dim):
        v = []
        for n in range(years):
            v_ = [0]
            for i in range(len(fill_limit)):
                v_.append(np.random.uniform(-fill_limit[i][1]/10, fill_limit[i][1]/10))
            for j in range(len(exploit_limit)):
                v_.append(np.random.uniform(-exploit_limit[j][1]/10, exploit_limit[j][1]/10))
            v.append([v_])
        result.append(v)
    return np.array(result)

def updata_pop(fill_limit,exploit_limit,pop,v):
    pop_new=copy.copy(pop)
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            for n in range(pop.shape[2]):
                for m in range(pop.shape[3]):
                    if 0<m<=len(fill_limit):
                        pop_new[i,j,n,m]=pop_new[i,j,n,m]+v[i,j,n,m]
                        if pop_new[i,j,n,m]<fill_limit[m-1][0]:
                            pop_new[i, j, n, m] = fill_limit[m - 1][0]
                        elif pop_new[i,j,n,m]>fill_limit[m-1][1]:
                            pop_new[i, j, n, m] = fill_limit[m - 1][1]
                    elif len(fill_limit)<m<=len(fill_limit)+len(exploit_limit):
                        pop_new[i, j, n, m] = pop_new[i, j, n, m] + v[i, j, n, m]
                        if pop_new[i, j, n, m] < exploit_limit[m - len(fill_limit)-1][0]:
                            pop_new[i, j, n, m] =  exploit_limit[m - len(fill_limit)-1][0]
                        elif pop_new[i, j, n, m] >  exploit_limit[m - len(fill_limit)-1][1]:
                            pop_new[i, j, n, m] =  exploit_limit[m - len(fill_limit)-1][1]
    return pop_new

def func(tar_level,pop,model_name='test1.keras'):
    model = keras.models.load_model(model_name)
    pre=[]
    pcr=[]
    for i in range(pop.shape[0]):
        y=model.predict(pop[i])
        pre.append(np.sum(y,axis=1))
        pcr.append(y)
    pre_sum=np.sum(np.array(pre),axis=1)
    result=abs(tar_level-pre_sum)
    return list(result),pcr

def updata_v(fill_limit,exploit_limit,v,pbest,gbest,pop,chi):
    v_new=copy.copy(v)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            for n in range(v.shape[2]):
                for m in range(v.shape[3]):
                    w = 1.2
                    r1, r2 = np.random.random(), np.random.random()
                    c1, c2 = 2,2
                    if 0 < m <= len(fill_limit):
                        v_new[i, j, n, m] =(w*v_new[i, j, n, m] + r1*c1*(
                                pop[i,j,n,m]-pbest[i,j,n,m])+r2*c2*(pop[i,j,n,m]-gbest[j,n,m]))*chi
                        if v_new[i, j, n, m] < -fill_limit[m - 1][1]/10:
                            v_new[i, j, n, m] = -fill_limit[m - 1][1]/10
                        elif v_new[i, j, n, m] > fill_limit[m - 1][1]/10:
                            v_new[i, j, n, m] = fill_limit[m - 1][1]/10
                    elif len(fill_limit) < m <= len(fill_limit) + len(exploit_limit):
                        v_new[i, j, n, m] = (w * v_new[i, j, n, m] + r1 * c1 * (
                                pop[i, j, n, m] - pbest[i, j, n, m]) + r2 * c2 * (pop[i, j, n, m] - gbest[j, n, m]))*chi
                        if v_new[i, j, n, m] < -exploit_limit[m - len(fill_limit) - 1][1]/10:
                            v_new[i, j, n, m] = -exploit_limit[m - len(fill_limit) - 1][1]/10
                        elif v_new[i, j, n, m] > exploit_limit[m - len(fill_limit) - 1][1]/10:
                            v_new[i, j, n, m] = exploit_limit[m - len(fill_limit) - 1][1]/10
    return v_new

def main(fill_limit,exploit_limit,dim,years,rain,tar_level,max_G,model_name='test1.keras'):
    record=[]
    record_pop=[]
    tar=[]
    pop = creat_pop(fill_limit,exploit_limit,dim,years,rain)
    v = creat_v(fill_limit,exploit_limit,dim,years)
    fitness,pcr=func(tar_level,pop,model_name)
    min_dis=min(fitness)
    record.append(min_dis)
    index=fitness.index(min_dis)
    pbest=pop
    gbest=pop[index]
    for j in range(len(fitness)):
        if fitness[j]==min_dis:
            record_pop.append(pop[j])
            tar.append(fitness[j])
    for G in range(max_G):
        print(f'mopso:{G+1}//max_mopso:{max_G},loss:{min_dis}')
        pop=updata_pop(fill_limit,exploit_limit,pop,v)
        if len(record)>2:
            if abs(record[-1]-record[-2])<tar_level*0.01:
                k=random.random()
                if k>0.5:
                    chi=100*random.random()
                else:
                    chi=-100*random.random()
            else:
                chi=1
        else:
            chi=1
        v=updata_v(fill_limit,exploit_limit,v,pbest,gbest,pop,chi)
        fitness1,pcr=func(tar_level,pop,model_name)
        min_dis1=min(fitness1)
        index1=fitness1.index(min_dis1)
        if min_dis1<min_dis:
            min_dis=min_dis1
            gbest=pop[index1]
            record.append(min_dis1)
        else:
            record.append(min_dis)
        for i in range(pbest.shape[0]):
            if fitness1[i]<fitness[i]:
                pbest[i]=pop[i]
                fitness[i]=fitness1[i]
            elif fitness1[i]==min_dis1:
                record_pop.append(pop[i])
                tar.append(fitness1[i])
    record_f=[]
    min_value=min(tar)
    for n in range(len(tar)):
        if tar[n]==min_value:
            record_f.append(record_pop[n])
    return record_f,record

#补水水量区间
fill_limit=[[0,6500],[0,2500],[0,600]]
#开采水量区间
exploit_limit=[[5588.042,10150.542]]
#目标变化水位，可以一年也可以多年
tar_level=4
#种群个数
dim=10
#降水，每年写一个
rain=[7454.702179536]
#年份
years=1
#最大迭代次数
max_G=65
#模型文件名称
model_name='test1.keras.h5'

#主函数
pop,record=main(fill_limit,exploit_limit,dim,years,rain,tar_level,max_G,model_name=model_name)
fit,lev=func(tar_level,np.array(pop),model_name=model_name)

#输出数据
print('==============================================================================================')
print('正在输出数据…………')
path=r'输出.xlsx'
for j in range(len(pop)):
    mid=[]
    mid1=[]
    for n in range(len(pop[0])):
        mid.append(pop[j][n][0])
    df=pd.DataFrame(np.array(mid))
    with pd.ExcelWriter(path, mode='a', engine="openpyxl",if_sheet_exists='new') as writer:
        df.to_excel(writer, sheet_name=f'pop{j+1}', index=False)

for p in range(len(lev)):
    df1 = pd.DataFrame(lev[p])
    with pd.ExcelWriter(path, mode='a', engine="openpyxl",if_sheet_exists='new') as writer:
        df1.to_excel(writer, sheet_name=f'level', index=False)
print('==============================================================================================')
print('数据输出完成!')