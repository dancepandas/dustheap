# 加油
# 时间：2023/11/5 16:15
import copy
import www
import numpy as np
from  tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.rcParams['font.sans-serif'] = ['SimSun']# 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def creat_pop(exploit_limit,fill_limit,rain_water,index_list,years,dim):
    POP_=[]
    targ=np.array([exploit_limit[0][1],exploit_limit[1][1],exploit_limit[2][1],(fill_limit[1]/100),np.sum(rain_water)+10])
    targ1= np.array([exploit_limit[0][0], exploit_limit[1][0], exploit_limit[2][0], (fill_limit[0] / 100), 0])
    for j in range(dim):
        Pop_=[]
        for i in range(years):
            pop=np.zeros((len(index_list[0]),5),dtype=float)
            sum_pop=np.sum(pop,axis=0)
            while np.any(sum_pop>targ) or np.all(sum_pop==0) or np.any(sum_pop<targ1):
                for index in range(len(index_list[i])):
                    pop[index_list[i][index],0]  = round(np.random.uniform(exploit_limit[0][0],exploit_limit[0][1]/8),5)
                    pop[index_list[i][index], 1] = round(np.random.uniform(exploit_limit[1][0],exploit_limit[1][1]/8),5)
                    pop[index_list[i][index], 2] = round(np.random.uniform(exploit_limit[2][0],exploit_limit[2][1]/8),5)
                    pop[index_list[i][index], 3] = round(np.random.uniform(fill_limit[0], fill_limit[1]) / 1000, 5)
                    pop[index_list[i][index], 4] = rain_water[i][index]
                    sum_pop=np.sum(pop,axis=0)
            Pop_.append(pop)
        POP_.append(Pop_)
    POP=np.array(POP_)
    return POP

def creat_velocity(exploit_limit,fill_limit,rain_water,index_list,years,dim):
    V_=[]
    for j in range(dim):
        Velocity_=[]
        for i in range(years):
            velocity=np.zeros((12,5),dtype=float)
            for index in range(len(index_list[i])):
                velocity[index_list[i][index],0]  = round(np.random.uniform(-exploit_limit[0][1]/1000,exploit_limit[0][1])/1000,2)
                velocity[index_list[i][index], 1] = round(np.random.uniform(-exploit_limit[1][1]/1000,exploit_limit[1][1])/1000,2)
                velocity[index_list[i][index], 2] = round(np.random.uniform(-exploit_limit[2][1]/1000,exploit_limit[2][1])/1000,2)
                velocity[index_list[i][index], 3] = round(np.random.uniform(-fill_limit[1]/1000,fill_limit[1])/1000,2)
                velocity[index_list[i][index], 4] = rain_water[i][index] * 0
            Velocity_.append(velocity)
        V_.append(Velocity_)
    V=np.array(V_)
    return V

def fun(pop,target_water_level,w,or_level):
    fg=2
    prediction_=[]
    for i in range(pop.shape[0]):
        pop_value=copy.copy(pop[i])
        pop_value=www.norm(pop_value)
        pop_value=www.softmax(pop_value)
        pop_value=np.sqrt(pop_value)
        list_=[]
        list_dis=[]
        for pp in range(pop_value.shape[0]):
            level=np.dot(pop_value[pp],np.array(w))
            a=copy.copy(level[:-1])
            b=copy.copy(level[1:])
            c=0.5*(a+b)
            levels=np.concatenate((level[0:1],c),axis=0)
            level_ture=[0 for i in range(levels.shape[0])]
            for j in range(len(level_ture)):
                if pp==0:
                    bu=or_level
                    level_ture[j]=bu+levels[j]-1.5
                else:
                    bu=list_[pp-1][-1]
                    level_ture[j] = bu + levels[j] - 1.5
                dis_=abs(fg-(level_ture[-1]-level_ture[0]))
            list_.append(level_ture)
            list_dis.append(dis_)
        prediction_.append(list_dis)
    prediction=np.array(prediction_)
    out=[item[0] for item in prediction]
    return out

def updata_velocity(exploit_limit,fill_limit,index_list,velocity,pop,pbest,gbest,fitness,M,Mopso_max):
    w = 1
    c1, c2 = 2, 2
    r1, r2 = np.random.random(), np.random.random()
    velocity_new = copy.copy(velocity)
    velocity_new = w * velocity + r1 * c1 * (pbest - pop) + r2 * c2 * (gbest - pop)
    for i in range(velocity_new.shape[-1]):
        a=velocity_new[:,:,:,i]
        if i <3:
            a[a>exploit_limit[i][1]/100]=exploit_limit[i][1]/100
            a[a<-exploit_limit[i][1]/100]=-exploit_limit[i][1]/100
        elif i==3:
            a[a>fill_limit[1]/100]=fill_limit[1]/100
            a[a < -fill_limit[1] / 100] = -fill_limit[1] / 100
    return velocity_new

def updata_pop(exploit_limit,fill_limit,velocity,pop,targ):
    a=copy.copy(pop)
    pop_new=pop+velocity
    pop_new[pop_new < 0] = 0
    for i in range(pop_new.shape[0]):
        for j in range(pop_new.shape[1]):
            pop_value=pop_new[i,j]
            sum_pop=np.sum(pop_value,axis=0)
            if np.any(sum_pop>targ):
                ch=np.zeros((sum_pop.shape[0]))
                dd=np.where(sum_pop>targ)
                ch[dd]=(sum_pop[dd]-targ[dd])/12
                pop_new[i,j]=pop_new[i,j]-ch
            elif np.any(sum_pop<targ1):
                chi = np.zeros((sum_pop.shape[0]))
                ddi = np.where(sum_pop < targ1)
                chi[ddi] = (sum_pop[ddi] - targ1[ddi]) / 12
                pop_new[i, j] = pop_new[i, j] - chi
            pop_new[pop_new < 0] = 0
    return pop_new

def fun2(pop,or_level):
    prediction_=[]
    for i in range(pop.shape[0]):
        pop_value=copy.copy(pop[i])
        pop_value=www.norm(pop_value)
        pop_value=www.softmax(pop_value)
        pop_value=np.sqrt(pop_value)
        list_ = []
        for pp in range(pop_value.shape[0]):
            level = np.dot(pop_value[pp], np.array(w))
            a = copy.copy(level[:-1])
            b = copy.copy(level[1:])
            c = 0.5 * (a + b)
            levels = np.concatenate((level[0:1], c), axis=0)
            level_ture = [0 for i in range(levels.shape[0])]
            for j in range(len(level_ture)):
                for j in range(len(level_ture)):
                    if pp == 0:
                        bu = or_level
                        level_ture[j] = bu + levels[j] - 1.5
                    else:
                        bu = list_[pp - 1][-1]
                        level_ture[j] = bu + levels[j] - 1.5
            list_.append(level_ture)
        prediction_.append(list_)
    prediction = np.array(prediction_)
    return prediction


path=r'输入.xlsx'
df=pd.read_excel(path,sheet_name='Sheet1')
#年份
years=int(df[df.columns[2]][0])
#计算月份索引
index_list_=list(df[df.columns[0]]-1)
#每月降雨量
rain_=list(df[df.columns[1]])
rain=[rain_ for g in range(years)]
index_list=[index_list_ for m in range(years)]


#补水量区间
exploit_limit=[[0,6500],[0,2500],[0,600]]
#开采量区间
fill_limit=[8037.5,13204.05]

#约束水位
target_water_level=55

#Mopso_max
Mopso_max=int(df[df.columns[6]][0])

#dim
dim=int(df[df.columns[7]][0])

#length_max
length_max=int(df[df.columns[8]][0])

w=[0.20897633,0.09258965,0.37861799,-3.77826351,9.4562935]
or_level=35

targ=np.array([exploit_limit[0][1],exploit_limit[1][1],exploit_limit[2][1],(fill_limit[1]/100),sum(rain[0])+1])
targ1= np.array([exploit_limit[0][0], exploit_limit[1][0], exploit_limit[2][0], (fill_limit[0] / 100), 0])

def main(exploit_limit, fill_limit, rain, index_list, years, dim,target_water_level,w,targ):
    record_list=[]
    record_pop=[]
    record_pop1 = []
    pop = creat_pop(exploit_limit, fill_limit, rain, index_list, years, dim)
    velocity = creat_velocity(exploit_limit, fill_limit, rain, index_list, years, dim)
    fitness = fun(pop, target_water_level, w,or_level)
    pbest=pop
    minfit=min(fitness)
    index=fitness.index(minfit)
    gbest=pop[index]
    for i in range(len(fitness)):
        if fitness[i]==minfit:
            for jj in range(pop.shape[1]):
                sum_pop=np.sum(pop[i][jj],axis=0)
                record_pop.append(sum_pop)
                record_pop1.append(pop[i])
    for M in range(Mopso_max):
        pop=updata_pop(exploit_limit,fill_limit,velocity,pop,targ)
        velocity=updata_velocity(exploit_limit,fill_limit,index_list,velocity,pop,pbest,gbest,fitness,M,Mopso_max)
        fitness1=fun(pop,target_water_level,w,or_level)
        minfit1=min(fitness1)
        index1=fitness1.index(minfit1)
        for j in range(len(fitness1)):
            if fitness1[j]<fitness[j]:
                fitness[j]=fitness1[j]
                pbest[j]=pop[j]
        if minfit1<minfit:
            gbest=pop[index1]
            record_list.append(minfit1)
            for p in range(len(fitness1)):
                if fitness1[p]==minfit1:
                    for jj in range(pop.shape[1]):
                        sum_pop = np.sum(pop[p][jj], axis=0)
                        record_pop.append(sum_pop)
                        record_pop1.append(pop[p])
    return record_list,record_pop,record_pop1
record_list,record_pop,record_pop1=main(exploit_limit, fill_limit, rain, index_list, years, dim, target_water_level, w, targ)
level=fun2(np.array(record_pop1),or_level)
level_=[]
for ij in range(level.shape[0]):
    level_.append(level[ij][0])
level_ar=np.array(level_)
print('==============================================================================================')
print('正在输出数据…………')
print(f'共产生优化方案{len(record_pop1)}组…………')
path1=r'输出.xlsx'
df1=pd.DataFrame(level_ar)
with pd.ExcelWriter(path1, mode='a', if_sheet_exists='overlay') as writer:
    df1.to_excel(writer, sheet_name=f'level', index=False)

df3=pd.DataFrame(np.array(record_pop))
with pd.ExcelWriter(path1, mode='a', if_sheet_exists='overlay') as writer:
    df3.to_excel(writer, sheet_name='sum', index=False)

for mm in range(len(record_pop1)):
    for tt in range(len(record_pop1[0])):
        df2=pd.DataFrame(np.array(record_pop1[mm][tt]))
        sheet_name=f'pop{mm+1}_{tt+1}'
        with pd.ExcelWriter(path1, mode='a', if_sheet_exists='overlay') as writer:
            df2.to_excel(writer, sheet_name=sheet_name, index=False)
print('==============================================================================================')
print('数据输出完成!')

