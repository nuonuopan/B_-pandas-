# C13 广义线性模型
# 这里讲的都是因变量是离散型变量的模型---因变量-->离散型
#13.2 逻辑回归
#因变量是二值的
#解释：
import pandas as pd
acs=pd.read_csv(r'F:\Learn\B_活用pandas库\data\pandas_for_everyone-master\data\acs_ny.csv')
acs.count()
acs.info()
for i in acs.columns:
    print(i,':')
    print(acs[i].unique())
    
#每个变量的可能取值的呈现
a=dict()
for i in acs.columns:
    a[i]=list(acs[i].unique())

#把连续变量(数值变量)变成二值响应变量的方法  pd.cut
acs['ge150k']=pd.cut(acs['FamilyIncome'],[0,150000,acs['FamilyIncome'].max()],labels=[0,1])

acs.info() #acs['ge150k']的dtype是category

#更改成数值型
acs['ge150k_i']=acs['ge150k'].astype(int)
acs.info()

acs.ge150k.dtype
acs.ge150k_i.dtype

#*******加餐********
#种类的计数
acs.ge150k.value_counts() 
#每个变量取值情况
acs.ge150k.unique()

#13.2.1 statsmodels
#logit函数
import statsmodels.formula.api as smf

acs.columns

model=smf.logit('ge150k_i~HouseCosts+NumWorkers+OwnRent+NumBedrooms+FamilyType',data=acs)
result=model.fit()
result.summary()


#所有的广义线性模型都需要使用连接函数执行一定的转换才能进行解释
# eg：logit函数,需要把结果指数化
import numpy as np
odd_ratio=np.exp(result.params)
odd_ratio
#解释：①数值变量：HouseCosts每增加一个单位，家庭收入FamilyIncome超过1500000的可能性增加1.007倍；②分类变量：对于房屋所有权状态，相对于按揭买房的家庭，那些拥有房屋完全产权的家庭的家庭收入FamilyIncome高于1500000的概率增加了1.82倍
#？？？？？分类变量不用e直接取系数，数字变量用e取系数？？

# acs.OwnRent.unique()
# array(['Mortgage', 'Rented', 'Outright'], dtype=object)



#13.2.2 sklearn
predictors=pd.get_dummies(acs[['HouseCosts','NumWorkers','OwnRent','NumBedrooms','FamilyType']],drop_first=True)
from sklearn import linear_model
lr=linear_model.LogisticRegression()
result=lr.fit(X=predictors, y=acs['ge150k_i'])
#这里用ge150k不会报错
result.coef_
result.intercept_
values=np.append(result.intercept_,result.coef_)
names=np.append('intercept',predictors.columns)
result_c=pd.DataFrame(values,index=names,columns=['coef'])
result_c
result_c['or']=np.exp(result_c['coef'])
result_c

#13.3 泊松回归
# 不是单纯的二值，而是计数数据 例如NumChildren变量
#13.3.1 statsmodels
#*******smf.poisson********   弄不出
import statsmodels.formula.api as smf
acs.NumChildren.dtype
model=smf.poisson('NumChildren~FamilyIncome+FamilyType+OwnRent',data=acs)
result=model.fit()
result.summary()

#*******smf.gl--Poissonm********   弄不出
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

model=smf.glm('NumChildren~FamilyIncome+FamilyType+OwnRent',data=acs,family=sm.families.Poisson(sm.genmod.families.links.log))
results=model.fit()
results.summary()

#*******smf.glm--NegativeBinomial********   弄不出
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

model=smf.glm('NumChildren~FamilyIncome+FamilyType+OwnRent',data=acs,family=sm.families.NegativeBinomial(sm.genmod.families.links.log))
results=model.fit()
results.summary()

# 13.5 生存分析
# 其目的是回答在不确定的情况下，为什么现在和以后发生事件（其中，事件可能指死亡、疾病缓解等）。这对于那些对测量生命周期感兴趣的研究人员来说是很好的：他们可以回答诸如什么因素可能影响死亡这样的问题？
#适用情况：对某件事情发生的时间进行建模
# ①判断某种疗法在预防严重不良时间方面是否优于其他标准疗法②当数据发生删失，即某件事的确切结果未知（某个治疗方案的患者有时会在跟进过程中失联）
bladder=pd.read_csv(r'F:\Learn\B_活用pandas库\data\pandas_for_everyone-master\data\bladder.csv')
bladder.info()
bladder.head()
#rx--不同疗法 stop--事件发生的时间 event--事件是否发生
bladder.rx.value_counts() #不同疗法的统计次数
bladder.event.value_counts()

type(bladder.event.value_counts())
bladder.stop.value_counts() 
bladder.stop.value_counts().sort_index()
b=bladder.stop.value_counts().sort_index()
c=b.sum()-b.cumsum()
c
c.plot()
bladder.dtypes

#*******生存分析********
#生存分析
#这里的变量都是数值型
from lifelines import KaplanMeierFitter
kmf=KaplanMeierFitter()
kmf.fit(durations=bladder.stop,event_observed=bladder.event)

#绘制生存曲线
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
ax=kmf.survival_function_.plot(ax=ax)
ax.set_title('Survival function')
plt.show()

##显示生存曲线+其置信区间
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
ax=kmf.plot(ax=ax)
ax.set_title('Survival with cnfidence interbals')
plt.show()



#*******这是一个模型来预测存活率********
#拟合出一个模型来预测存活率，称为Cox比例风险模型
from lifelines import CoxPHFitter
cph=CoxPHFitter()
#传入自变量列
cph_bladder_df=bladder[['rx','number','size','enum','stop','event']]
cph.fit(cph_bladder_df,duration_col='stop',event_col='event')
cph.print_summary()



#*******这是一个检验是否要对某一变量进行分类，来分别建立模型预测存活率********
#*******检验部分********
#检验Cox模型假设
#*******检验方法********
#一种检验方法是根据分层单独绘制生存曲线；即为每一种疗法rx绘制一条曲线。如果log(-log(生存曲线))与log(时间)曲线相交，则表示模型需要按变量进行分层
#*******检验原理（待查）********
rx1=bladder[bladder['rx']==1]
rx2=bladder[bladder['rx']==2]
from lifelines import KaplanMeierFitter

kmf1=KaplanMeierFitter()
kmf1.fit(durations=rx1.stop,event_observed=rx1.event)

kmf2=KaplanMeierFitter()
kmf2.fit(durations=rx2.stop,event_observed=rx2.event)

import matplotlib.pyplot as plt
fig,axes=plt.subplots()


kmf1.plot_loglogs(ax=axes)
kmf2.plot_loglogs(ax=axes)
axes.legend(['rx1','rx2'])
plt.show()


#因为两条线相互交叉，因此对分析进行分层是有意义的
#*******这里基于分层建立模型，预测存活率，strata--进行分层********
from lifelines import CoxPHFitter
cph_start=CoxPHFitter()
#传入自变量列
cph_bladder_df=bladder[['rx','number','size','enum','stop','event']]
cph_start.fit(cph_bladder_df,duration_col='stop',event_col='event',strata=['rx'])
cph_start.print_summary()


#*******？分层有意义，那是否建立不同rx的生存曲线？********
