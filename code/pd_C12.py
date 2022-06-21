# C12 线性模型


# 12.2 单变量线性回归
# 12.2.1 使用统计模型库statsmodels
import pandas as pd
import seaborn as sns

tips = pd.read_csv(r"F:\Learn\B_活用pandas库\data\seaborn-data\tips.csv")
tips.head()
tips.dtypes  # 查看变量类型
import statsmodels.formula.api as smf

model = smf.ols(formula="tip~total_bill", data=tips)  # 指定模型
result = model.fit()  # 拟合数据
result.summary()  # 结果
result.params  # 参数
result.conf_int()  # 置信区间，值在 [0.025,0.975]直接
# 12.2.2 使用sklearn库
from sklearn import linear_model

lr = linear_model.LinearRegression()  # 创建线性回归对象
predicted = lr.fit(X=tips["total_bill"].values.reshape(-1, 1), y=tips["tip"])  # 数据拟合
# type(tips['total_bill'].values)
# tips['total_bill'].values.reshape(-1,1).shape
# tips['total_bill'].values.shape

predicted.coef_
predicted.intercept_

#%%
# 12.3 多元回归
# 多个数值自变量：①多个数值自变量②数值+分类自变量
# 12.3.1 使用统计模型库statsmodels
model = smf.ols(formula="tip~total_bill+size", data=tips)  # 指定模型
result = model.fit()  # 拟合数据
result.summary()  # 结果
# 每个参数的解释都是在“在所以其他变量保持不变的情况下，xx每增加一个单位，tip增加xx”

# 多个自变量，可以包含分类变量
tips = pd.read_csv(r"F:\Learn\B_活用pandas库\data\seaborn-data\tips.csv")
tips.info()
# sex\smoker\day\time都可以转换成category
# 转换为category类型
tips.sex = tips.sex.astype("category")
tips.smoker = tips.smoker.astype("category")
tips.day = tips.day.astype("category")
tips.time = tips.time.astype("category")
tips.info()
tips.sex.unique()  # 查看变量可能取值
tips.day.unique()  # 查看变量可能取值
# 对分类变量建模时，必须创建虚拟变量，即分类中的每个唯一值都变成了新的二元特征；eg：Female or Male
# statsmodels会自动创建虚拟变量,并删除参考变量来避免多重共线性
# 为了避免多重共线性，通常会删除其中一个虚拟变量，比如Female和Male，可以知道不是男性就是女性，在这种情况下可以删除代表男性的虚拟变量

model = smf.ols(formula="tip~total_bill+size+sex+smoker+day+time", data=tips)  # 指定模型
result = model.fit()  # 拟合数据
result.summary()  # 结果
# 结果显示 sex[T.Male],代表删除的参考变量是Female，解释为：当sex从Female变成Male时，tip减少了0.0342

#%%
# 12.3.3 使用sklearn
# 多元线性回归不用重塑X值了
lr = linear_model.LinearRegression()  # 创建线性回归对象
predicted = lr.fit(X=tips[["total_bill", "size"]], y=tips["tip"])
predicted.coef_
predicted.intercept_
# 多个自变量，可以包含分类变量
# 必须为sklearn手动创建虚拟变量
# pandas的get_dummies可以实现，自动把所以分类变量转换为虚拟变量
tips_dummy = pd.get_dummies(tips)
tips_dummy_drop = pd.get_dummies(tips, drop_first=True)  # 删除参考变量
del tips_dummy_drop["tip"]  # 不要把因变量也带进来
predicted = lr.fit(X=tips_dummy_drop, y=tips["tip"])
predicted.coef_
predicted.intercept_
# 12.4 保留sklearn索引
# sklearn模型的系数不带标签，可以收到存储标签，并添加系数
import numpy as np

lr = linear_model.LinearRegression()
predicted = lr.fit(X=tips_dummy_drop, y=tips["tip"])
# 获取截距及其他系数
values = np.append(predicted.intercept_, predicted.coef_)
names = np.append("intercept", tips_dummy_drop.columns)
a = pd.Series(values, index=names)
print(a)
b = pd.DataFrame(values, index=names, columns=["coef"])
b

