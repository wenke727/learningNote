#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
import datetime as dt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

base_folder = 'E:/Data/AI/Aliyun_Learning/03O2O优惠券/data'

# %%
# 读取数据
off_train = pd.read_csv( os.path.join(base_folder, 'ccf_offline_stage1_train.csv'), keep_default_na=True )
off_test  = pd.read_csv( os.path.join(base_folder, 'ccf_offline_stage1_test_revised.csv'), keep_default_na=True )
on_train  = pd.read_csv( os.path.join(base_folder, 'ccf_online_stage1_train.csv'), keep_default_na=True )

off_train.columns = [
    'user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
    'date_received', 'date'
]
off_test.columns = [
    'user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
    'date_received'
]
on_train.columns = [
    'user_id', 'merchant_id', 'action', 'coupon_id', 'discount_rate',
    'date_received', 'date'
]

for df in [off_test, off_train, on_train]:
    df[['user_id', 'merchant_id','coupon_id']] = df[['user_id', 'merchant_id','coupon_id']].astype(str)


# %%
off_train.isnull().sum()

# %%
# 通过探索可以发现训练数据的用券数据是到6月30日，而领券日期并不是到6月30日，而是到6月15日，这在设计滑窗结构的时候需要注意。
def get_min_max(df, att='date_received'):
    return int(df[ df[att]!='null' ][att].min()), int(df[ df[att]!='null' ][att].max())

print(f"offline train date_received: \n\t{get_min_max(off_train)} ")
print(f"online train date_received: \n\t{get_min_max(on_train)} ")
print(f"test date_received: \n\t{get_min_max(off_test)} \n")

print(f"offline train date: \n\t{get_min_max(off_train, 'date')} ")
print(f"online train date: \n\t{get_min_max(on_train, 'date')} ")

# %%
# 对用户，商家，优惠券在训练集和测试集的重合情况进行探索发现： 测试集的用户ID与Offline训练集重复占比0.999以上，与Online训练集重复占比0.565。 测试集的商家ID与Offline训练集重复占比0.999以上，与Online训练集没有重复。 测试集的优惠券ID与训练集都没有重复。 结论：Online数据价值比较低，后续特征提取将以Offline训练集为主。在提取优惠券统计特征的时候不能通过ID进行合并。 在后续可视化分析中将主要在Offline训练集及测试集之间进行。

def check_att(att = 'user_id'):
    off_train_user = off_train[[att]].copy().drop_duplicates()
    off_test_user  = off_test[[att]].copy().drop_duplicates()
    on_train_user  = on_train[[att]].copy().drop_duplicates()
    print(f'{att} offline, online, offline 训练集数量\n\t: {off_train_user[att].count()}, {on_train_user[att].count()}, {off_test_user[att].count()}')

    off_train_user['off_train_flag'] = 1
    off_merge = off_test_user.merge(off_train_user, on=att, how='left').reset_index().fillna(0)

    print(f"offline 训练集与测试集{att}:\n\t重复数量: {off_merge['off_train_flag'].sum()}, 占比: {off_merge['off_train_flag'].sum() / off_merge['off_train_flag'].count() *100 :.3f}%")

    on_train_user['on_train_flag'] = 1
    on_merge = off_test_user.merge(on_train_user, on=att, how='left').reset_index().fillna(0)

    print(f"online 训练集与测试集{att}:\n\t重复数量: {on_merge['on_train_flag'].sum()}, 占比: {on_merge['on_train_flag'].sum() / on_merge['on_train_flag'].count() *100 :.3f}%")

    return

check_att('user_id')
check_att('merchant_id')
check_att('coupon_id')


# %%
off_train.discount_rate.value_counts()
off_test.discount_rate.value_counts()

# %%
# 通过初步观察感觉训练集和测试集数据分布比较一致。
plt.rcParams['figure.figsize'] = (25, 4)
plt.plot( off_train.discount_rate.value_counts() )

# %%
# 将特征数值化
separator = ':'

def get_discount_rate(s):
    s = str(s)
    if s =='null':
        return -1
    s = s.split(separator)
    if len(s) == 1:
        return float(s[0])
    else:
        return 1.0 - float(s[1]) / float(s[0])

def get_if_fd(s):
    s = str(s)
    s = s.split(separator)
    return 0 if len(s) == 1 else 1

def get_full_value(s):
    s = str(s)
    s = s.split(separator)
    return -1 if len(s) == 1 else int(s[0])

def get_reduction_value(s):
    s = str(s)
    s = s.split(separator)
    return -1 if len(s) == 1 else int(s[1])

def get_month(s):
    return -1 if s[0] == 'null' else int(s[4:6])

def get_day(s):
    return -1 if s[0] == 'null' else int(s[6:8])

def get_day_gap(s):
    s = s.split(separator)
    if s[0] == 'null' or s[1] == 'null' or s[0] == 'nan' or s[1] == 'nan':
        return -1
    
    return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) -\
            date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days

def get_label(s):
    s = s.split(separator)
    if s[0] == 'null' or s[1] == 'null' or s[0] == 'nan' or s[1] == 'nan':
        return -1
    days = (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - \
            date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days

    return 1 if days <= 15 else -1

def add_feature(df):
    df['if_fd']           = df['discount_rate'].apply(get_if_fd)
    df['full_value']      = df['discount_rate'].apply(get_full_value)
    df['reduction_value'] = df['discount_rate'].apply(get_reduction_value)
    df['discount_rate']   = df['discount_rate'].apply(get_discount_rate)
    # df['distance']        = df['distance'].astype(str).replace('nan', -1).astype(float).astype(int)
    df['distance']        = df['distance'].fillna(-1).astype(np.int)
    #df['month_received'] = df['date_received'].apply(get_month)
    #df['month'] = df['date'].apply(get_month)
    return df


def add_label(df):
    df['day_gap'] = df['date'].astype('str') + ':' + df['date_received'].astype('str')
    df['label']   = df['day_gap'].apply(get_label)
    df['day_gap'] = df['day_gap'].apply(get_day_gap)

    return df


#拷贝数据，免得调试的时候重读文件
dftrain = off_train.copy()
dftest = off_test.copy()

dftrain = add_feature(dftrain)
dftrain = add_label(dftrain)
dftest = add_feature(dftest)

# %%
fig = plt.figure(figsize=(4,6))
sns.boxplot(dftrain.query( "label >= 0 and distance >= 0" ).distance, orient='v', width=.5)

# %%
fig = plt.figure(figsize=(4,6))
sns.boxplot(dftrain.query( "label >= 0 and discount_rate >= 0" ).discount_rate, orient='v', width=.5)

# %%
def plot_his_qq(df, att):
    plt.figure(figsize=(10,5))
    data = df.query(f"label >= 0 and {att}>=0")[att]
    
    ax = plt.subplot(1,2,1)
    sns.distplot(data, fit=stats.norm)

    ax = plt.subplot(1,2,2)
    res = stats.probplot(data, plot=plt)
    plt.close()
    
    return

plot_his_qq(dftrain, 'distance')
plot_his_qq(dftrain, 'discount_rate')

# %%
plt.rcParams['figure.figsize'] = (6.0, 4.0)  #设置图片大小
ax = sns.kdeplot(dftrain[(dftrain.label >= 0)
                         & (dftrain.discount_rate >= 0)]['discount_rate'],
                 color="Red",
                 shade=True)
ax = sns.kdeplot(dftest[(dftest.discount_rate >= 0)]['discount_rate'],
                 color="Blue",
                 shade=True)
ax.set_xlabel('discount_rate')
ax.set_ylabel("Frequency")
ax = ax.legend(["train", "test"])
# %%
ax = sns.kdeplot(dftrain[(dftrain.label >= 0)
                         & (dftrain.distance >= 0)]['distance'],
                 color="Red",
                 shade=True)
ax = sns.kdeplot(dftest[(dftest.distance >= 0)]['distance'],
                 color="Blue",
                 shade=True)
ax.set_xlabel('distance')
ax.set_ylabel("Frequency")
ax = ax.legend(["train", "test"])
# %%
# 可视化线形关系
fcols = 2
frows = 1
plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 2, 1)
sns.regplot(x='distance',
            y='label',
            data=dftrain[(dftrain.label >= 0)
                         & (dftrain.distance >= 0)][['distance', 'label']],
            ax=ax,
            scatter_kws={
                'marker': '.',
                's': 3,
                'alpha': 0.3
            },
            line_kws={'color': 'k'})
plt.xlabel('distance')
plt.ylabel('label')
ax = plt.subplot(1, 2, 2)
sns.distplot(dftrain[(dftrain.label >= 0)
                     & (dftrain.distance >= 0)]['distance'].dropna())
plt.xlabel('distance')
plt.show()


# %%
fcols = 2
frows = 1
plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 2, 1)
sns.regplot(x='distance',
            y='label',
            data=dftrain[(dftrain.label >= 0)
                         & (dftrain.distance >= 0)][['distance', 'label']],
            ax=ax,
            scatter_kws={
                'marker': '.',
                's': 3,
                'alpha': 0.3
            },
            line_kws={'color': 'k'})
plt.xlabel('distance')
plt.ylabel('label')
ax = plt.subplot(1, 2, 2)
sns.distplot(dftrain[(dftrain.label >= 0)
                     & (dftrain.distance >= 0)]['distance'].dropna())
plt.xlabel('distance')
plt.show()

# %%
