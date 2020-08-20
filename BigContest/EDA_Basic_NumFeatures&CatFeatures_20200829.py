#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 모듈 불러오기
import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
pd.options.display.float_format = '{:.5f}'.format
pd.set_option('display.max_rows', 130)

from sklearn.preprocessing import PowerTransformer
from wordcloud import WordCloud
import string
from collections import Counter

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker

import platform
get_ipython().run_line_magic('matplotlib', 'inline')
if platform.system() == 'Darwin': # Mac 환경 폰트 설정
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': # Windows 환경 폰트 설정
    plt.rc('font', family='Malgun Gothic')

plt.rc('axes', unicode_minus=False) # 마이너스 폰트 설정
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[2]:


# 경로 설정
SALES_DATA = './data/방송시간데이터.xlsx'


# In[4]:


# 데이터 불러오기
data = pd.read_excel(SALES_DATA, parse_dates=True, usecols=lambda x: 'Unnamed' not in x)
data.head()


# # 컬럼 추가/변경
# 
# - ~~판매량~~: 추가하면서 무형인 애들 있다.
# - ~~월~~
# - 계절
# - ~~요일~~
# - ~~방송시간대~~
# 
# ---
# 고민중
# 
# - 계절: 필요할까? 월로 보면 될 것 같기도?
# - 공휴일 여부: 공휴일, +- 3까지?

# In[6]:


# 컬럼 추가
data['판매량'] = data['취급액'] / data['판매단가']
data['월'] = data['방송시작'].dt.month
data['요일'] = data['방송시작'].dt.dayofweek

day_mapping_dict = {0:'월요일', 1:'화요일', 2:'수요일', 3:'목요일', 4:'금요일', 5:'토요일', 6:'일요일'}
data['요일'] = data['요일'].map(day_mapping_dict)

def time_to_str(x):
    return x.strftime('%H')
data['방송시간대'] = data['방송시작'].dt.time.apply(lambda x: time_to_str(x))


# In[7]:


# 범주형 변수 변경
data['날짜'] = pd.Categorical(data['날짜'])
data['마더코드'] = pd.Categorical(data['마더코드'])
data['상품코드'] = pd.Categorical(data['상품코드'])
data['상품군'] = pd.Categorical(data['상품군'])
data['월'] = pd.Categorical(data['월'])
data['요일'] = pd.Categorical(data['요일'])
data['방송시간대'] = pd.Categorical(data['방송시간대'])
data.info()


# In[40]:


# 무형 상품 제외
data = data[data['상품군'] != '무형'].reset_index(drop=True)
data


# # 종속변수 확인
# 
# - 종속변수: 취급액

# In[12]:


# 전체적인 분포 확인
sns.distplot(data['취급액'])


# In[13]:


# 적절한 경계값 찾기
for i in range(5, 30):
    _, bins, patches = plt.hist(data['취급액'], bins=i)
    plt.show()
    print(f"{i}개의 범주일 때 경계값: {bins}\n")


# In[ ]:





# In[ ]:





# In[ ]:





# ## 수치형 변수 확인

# In[14]:


# 수치형 변수 확인
data.describe()


# In[16]:


# 수치형 변수 간 상관관계
display(data[['노출(분)', '판매단가', '취급액', '판매량']].corr())
sns.pairplot(data[['노출(분)', '판매단가', '취급액', '판매량']])
plt.show()


# 취급액 구간별로 상관관계 확인

# In[30]:


def map_sales(x):
    if x < 63591400.0:
        return "하"
    elif x < 127182800.0:
        return "중"
    else:
        return "상"

data['취급액_범주'] = data['취급액'].apply(lambda x: map_sales(x))
data['취급액_범주'] = pd.Categorical(data['취급액_범주'])
data.info()


# In[31]:


sns.pairplot(data[['노출(분)', '판매단가', '취급액', '판매량', '취급액_범주']], hue='취급액_범주')
plt.show()


# # 범주형 변수

# ### 한 변수 확인

# In[15]:


# 노출(분)
sns.distplot(data['노출(분)'])

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

_, bins, patches = ax1.hist(data['노출(분)'])
_, bins2, patches2 = ax2.hist(data[(data['노출(분)'] >= bins[2]) & (data['노출(분)'] < bins[3])]['노출(분)'], bins=20) # 몰린 구간 확대


# In[17]:


# 판매단가
sns.distplot(data['판매단가'])

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

_, bins, patches = ax1.hist(data['판매단가'])
_, bins2, patches2 = ax2.hist(data[(data['판매단가'] >= bins[3]) & (data['판매단가'] < bins[4])]['판매단가'], bins=20) # 몰린 구간 확대


# In[18]:


# 판매량
sns.distplot(data['판매량'])

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

_, bins, patches = ax1.hist(data['판매량'])
_, bins2, patches2 = ax2.hist(data[(data['판매량'] >= bins[0]) & (data['판매량'] < bins[1])]['판매량'], bins=20) # 몰린 구간 확대


# In[19]:


# 방송시간대 분포 확인
sns.countplot(x='방송시간대', data=data, )


# In[20]:


# 방송 요일 분포
sns.countplot(x='요일', data=data, order=['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'])


# In[21]:


# 방송 상품군 분포
sns.countplot(x='상품군', data=data)


# In[22]:


# 마더코드 건드려 보자. 응 아니야... 응? 뭔가 많은 게 있긴 한데.
sns.countplot(x='마더코드', data=data)


# ## 두 변수 간 관계 확인
# - 수치형 + 범주형
# - 범주형 + 범주형

# In[23]:


# 각 범주형 변수와 수치형 변수 간 sum, mean 확인
def grouping(df, colname, agg_funcs):
    print(f"집계 기준 범주: {colname}")
    display(data.groupby(by=colname, as_index=False).agg(agg_funcs).sort_values(by=('취급액', 'mean'), ascending=False))
    print()

for col in ['상품군', '마더코드', '방송시간대', '요일']:
    grouping(data, col, ['sum', 'mean'])


# ### 취급액과의 관계

# In[24]:


# 각 범주별 취급액 합계
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
fig.suptitle("각 범주별 취급액 합계", size=18)

sns.barplot(data=data, y="취급액", x="상품군", orient="v", ax=axes[0][0], estimator=np.sum)
sns.barplot(data=data, y="취급액", x="월", orient="v", ax=axes[0][1], estimator=np.sum)
sns.barplot(data=data, y="취급액", x="요일", orient="v", ax=axes[1][0], estimator=np.sum)
sns.barplot(data=data, y="취급액", x="방송시간대", orient="v", ax=axes[1][1], estimator=np.sum)

axes[0][0].set(xlabel='상품군', ylabel='취급액', title='상품군별 취급액')
axes[0][1].set(xlabel='월', ylabel='취급액', title='월별 취급액')
axes[1][0].set(xlabel='요일', ylabel='취급액', title='요일별 취급액')
axes[1][1].set(xlabel='방송시간대', ylabel='취급액', title='방송시간대별 취급액')
plt.show()

# 각 범주별 취급액
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
fig.suptitle("각 범주별 취급액 평균", size=18)

sns.barplot(data=data, y="취급액", x="상품군", orient="v", ax=axes[0][0])
sns.barplot(data=data, y="취급액", x="월", orient="v", ax=axes[0][1])
sns.barplot(data=data, y="취급액", x="요일", orient="v", ax=axes[1][0])
sns.barplot(data=data, y="취급액", x="방송시간대", orient="v", ax=axes[1][1])

axes[0][0].set(xlabel='상품군', ylabel='취급액', title='상품군별 취급액')
axes[0][1].set(xlabel='월', ylabel='취급액', title='월별 취급액')
axes[1][0].set(xlabel='요일', ylabel='취급액', title='요일별 취급액')
axes[1][1].set(xlabel='방송시간대', ylabel='취급액', title='방송시간대별 취급액')


# In[115]:


# 각 범주별 취급액 boxplot
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
fig.suptitle("각 범주별 취급액 분포", size=18)

sns.boxplot(data=data, y="취급액", x="상품군", orient="v", ax=axes[0][0])
sns.boxplot(data=data, y="취급액", x="월", orient="v", ax=axes[0][1])
sns.boxplot(data=data, y="취급액", x="요일", orient="v", ax=axes[1][0])
sns.boxplot(data=data, y="취급액", x="방송시간대", orient="v", ax=axes[1][1])

axes[0][0].set(xlabel='상품군', ylabel='취급액', title='상품군별 취급액')
axes[0][1].set(xlabel='월', ylabel='취급액', title='월별 취급액')
axes[1][0].set(xlabel='요일', ylabel='취급액', title='요일별 취급액')
axes[1][1].set(xlabel='방송시간대', ylabel='취급액', title='방송시간대별 취급액')


# ### 판매량과의 관계

# In[122]:


# 각 범주별 판매량 합계
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
fig.suptitle("각 범주별 판매량 합계", size=18)

sns.barplot(data=data, y="판매량", x="상품군", orient="v", ax=axes[0][0], estimator=np.sum)
sns.barplot(data=data, y="판매량", x="월", orient="v", ax=axes[0][1], estimator=np.sum)
sns.barplot(data=data, y="판매량", x="요일", orient="v", ax=axes[1][0], estimator=np.sum)
sns.barplot(data=data, y="판매량", x="방송시간대", orient="v", ax=axes[1][1], estimator=np.sum)

axes[0][0].set(xlabel='상품군', ylabel='판매량', title='상품군별 판매량')
axes[0][1].set(xlabel='월', ylabel='판매량', title='월별 판매량')
axes[1][0].set(xlabel='요일', ylabel='판매량', title='요일별 판매량')
axes[1][1].set(xlabel='방송시간대', ylabel='판매량', title='방송시간대별 판매량')
plt.show()

# 각 범주별 판매량
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
fig.suptitle("각 범주별 판매량 평균", size=18)

sns.barplot(data=data, y="판매량", x="상품군", orient="v", ax=axes[0][0])
sns.barplot(data=data, y="판매량", x="월", orient="v", ax=axes[0][1])
sns.barplot(data=data, y="판매량", x="요일", orient="v", ax=axes[1][0])
sns.barplot(data=data, y="판매량", x="방송시간대", orient="v", ax=axes[1][1])

axes[0][0].set(xlabel='상품군', ylabel='판매량', title='상품군별 판매량')
axes[0][1].set(xlabel='월', ylabel='판매량', title='월별 판매량')
axes[1][0].set(xlabel='요일', ylabel='판매량', title='요일별 판매량')
axes[1][1].set(xlabel='방송시간대', ylabel='판매량', title='방송시간대별 판매량')


# In[123]:


# 각 범주별 판매량 boxplot
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
fig.suptitle("각 범주별 판매량 분포", size=18)

sns.boxplot(data=data, y="판매량", x="상품군", orient="v", ax=axes[0][0])
sns.boxplot(data=data, y="판매량", x="월", orient="v", ax=axes[0][1])
sns.boxplot(data=data, y="판매량", x="요일", orient="v", ax=axes[1][0])
sns.boxplot(data=data, y="판매량", x="방송시간대", orient="v", ax=axes[1][1])

axes[0][0].set(xlabel='상품군', ylabel='판매량', title='상품군별 판매량')
axes[0][1].set(xlabel='월', ylabel='판매량', title='월별 판매량')
axes[1][0].set(xlabel='요일', ylabel='판매량', title='요일별 판매량')
axes[1][1].set(xlabel='방송시간대', ylabel='판매량', title='방송시간대별 판매량')


# ### 판매단가와의 관계
# - 합계 의미 없다.
# - 건강?

# In[116]:


# 각 범주별 판매단가
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
fig.suptitle("각 범주별 판매단가 평균", size=18)

sns.barplot(data=data, y="판매단가", x="상품군", orient="v", ax=axes[0][0])
sns.barplot(data=data, y="판매단가", x="월", orient="v", ax=axes[0][1])
sns.barplot(data=data, y="판매단가", x="요일", orient="v", ax=axes[1][0])
sns.barplot(data=data, y="판매단가", x="방송시간대", orient="v", ax=axes[1][1])

axes[0][0].set(xlabel='상품군', ylabel='판매단가', title='상품군별 판매단가')
axes[0][1].set(xlabel='월', ylabel='판매단가', title='월별 판매단가')
axes[1][0].set(xlabel='요일', ylabel='판매단가', title='요일별 판매단가')
axes[1][1].set(xlabel='방송시간대', ylabel='판매단가', title='방송시간대별 판매단가')


# In[158]:


data.groupby(by=['상품군']).count()


# In[117]:


# 각 범주별 판매단가
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
fig.suptitle("각 범주별 판매단가 분포", size=18)

sns.boxplot(data=data, y="판매단가", x="상품군", orient="v", ax=axes[0][0])]]]]]
sns.boxplot(data=data, y="판매단가", x="월", orient="v", ax=axes[0][1])
sns.boxplot(data=data, y="판매단가", x="요일", orient="v", ax=axes[1][0])
sns.boxplot(data=data, y="판매단가", x="방송시간대", orient="v", ax=axes[1][1])

axes[0][0].set(xlabel='상품군', ylabel='판매단가', title='상품군별 판매단가')
axes[0][1].set(xlabel='월', ylabel='판매단가', title='월별 판매단가')
axes[1][0].set(xlabel='요일', ylabel='판매단가', title='요일별 판매단가')
axes[1][1].set(xlabel='방송시간대', ylabel='판매단가', title='방송시간대별 판매단가')


# ### 방송 노출 시간과의 관계

# In[120]:


# 각 범주별 노출시간 합계
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
fig.suptitle("각 범주별 노출시간(분) 합계", size=18)

sns.barplot(data=data, y="노출(분)", x="상품군", orient="v", ax=axes[0][0], estimator=np.sum)
sns.barplot(data=data, y="노출(분)", x="월", orient="v", ax=axes[0][1], estimator=np.sum)
sns.barplot(data=data, y="노출(분)", x="요일", orient="v", ax=axes[1][0], estimator=np.sum)
sns.barplot(data=data, y="노출(분)", x="방송시간대", orient="v", ax=axes[1][1], estimator=np.sum)

axes[0][0].set(xlabel='상품군', ylabel='노출(분)', title='상품군별 노출시간')
axes[0][1].set(xlabel='월', ylabel='노출(분)', title='월별 노출시간')
axes[1][0].set(xlabel='요일', ylabel='노출(분)', title='요일별 노출시간')
axes[1][1].set(xlabel='방송시간대', ylabel='노출(분)', title='방송시간대별 노출시간')
plt.show()

# 각 범주별 노출시간 평균
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
fig.suptitle("각 범주별 노출시간(분) 평균", size=18)

sns.barplot(data=data, y="노출(분)", x="상품군", orient="v", ax=axes[0][0])
sns.barplot(data=data, y="노출(분)", x="월", orient="v", ax=axes[0][1])
sns.barplot(data=data, y="노출(분)", x="요일", orient="v", ax=axes[1][0])
sns.barplot(data=data, y="노출(분)", x="방송시간대", orient="v", ax=axes[1][1])

axes[0][0].set(xlabel='상품군', ylabel='노출(분)', title='상품군별 노출시간')
axes[0][1].set(xlabel='월', ylabel='노출(분)', title='월별 노출시간')
axes[1][0].set(xlabel='요일', ylabel='노출(분)', title='요일별 노출시간')
axes[1][1].set(xlabel='방송시간대', ylabel='노출(분)', title='방송시간대별 노출시간')


# In[121]:


# 각 범주별 노출시간
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
fig.suptitle("각 범주별 노출시간 분포", size=18)

sns.boxplot(data=data, y="노출(분)", x="상품군", orient="v", ax=axes[0][0])
sns.boxplot(data=data, y="노출(분)", x="월", orient="v", ax=axes[0][1])
sns.boxplot(data=data, y="노출(분)", x="요일", orient="v", ax=axes[1][0])
sns.boxplot(data=data, y="노출(분)", x="방송시간대", orient="v", ax=axes[1][1])

axes[0][0].set(xlabel='상품군', ylabel='노출(분)', title='상품군별 노출시간')
axes[0][1].set(xlabel='월', ylabel='노출(분)', title='월별 노출시간')
axes[1][0].set(xlabel='요일', ylabel='노출(분)', title='요일별 노출시간')
axes[1][1].set(xlabel='방송시간대', ylabel='노출(분)', title='방송시간대별 노출시간')


# ## 두 변수 간 관계 확인: 범주형-범주형

# In[151]:


# 월별 상품군 개수
plt.figure(figsize=(12, 6))
sns.heatmap(data.groupby(by='상품군')['월'].value_counts().unstack().fillna(0), 
            annot=True, cmap='YlGnBu')


# In[150]:


# 요일별 상품군 개수
plt.figure(figsize=(8, 6))
sns.heatmap(data.groupby(by='상품군')['요일'].value_counts().unstack().fillna(0), 
             annot=True, cmap='YlGnBu')


# In[153]:


# 방송시간대별 상품군 개수
plt.figure(figsize=(13, 6))
sns.heatmap(data.groupby(by='상품군')['방송시간대'].value_counts().unstack().fillna(0), 
             annot=True, cmap='YlGnBu')


# ### 세 변수 간 관계: (범주+범주->수치)

# In[26]:


def check_multivariate(df, x, y, cat, agg_func):
    
    temp = pd.pivot_table(df, values=y, index=x, columns=cat, aggfunc=agg_func).fillna(0)
    display(temp)
    
    # 그래프 설정        
    fig, (ax1, ax2) =plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    if agg_func == np.mean:
        plt.suptitle(f"{x}별 {cat}별 {y}: 평균", size=18)
    elif agg_func == np.sum:
        plt.suptitle(f"{x}별 {cat}별 {y}: 합계", size=18)
    else:
        plt.subtitle(f"{x}별 {cat}별 {y}: {agg_func}", size=18)

    
    # 시계열 그래프
    sns.lineplot(x=x, y=y, hue=cat, data=df, estimator=agg_func, ci=None, ax=ax1)
    ax1.legend(loc='best')
    
    # 단순 관계
    sns.heatmap(temp.T, cmap='YlGnBu', ax=ax2)
    plt.show()


# #### 판매량 + 다른 조합

# In[248]:


# 월별 상품군별 판매량 
check_multivariate(data, '월', '판매량', '상품군', np.mean) # 평균
check_multivariate(data, '월', '판매량', '상품군', np.sum) # 합계


# In[250]:


# 월별 요일별 판매량
check_multivariate(data, '월', '판매량', '요일', np.mean)
check_multivariate(data, '월', '판매량', '요일', np.sum)


# In[ ]:


# 월별 방송시간대별 판매량: 의미없다.
check_multivariate(data, '월', '판매량', '방송시간대', np.mean)
check_multivariate(data, '월', '판매량', '방송시간대', np.sum)


# In[253]:


# 상품군별 월별 판매량
check_multivariate(data, '상품군', '판매량', '월', np.mean)
check_multivariate(data, '상품군', '판매량', '월', np.sum)


# In[254]:


# 상품군별 요일별 판매량
check_multivariate(data, '상품군', '판매량', '요일', np.mean)
check_multivariate(data, '상품군', '판매량', '요일', np.sum)


# In[255]:


# 상품군별 방송시간대별 판매량
check_multivariate(data, '상품군', '판매량', '방송시간대', np.mean)
check_multivariate(data, '상품군', '판매량', '방송시간대', np.sum)


# In[257]:


# 요일별 월별 판매량
check_multivariate(data, '요일', '판매량', '월', np.mean)
check_multivariate(data, '요일', '판매량', '월', np.sum)


# In[258]:


# 요일별 상품군별 판매량
check_multivariate(data, '요일', '판매량', '상품군', np.mean)
check_multivariate(data, '요일', '판매량', '상품군', np.sum)


# In[259]:


# 요일별 상품군별 판매량
check_multivariate(data, '요일', '판매량', '상품군', np.mean)
check_multivariate(data, '요일', '판매량', '상품군', np.sum)


# In[32]:


# 요일별 상품군별 판매량
check_multivariate(data, '요일', '판매량', '방송시간대', np.mean)
check_multivariate(data, '요일', '판매량', '방송시간대', np.sum)


# #### 취급액 + 다른 조합

# In[33]:


# 월별 요일별 취급액
check_multivariate(data, '월', '취급액', '요일', np.mean)
check_multivariate(data, '월', '취급액', '요일', np.sum)


# In[41]:


# 월별 상품군별 취급액
check_multivariate(data, '월', '취급액', '상품군', np.mean)
check_multivariate(data, '월', '취급액', '상품군', np.sum)


# In[ ]:


# # 월별 방송시간대별 취급액: 의미없다
# check_multivariate(data, '월', '취급액', '방송시간대', np.mean)
# check_multivariate(data, '월', '취급액', '방송시간대', np.sum)


# In[ ]:





# # 판매량 상위로 정렬해서 확인

# In[13]:


# 상위 50개 확인
data.sort_values(by='판매량', ascending=False).head(50)


# In[14]:


# 판매량 하위 50개 확인: 0인 게 많다.
data.sort_values(by='판매량', ascending=False).tail(50)


# In[15]:


# 0인 것만 따로 떼서 확인
zero_mask = (data['판매량'] == 0)
df_zeros, df_nonzeros = data[zero_mask], data[~zero_mask]
print(len(df_zeros), len(df_nonzeros))
display(df_zeros.head())
display(df_nonzeros.tail())


# In[16]:


# 판매량 0인 상품 전부 보기: 머더코드, 상품코드 비슷한 애들 찾아야 한다(주최 측에서).
display(df_zeros)
print(f'취급액 0인 상품 개수: {len(df_zeros)}')
print(f"최대 노출: {df_zeros['노출(분)'].max()}, 최소 노출: {df_zeros['노출(분)'].min()}, 평균 노출: {df_zeros['노출(분)'].mean()}")
print(f"최대 판매단가: {df_zeros['판매단가'].max()}, 최소 판매단가: {df_zeros['판매단가'].min()}, 평균 판매단가: {df_zeros['판매단가'].mean()}")
print(f"\n{df_zeros['방송시작'].dt.month.value_counts()}")
print(f"\n{df_zeros['상품군'].value_counts()}")
print(f"\n{df_zeros['상품명'].value_counts()}")


# ## 컬럼별로 체크
# 
# '날짜', '마더코드', '상품코드', '상품명', ~~'상품군'~~, '노출(분)', '판매단가', '취급액', '방송시작',
#        '방송종료', '판매량', ~~'월'~~, '방송시간대'

# In[20]:


def checkSalesByColumnGroup(df, column, nrows, ascending=False):
    temp = df_nonzeros.sort_values(by='판매량', ascending=ascending).head(nrows)
    grps = temp[column].value_counts()
    if ascending: # 하위
        df[f'하위 {nrows}'] = 0
        for i, v in zip(grps.index, grps.values):
            df[f'하위 {nrows}'][i] = v
    else:
        df[f'상위 {nrows}'] = 0
        for i, v in zip(grps.index, grps.values):
            df[f'상위 {nrows}'][i] = v
    return df


# ### 월별 분포

# highest_sales_month = pd.DataFrame(index=range(1, 13))
# lowest_sales_month = pd.DataFrame(index=range(1, 13)) # 0인 거 제외하고!

# In[25]:


highest_sales_month = pd.DataFrame(index=data['월'].unique())
lowest_sales_month = pd.DataFrame(index=data['월'].unique())


# In[27]:


for i in range(3, 11):
    checkSalesByColumnGroup(highest_sales_month, '월', 10*i, ascending=False)
    checkSalesByColumnGroup(lowest_sales_month, '월', 10*i, ascending=True)


# In[28]:


# 상위 n개 월별 분포
plt.figure(figsize=(8,6))
sns.heatmap(highest_sales_month, annot=True, cmap='YlGnBu')
plt.yticks(rotation=-0)
plt.title('판매량 상위 n개 상품 월별 분포', size=18)
plt.show()


# In[29]:


# 하위 n개 월별 분포
plt.figure(figsize=(8,6))
sns.heatmap(lowest_sales_month, annot=True, cmap='YlGnBu')
plt.yticks(rotation=-0)
plt.suptitle('판매량 하위 n개 상품 월별 분포', size=18)
plt.title('판매량 0인 상품 제외')
plt.show()


# ### 상품군

# In[30]:


highest_sales_product_group = pd.DataFrame(index=data['상품군'].unique())
lowest_sales_product_group = pd.DataFrame(index=data['상품군'].unique()) # 0인 거 제외하고!


# In[31]:


for i in range(3, 11):
    checkSalesByColumnGroup(highest_sales_product_group, '상품군', 10*i, ascending=False)
    checkSalesByColumnGroup(lowest_sales_product_group, '상품군', 10*i, ascending=True)


# In[32]:


# 상위 n개 월별 분포
plt.figure(figsize=(8,6))
sns.heatmap(highest_sales_product_group, annot=True, cmap='YlGnBu')
plt.yticks(rotation=-0)
plt.title('판매량 상위 n개 상품 상품군별 분포', size=18)
plt.show()


# In[33]:


# 상위 n개 월별 분포
plt.figure(figsize=(8,6))
sns.heatmap(lowest_sales_product_group, annot=True, cmap='YlGnBu')
plt.yticks(rotation=-0)
plt.title('판매량 하위 n개 상품 상품군별 분포', size=18)
plt.show()

