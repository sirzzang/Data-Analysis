#!/usr/bin/env python
# coding: utf-8

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

# 경로 설정
SALES_DATA = './data/방송시간데이터.xlsx'

# 데이터 불러오기
data = pd.read_excel(SALES_DATA, parse_dates=True, usecols=lambda x: 'Unnamed' not in x)
data.head()

# 컬럼 추가
data['판매량'] = data['취급액'] / data['판매단가']
data['월'] = data['방송시작'].dt.month
data['요일'] = data['방송시작'].dt.dayofweek

day_mapping_dict = {0:'월요일', 1:'화요일', 2:'수요일', 3:'목요일', 4:'금요일', 5:'토요일', 6:'일요일'}
data['요일'] = data['요일'].map(day_mapping_dict)

def time_to_str(x):
    return x.strftime('%H')
data['방송시간대'] = data['방송시작'].dt.time.apply(lambda x: time_to_str(x))

# 범주형 변수 변경
data['날짜'] = pd.Categorical(data['날짜'])
data['마더코드'] = pd.Categorical(data['마더코드'])
data['상품코드'] = pd.Categorical(data['상품코드'])
data['상품군'] = pd.Categorical(data['상품군'])
data['월'] = pd.Categorical(data['월'])
data['요일'] = pd.Categorical(data['요일'])
data['방송시간대'] = pd.Categorical(data['방송시간대'])
data.info()

# 무형 상품 제외
data = data[data['상품군'] != '무형'].reset_index(drop=True)

# 종속변수 분포
sns.distplot(data['취급액'])

# 종속변수 히스토그램 적절한 경계값 찾기
for i in range(5, 30):
    _, bins, patches = plt.hist(data['취급액'], bins=i)
    plt.show()
    print(f"{i}개의 범주일 때 경계값: {bins}\n")

# 수치형 변수 요약통계량
data.describe()

# 수치형 변수 간 상관관계
display(data[['노출(분)', '판매단가', '취급액', '판매량']].corr())
sns.pairplot(data[['노출(분)', '판매단가', '취급액', '판매량']])
plt.show()

# 수치형 변수 간 상관관계: 취급액 구간 정보 추가
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

sns.pairplot(data[['노출(분)', '판매단가', '취급액', '판매량', '취급액_범주']], hue='취급액_범주')
plt.show()

# 노출시간
sns.distplot(data['노출(분)'])

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

_, bins, patches = ax1.hist(data['노출(분)'])
_, bins2, patches2 = ax2.hist(data[(data['노출(분)'] >= bins[2]) & (data['노출(분)'] < bins[3])]['노출(분)'], bins=20) # 몰린 구간 확대

# 판매단가
sns.distplot(data['판매단가'])

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

_, bins, patches = ax1.hist(data['판매단가'])
_, bins2, patches2 = ax2.hist(data[(data['판매단가'] >= bins[3]) & (data['판매단가'] < bins[4])]['판매단가'], bins=20) # 몰린 구간 확대

# 판매량
sns.distplot(data['판매량'])

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

_, bins, patches = ax1.hist(data['판매량'])
_, bins2, patches2 = ax2.hist(data[(data['판매량'] >= bins[0]) & (data['판매량'] < bins[1])]['판매량'], bins=20) # 몰린 구간 확대

# 방송시간대 분포
sns.countplot(x='방송시간대', data=data, )

# 방송 요일 분포
sns.countplot(x='요일', data=data, order=['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'])

# 상품군 분포
sns.countplot(x='상품군', data=data)

# 마더코드 건드려 보자. 응 아니야... 응? 뭔가 많은 게 있긴 한데.
sns.countplot(x='마더코드', data=data)

# 각 범주형 변수와 수치형 변수 간 sum, mean 확인
def grouping(df, colname, agg_funcs):
    print(f"집계 기준 범주: {colname}")
    display(data.groupby(by=colname, as_index=False).agg(agg_funcs).sort_values(by=('취급액', 'mean'), ascending=False))
    print()

for col in ['상품군', '마더코드', '방송시간대', '요일']:
    grouping(data, col, ['sum', 'mean'])

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

# 월별 상품군 개수
plt.figure(figsize=(12, 6))
sns.heatmap(data.groupby(by='상품군')['월'].value_counts().unstack().fillna(0), 
            annot=True, cmap='YlGnBu')

# 요일별 상품군 개수
plt.figure(figsize=(8, 6))
sns.heatmap(data.groupby(by='상품군')['요일'].value_counts().unstack().fillna(0), 
             annot=True, cmap='YlGnBu')

# 방송시간대별 상품군 개수
plt.figure(figsize=(13, 6))
sns.heatmap(data.groupby(by='상품군')['방송시간대'].value_counts().unstack().fillna(0), 
             annot=True, cmap='YlGnBu')

# 세 변수 간 관계
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

# 월별 상품군별 판매량 
check_multivariate(data, '월', '판매량', '상품군', np.mean) # 평균
check_multivariate(data, '월', '판매량', '상품군', np.sum) # 합계

# 판매량 상위로 정렬해서 확인
data.sort_values(by='판매량', ascending=False).head(50) # 상위 50개
data.sort_values(by='판매량', ascending=False).tail(50) # 하위 50개: 0인 게 많음.

# 판매량 0인 것만 따로 떼서 확인
zero_mask = (data['판매량'] == 0)
df_zeros, df_nonzeros = data[zero_mask], data[~zero_mask]
print(len(df_zeros), len(df_nonzeros))

# 판매량 0인 상품 전부 보기: 머더코드, 상품코드 비슷한 애들 찾아야 한다(주최 측에서).
display(df_zeros)
print(f'취급액 0인 상품 개수: {len(df_zeros)}')
print(f"최대 노출: {df_zeros['노출(분)'].max()}, 최소 노출: {df_zeros['노출(분)'].min()}, 평균 노출: {df_zeros['노출(분)'].mean()}")
print(f"최대 판매단가: {df_zeros['판매단가'].max()}, 최소 판매단가: {df_zeros['판매단가'].min()}, 평균 판매단가: {df_zeros['판매단가'].mean()}")
print(f"\n{df_zeros['방송시작'].dt.month.value_counts()}")
print(f"\n{df_zeros['상품군'].value_counts()}")
print(f"\n{df_zeros['상품명'].value_counts()}")

# 컬럼별로 판매량 상위, 하위 데이터 확인
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

# 월별 판매량 상위, 하위 100개
highest_sales_month = pd.DataFrame(index=data['월'].unique())
lowest_sales_month = pd.DataFrame(index=data['월'].unique())
for i in range(3, 11):
    checkSalesByColumnGroup(highest_sales_month, '월', 10*i, ascending=False)
    checkSalesByColumnGroup(lowest_sales_month, '월', 10*i, ascending=True)

# 상위 n개 월별 분포
plt.figure(figsize=(8,6))
sns.heatmap(highest_sales_month, annot=True, cmap='YlGnBu')
plt.yticks(rotation=-0)
plt.title('판매량 상위 n개 상품 월별 분포', size=18)
plt.show()

# 하위 n개 월별 분포
plt.figure(figsize=(8,6))
sns.heatmap(lowest_sales_month, annot=True, cmap='YlGnBu')
plt.yticks(rotation=-0)
plt.suptitle('판매량 하위 n개 상품 월별 분포', size=18)
plt.title('판매량 0인 상품 제외')
plt.show()