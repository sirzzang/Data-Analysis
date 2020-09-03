#!/usr/bin/env python
# coding: utf-8

# 모듈 불러오기 및 설정
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='Malgun Gothic') # 맑은 고딕 설정
plt.rc('axes', unicode_minus=False) # 마이너스 폰트 설정

# 컬럼 추가 및 변경
data['방송날짜'] = data['방송시작'].dt.date
data['판매량'] = data['취급액'] / data['판매단가']
data['월'] = data['방송시작'].dt.month

season_mapping_dict = {1:'겨울', 2:'겨울', 3:'봄', 4:'봄', 5:'봄', 6:'여름', \
    7:'여름', 8:'여름', 9:'가을', 10:'가을', 11:'가을', 12:'겨울'}
data['계절'] = data['월'].map(season_mapping_dict)

data['요일'] = data['방송시작'].dt.dayofweek
day_mapping_dict = {0:'월요일', 1:'화요일', 2:'수요일', 3:'목요일', 4:'금요일', 5:'토요일', 6:'일요일'}
data['요일'] = data['요일'].map(day_mapping_dict)

holiday = pd.read_csv('./data/국경일공휴일기념일절기.csv')
holiday_date = holiday['date'].tolist()
holiday_mapping_dict = {}
for date in holiday_date:
    holiday_mapping_dict[date] = 'Y'
data['공휴일_여부'] = data['방송시작'].map(holiday_mapping_dict)
data['공휴일_여부'] = data['공휴일_여부'].fillna('N')

def time_to_str(x):
    return x.strftime('%H')
data['방송시간대'] = data['방송시작'].dt.time.apply(lambda x: time_to_str(x))

data = data[['방송날짜', '월', '계절', '요일', '공휴일_여부', '방송시작', '방송종료', '방송시간대', '노출(분)', \
    '상품군', '마더코드', '상품코드', '상품명', '판매단가', '판매량', '취급액']]

# 범주형 변수 변경
data['계절'] = pd.Categorical(data['계절'])
data['요일'] = pd.Categorical(data['요일'])
data['방송시간대'] = data['방송시간대'].astype('int')
data['공휴일_여부'] = pd.Categorical(data['공휴일_여부'])
data['상품군'] = pd.Categorical(data['상품군'])
data['마더코드'] = pd.Categorical(data['마더코드'])
data['상품코드'] = pd.Categorical(data['상품코드'])
data['판매단가'] = data['판매단가'].astype('float')
data['취급액'] = data['취급액'].astype('float')
data.info()

# 통계량 기반 이상치 확인
def get_outlier(df, col, threshold=75):
    boundary = np.percentile(df[col], threshold)
    out_df = df[df[col] >= boundary]
    return out_df

# One Variable: countplot, distplot, ...

# TwoVariables

# 각 피쳐 간 sum, mean 확인
# 각 피쳐 간 sum, mean 확인
def grouping(df, colname, agg_funcs):
    print(f"집계 기준 범주: {colname}")
    grouped_df = df.groupby(by=colname, as_index=False).agg(agg_funcs).sort_values(by=('취급액', 'mean'), ascending=False)
    display(grouped_df)
    return grouped_df

# 순위별 함수
def plotRanks(df, col_list, agg_func_list, ascending=False):
    df['1'] = df[(col_list[0], agg_func_list[0])].rank(ascending=ascending)
    df['2'] = df[(col_list[1], agg_func_list[1])].rank(ascending=ascending)
    temp = df[['1', '2']].reset_index()
    
    x, y, n = temp['1'], temp['2'], temp.iloc[:, 0]
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.scatter(x, y)
    ax.grid(True)
    ax.set_xlabel(f'{col_list[0]} {agg_func_list[0]} 순위')
    ax.set_ylabel(f'{col_list[1]} {agg_func_list[1]} 순위')
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))    
    plt.show()
    
# 각 범주별 수치와의 관계
def plotCatNum(df, x_list, y, plot_type, agg_func=None):
    assert len(x_list) % 2 == 0
    
    # 그래프 유형
    if plot_type == 'bar':
        plot = sns.barplot        
        # bar type일 때 집계 방법
        if agg_func == 'sum':
            func = np.sum
        elif agg_func == 'mean':
            func = np.mean
    elif plot_type == 'box':
        plot = sns.boxplot
    else:
        plot = plot_type
        print("plot 유형 정확한지 확인")
    
    row_num, col_num = 2, len(x_list)//2
    
    fig, axes = plt.subplots(nrows=row_num, ncols=col_num)
    fig.set_size_inches(col_num*8, row_num*6)
    
    for i in range(len(x_list)):
        row, col = i // col_num, i % col_num
        
        if plot_type=='bar':
            if agg_func == 'sum':
                sns.barplot(data=data, y=y, x=x_list[i], orient='v', ax=axes[row][col], estimator=np.sum)
            elif agg_func == 'mean':
                sns.barplot(data=data, y=y, x=x_list[i], orient='v', ax=axes[row][col], estimator=np.mean)
            else:
                try:
                    sns.barplot(data=data, y=y, x=x_list[i], orient='v', ax=axes[row][col], estimator=agg_func)
                except:
                    return "집계함수 확인할 것"
                
        elif plot_type=='box':
            sns.boxplot(data=data, y=y, x=x_list[i], orient='v', ax=axes[row][col])
            agg_func = ''
    
    # 요일인 경우, 월화수목금토일 순서로 정렬.
    if y == '요일':
        locs, labels = plt.xticks()
        plt.xticks(locs, ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'])
    
    fig.suptitle(f'각 범주별 {y} {agg_func}', size=24)    
    plt.show()

# 범주형-범주형 개수 파악
def countCatCat(df, x, y):
    plt.figure(figsize=(12, 6))
    sns.heatmap(data.groupby(by=x)[y].value_counts().unstack().fillna(0),
                annot=True, cmap='YlGnBu', fmt='g')
    plt.title(f'{y}별 {x}의 개수')
    
    # 요일인 경우, 월화수목금토일 순서로 정렬.
    if y == '요일':
        locs, labels = plt.xticks()
        plt.xticks(locs, ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'])

    plt.show()

# Three Variables

# 종류별 집계 후 관계 파악
def check_multivariate(df, x, y, cat, agg_func):
    
    temp = pd.pivot_table(df, values=y, index=x, columns=cat, aggfunc=agg_func).fillna(0)
    
    # 그래프 설정        
    fig, (ax1, ax2) =plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    if agg_func == np.mean:
        fig.suptitle(f"{x}별 {cat}별 {y}: 평균", size=18)
    elif agg_func == np.sum:
        fig.suptitle(f"{x}별 {cat}별 {y}: 합계", size=18)
    else:
        fig.suptitle(f"{x}별 {cat}별 {y}", size=18)
    
    # 시계열 그래프
    sns.lineplot(x=x, y=y, hue=cat, data=df, estimator=agg_func, ci=None, ax=ax1)
    ax1.legend(loc='center left', ncol=1, bbox_to_anchor=(-0.3, 0.5))
    
    # 단순 관계
    sns.heatmap(temp.T, cmap='YlGnBu', ax=ax2)
    
    # 요일인 경우, 월화수목금토일 순서로 정렬.
    if x == '요일':
        ax1.set_xticks(range(0, 7))
        ax1.set_xticklabels(['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'])
        ax2.set_xticklabels(['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'])
        
    plt.show()
    
# 판매량 상위, 하위
def checkSalesByColumnGroup(df, column, nrows, ascending=False):
    temp = data.sort_values(by='판매량', ascending=ascending).head(nrows)
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


# 상위 n개 화살표 annotation
def annotMaxValues(x, y, n=10, ax=None):
    x_max_values = x[y.argsort()[-n:][::-1]]
    y_max_values = y[x_max_values]
    for xmax, ymax in zip(x_max_values, y_max_values):
        # text= "x={}, y={:.3f}".format(xmax, ymax)"
        text = xmax
        if not ax:
            ax = plt.gca()
        # bboxprops = dict(boxstyle=square, pad=0.3, fc=w, ec=k, lw=0.72)
        arrowprops = dict(arrowstyle='->', connectionstlye='angle, angleA=0, angleB=60')
        ax.annotate(text, xy=(xmax, ymax), xycoords='data',
                    xytext=(xmax-8, ymax), textcoords='data',
                    arrowprops=arrowprops, #bbox=bboxprops,
                    ha='right', va='top')