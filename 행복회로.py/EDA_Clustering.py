import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.ndimage import gaussian_filter1d

# 1. KMeans 클러스터링 후 시각화.
# 향후 기능 분리 필요
def kmeansClusterPlot(num_cluster, data, plotCentroid=True, plotData=True, returnCluster=False):
    
    # 군집화    
    data_temp = data.copy()
    X_data = data_temp.values
    km = KMeans(n_clusters=num_cluster, init='k-means++', n_init=300, max_iter=10000, random_state=42)
    km.fit(X_data)
    y = km.predict(X_data)
    data_temp['Cluster'] = y
    
    # 중심 패턴 시각화
    if plotData:
        fig = plt.figure(figsize=(16, 6))
        colors = "bgrcmykw"
        centXY = km.cluster_centers_
        for i in range(num_cluster):
            p = fig.add_subplot(2, (num_cluster+1)//2, i+1)
            p.plot(centXY[i], 'b-o', markersize=3,
            color=colors[np.random.randint(0,7)],linewidth=1.0)
            p.set_title(f'Cluster {i}')
        plt.suptitle('Cluster 별 중심 패턴', size=18)
        plt.show()
    
    # 전체 시각화
    if plotData:
        for c in sorted(data_temp['Cluster'].unique().tolist()):
            c_temp = data_temp[data_temp['Cluster'] == c]
            print(f"Cluster {c}에 해당하는 일자: {c_temp.index.tolist()}")
            c_temp = c_temp.reset_index(drop=True).iloc[:, :-1]
            plt.figure(figsize=(16, 4))
            for i in range(len(c_temp)):
                plt.plot(c_temp.iloc[i])
            plt.title(f"Cluster {c}", size=18)
            plt.xticks([c_temp.columns[0], c_temp.columns[-1]])
            plt.show()
    
    if returnCluster:
        return data_temp['Cluster']