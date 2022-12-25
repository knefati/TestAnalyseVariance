import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation, Birch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# load data
from bussiness.features_selection import features_selection

# data = pd.read_csv("/home/knefati/Documents/MyWork-Ensai/python-environement/VisLing/data/df_sampleALE_allMetrics.csv")
data = pd.read_csv("/home/knefati/Documents/MyWork-Ensai/python-environement/VisLing/data/df_sampleALE_allMetrics_new.csv")
data = data.dropna()

data["CECR.niveau"] = data["CECR.niveau"].map({"A1": "A1/A2",
                                               "A2": "A1/A2",
                                               "B1": "B1/B2",
                                               "B2": "B1/B2",
                                               "C1": "C1/C2",
                                               "C2": "C1/C2"})
raw_variables = ["CTTR",
                 "NDW",
                 "NDWC",
                 "W",
                 "T",
                 "RIX",
                 "MLT",
                 "CN.T",
                 "CP.T",
                 "R",
                 'Coleman', 'Coleman.Liau.ECP', 'Coleman.Liau.grade',
                 'Coleman.Liau.short', 'Dale.Chall', 'Dale.Chall.old', 'Dale.Chall.PSK',
                 'Danielson.Bryan', 'Danielson.Bryan.2', 'ELF',

                 'FOG', 'FOG.PSK',
                 'FOG.NRI',  # important not increase quality after
                 'FORCAST', 'FORCAST.RGL',

                 'Linsear.Write', 'LIW',
                 'nWS',
                 'nWS.2', 'nWS.3',

                 'SMOG', 'SMOG.C', 'SMOG.simple', 'SMOG.de',
                 'Traenkle.Bailer.2',
                 'meanSentenceLength',
                # 'S.1',
                 'Wheeler.Smith',
                 'Flesch',

                 'Spache',
                 'Spache.old', 'Strain',

                 'meanWordSyllables',

                 'Farr.Jenkins.Paterson',
                 'Flesch.PSK', 'Flesch.Kincaid',

                 'nWS.4',

                 'Scrabble',
                 'VP.T', 'C.T', 'DC.C', 'DC.T', 'T.S', 'CT.T', 'CP.C'
                 ]
features_inverse = ["K", "Fucks", "Dickes.Steiwer", "DRP", "Traenkle.Bailer"]

data["NDWC"] = data["NDWC"]/data["NDW"]

var_inverse = True
for var in features_inverse:
    if var_inverse:
        data[var] = 1 / data[var]
        new_var_name = '1/{}'.format(var)
        data = data.rename(columns={var: new_var_name})
        raw_variables.append(new_var_name)
        title = "Features importance after inversing:\n{}".format(features_inverse)
    else:
        raw_variables.append(var)
        title = "Features importance without inversing any feature"

raw_df = data[raw_variables]
Y = data["CECR.niveau"]

fs = features_selection()
important_variabels, features_score, df_corr, variables_high_corr = fs.selecte_relevant_feature(raw_df, output=Y, method_correlation="spearman")
df_clust = raw_df[important_variabels]  # df

# pca = PCA(n_components=5, svd_solver='full')
# df_pca = pca.fit_transform(raw_df.values)
# km = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=50, random_state=42).fit(df_pca)

n_clusters = len(set(data["CECR.niveau"]))

km = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=50, random_state=42).fit(df_clust)
# km = Birch(n_clusters=n_clusters,  threshold=0.01).fit(df_clust)
data['kmeans_cluster'] = km.labels_
cross_tbl = pd.crosstab(data["kmeans_cluster"], data["CECR.niveau"], margins=False)
print(cross_tbl.sum(0))
print(cross_tbl)
mat = round(100 * cross_tbl / cross_tbl.sum(0), 2)
print(mat)
mat_copy = mat.copy()
val_max_percent = []
choiced_k = []
for col in mat_copy.columns:
    m = mat_copy[col]
    for k in choiced_k:
        m[k] = -1
    k = m.argmax()
    choiced_k.append(k)
    val_max_percent.append(m[k])
print(np.sum(val_max_percent))
# print("-"*20)
exit()
k_means_cluster_centers = km.cluster_centers_

# clustering = AffinityPropagation(random_state=5).fit(df_clust)
corpus_clusters = data.sort_values(
    by=['kmeans_cluster'], ascending=False)

colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#EEACC5', '#FFFC34', '#4E9ADD']

tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=3)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(df_clust)
df_T = pd.DataFrame(T)

labels = data["kmeans_cluster"]  # df['kmeans_cluster']
plt.figure(figsize=(14, 14))
for i in range(n_clusters):
    Ti = df_T[data["kmeans_cluster"] == i].values
    plt.scatter(Ti[:, 0], Ti[:, 1], c=colors[i],  # edgecolors='r',
                label="Cluster{}".format(i + 1))

for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x + 1.5, y + 1.5), xytext=(0, 0), textcoords='offset points')
plt.legend()
plt.title("Kmeans algorithm with dim reduction\n t-distributed Stochastic Neighbor Embedding")
plt.grid(True)
plt.savefig("imgs/kmeans_{}variables.png".format(len(variables)))
plt.show()
