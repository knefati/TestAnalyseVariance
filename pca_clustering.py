# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Set data
df = pd.DataFrame({
    'group': ['A', 'B', 'C', 'D'],
    'var1': [38, 1.5, 30, 4],
    'var2': [29, 10, 9, 34],
    'var3': [8, 39, 23, 24],
    'var4': [7, 31, 33, 14],
    'var5': [28, 15, 32, 14]
})

# number of variable
categories = list(df)[1:]
N = len(categories)

# We are going to plot the first line of the data frame.
# But we need to repeat the first value to close the circular graph:
values = df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
values

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
plt.ylim(0, 40)

# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')

# Fill area
ax.fill(angles, values, 'b', alpha=0.1)


































# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.manifold import TSNE

# df = pd.read_csv("data/cohort_df_sampleALE_allMetrics.csv")
# variables = ["CTTR", "W", "T", "RIX", "MLT", "CN.T", "CP.T", "K"]

data = pd.read_csv("data/df_sampleALE_allMetrics.csv")
data = data.dropna()

data["CECR.niveau"] = data["CECR.niveau"].map({"A1": "A1/A2",
                                               "A2": "A1/A2",
                                               "B1": "B1/B2",
                                               "B2": "B1/B2",
                                               "C1": "C1/C2",
                                               "C2": "C1/C2"})

variables = ["CTTR",
             "NDW",
             "W",
             "T",
             "RIX",
             "MLT",
             "CN.T",
             "CP.T",

             'Coleman', 'Coleman.C2', 'Coleman.Liau.ECP', 'Coleman.Liau.grade',
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
             'S.1',
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

var_inverse = True
for var in features_inverse:
    if var_inverse:
        data[var] = 1 / data[var]
        new_var_name = '1/{}'.format(var)
        data = data.rename(columns={var: new_var_name})
        variables.append(new_var_name)
        title = "Features importance after inversing:\n{}".format(features_inverse)
    else:
        variables.append(var)
        title = "Features importance without inversing any feature"

df = data[variables]
n_clusters = len(set(data["CECR.niveau"]))


X = df.values
Y = data["CECR.niveau"]

# feature extraction
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
scores_features = fit.scores_
# print(scores_features)
features = fit.transform(X)
# summarize selected features
# print(features[0:5,:])
features_score = pd.DataFrame({"Features":df.columns,
                               "scores":scores_features})
features_score = features_score.sort_values('scores', ascending=False)

important_variabels = features_score["Features"][0:11]

df = df[important_variabels]#df

plt.figure(figsize=(14, 8))
X_std = StandardScaler().fit_transform(df)  # Create a PCA instance: pca
pca = PCA(n_components=df.shape[1])
principalComponents = pca.fit_transform(X_std)  # Plot the explained variances
# features = range(pca.n_components_)
# var = pca.explained_variance_ratio_
# plt.bar(features, var, color='black')
# plt.xlabel('PCA features')
# plt.ylabel('variance %')
# plt.xticks(features)
# plt.show()
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)

df_clust = PCA_components  # [[0,1,2]]

km = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=50, random_state=42).fit(df_clust)
data['kmeans_cluster'] = km.labels_
cross_tbl = pd.crosstab(data["kmeans_cluster"], data["CECR.niveau"], margins=False)
print(cross_tbl.sum(0))
print(cross_tbl)
print(round(100 * cross_tbl / cross_tbl.sum(0), 2))
T = df_clust.values
df_T = df_clust
df_T['kmeans_cluster'] = km.labels_
labels = df_T["kmeans_cluster"]
import matplotlib.pyplot as plt

colors = ['#4EACC5', '#FF9C34', '#4E9A06']

plt.figure(figsize=(20, 20))
for i in range(n_clusters):
    Ti = df_T[df_T["kmeans_cluster"] == i].values
    plt.scatter(Ti[:, 0], Ti[:, 1], c=colors[i],  # edgecolors='r',
                label="Cluster{}".format(i + 1))

for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x + 1.5, y + 1.5), xytext=(0, 0), textcoords='offset points')
plt.legend()
plt.grid(True)
plt.show()

# plt.figure(figsize=(14, 8))
#
# plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')

# fig, axes = plt.subplots(1,2)
# axes[0].scatter(df["CTTR"], df["NDW"])
# axes[0].set_xlabel('x1')
# axes[0].set_ylabel('x2')
# axes[0].set_title('Before PCA')
#
# axes[1].scatter(principalComponents[:,0], principalComponents[:,1])#, c=data["CECR.niveau"])
# axes[1].set_xlabel('PC1')
# axes[1].set_ylabel('PC2')
# axes[1].set_title('After PCA')
# plt.show()

# find clusters

# ks = range(1, 10)
#
# inertias = []
# for k in ks:
#     # Create a KMeans instance with k clusters: model
#     model = KMeans(n_clusters=k)
#
#     # Fit model to samples
#     model.fit(PCA_components.iloc[:, :3])
#
#     # Append the inertia to the list of inertias
#     inertias.append(model.inertia_)
#
# plt.figure(figsize=(14, 8))
# plt.plot(ks, inertias, '-o', color='black')
# plt.xlabel('number of clusters, k')
# plt.ylabel('inertia')
# plt.xticks(ks)
# plt.show()
