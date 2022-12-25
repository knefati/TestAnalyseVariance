
exit()
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, GRU, Dropout
from keras.models import Model
from sklearn.cluster import KMeans

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# from numpy.random import seed

# seed(1)
# import tensorflow as tf

# tf.random.set_seed(0)


def pca_autoencoder(input_size, list_units, pca_dim=2, activation='relu'):
    input_data = Input(shape=input_size)
    if len(list_units) > 0:
        layer = input_data
        i = 0
        while i < len(list_units):
            units = list_units[i]
            encoded = Dense(units, activation=activation)(layer)
            layer = encoded
            i += 1
        # bottleneck
        intermediate_layer = Dense(pca_dim, activation=activation)(encoded)
        intermediate_layer = Dropout(0.5)(intermediate_layer)
        layer = intermediate_layer
        i = len(list_units) - 1
        while i >= 0:
            units = list_units[i]
            decoded = Dense(units, activation=activation)(layer)
            layer = decoded
            i -= 1
        decoded = Dense(input_size, activation=activation)(decoded)
    else:
        intermediate_layer = Dense(pca_dim, activation=activation)(input_data)
        decoded = Dense(input_size, activation=activation)(intermediate_layer)
    autoencoder = Model(input_data, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_data, intermediate_layer)
    return autoencoder, encoder


data = pd.read_csv("data/df_sampleALE_allMetrics2.csv")
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
        raw_variables.append(new_var_name)
        title = "Features importance after inversing:\n{}".format(features_inverse)
    else:
        raw_variables.append(var)
        title = "Features importance without inversing any feature"

raw_df = data[raw_variables]
corr_variables = raw_df.corr()
variables_high_corr = []
# find one correlation variable
for i, var1 in enumerate(raw_variables):
    j = i + 1
    while (j < len(raw_variables)):
        var2 = raw_variables[j]
        if corr_variables[var1][var2] > 0.98:
            variables_high_corr.append((var1, var2))
            j = len(raw_variables)
        j += 1
var_to_remove = []
for var1, var2 in variables_high_corr:
    var_to_remove.append(var2)

variables = list(set(raw_variables) - set(var_to_remove))
df = raw_df[variables]

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
features_score = pd.DataFrame({"Features": df.columns,
                               "scores": scores_features})
features_score = features_score.sort_values('scores', ascending=False)

alpha = 0.05
nb_important_variabels = len(np.array(variables)[test.pvalues_ < alpha])
important_variabels = features_score["Features"][0:nb_important_variabels]

df_clust = df[important_variabels]  # df

n_clusters = len(set(data["CECR.niveau"]))
x_train = df_clust
x_train.index = df[df.columns[0]]
data_size = x_train.shape[1]

epochs = 100
# pca_dim, unit1,unit2 = 3,31,11 #2, 31, 41#3, 3, 51#4, 31, 61 #3, 41, 1 #3, 31, 51#2,291,1#2,251,11# = 2,227#2, 185
pca_dim, unit1 = 2, 227
autoencoder, encoder = pca_autoencoder(input_size=x_train.shape[1],
                                       pca_dim=pca_dim,
                                       list_units=[unit1],
                                       activation='linear')  # tanh relu linear

autoencoder.compile(optimizer='adam', loss='mae')  # , loss='mse' 'mae')
x_train = x_train.astype('float32')

autoencoder.fit(x=x_train,
                y=x_train,
                epochs=epochs,
                verbose=0
                # batch_size=150,
                # validation_split=0.1
                )

intermediate_output = encoder.predict(x_train)
df_clust = intermediate_output

km = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=50, random_state=42).fit(df_clust)
# data['kmeans_cluster'] = km.labels_
# cross_tbl = pd.crosstab(data["kmeans_cluster"], data["CECR.niveau"], margins=False)
cross_tbl = pd.crosstab(km.labels_, data["CECR.niveau"], margins=False)
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

print(val_max_percent)

T = intermediate_output
df_T = pd.DataFrame(T)
km = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=50, random_state=42).fit(intermediate_output)
df_T['kmeans_cluster'] = km.labels_
df_T.to_csv("csvs/dim_reduction_{}epochs_{}variables.csv".format(epochs, len(variables)))
labels = df_T["kmeans_cluster"]
import matplotlib.pyplot as plt

colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#EEACC5', '#FFFC34', '#4E9ADD']

plt.figure(figsize=(20, 20))
for i in range(n_clusters):
    Ti = df_T[df_T["kmeans_cluster"] == i].values
    plt.scatter(Ti[:, 0], Ti[:, 1], c=colors[i],  # edgecolors='r',
                label="Cluster{}".format(i + 1))

for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x + 1.5, y + 1.5), xytext=(0, 0), textcoords='offset points')
plt.legend()
plt.grid(True)
plt.savefig("imgs/dim_reduction_{}epochs_{}variables.png".format(epochs, len(variables)))
# plt.show()
