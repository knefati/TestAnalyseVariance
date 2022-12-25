import numpy as np
import pandas as pd
from keras.layers import Input, Dense, LSTM, RepeatVector, GRU, Dropout
from keras.models import Model
from numpy import set_printoptions
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# from numpy.random import seed
#
# seed(1)
# import tensorflow as tf
#
# tf.random.set_seed(0)
from sklearn.manifold import TSNE

from bussiness.features_selection import features_selection


def pca_autoencoder(input_size, list_units, dim_encoder=2, activation='relu'):
    input_data = Input(shape=(input_size,))
    if len(list_units) > 0:
        layer = input_data
        i = 0
        while i < len(list_units):
            units = list_units[i]
            encoded = Dense(units, activation=activation)(layer)
            layer = encoded
            i += 1
        # bottleneck
        intermediate_layer = Dense(dim_encoder, activation=activation)(encoded)
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
        intermediate_layer = Dense(dim_encoder, activation=activation)(input_data)
        decoded = Dense(input_size, activation=activation)(intermediate_layer)
    autoencoder = Model(input_data, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_data, intermediate_layer)
    return autoencoder, encoder


def lstm_autoencoder(nb_units, x_shape, activation="linear"):
    timesteps = x_shape[1]
    n_features = x_shape[2]
    input_layer = Input(shape=(timesteps, n_features))
    encoded = GRU(nb_units, activation=activation)(input_layer)
    layer_repeat_vec = RepeatVector(timesteps)(encoded)
    # bottleneck
    intermediate_layer = Dense(2, activation=activation)(layer_repeat_vec)

    decoded = GRU(nb_units, activation=activation)(intermediate_layer)
    decoded = Dense(n_features)(decoded)

    # time_distib_layer = TimeDistributed(Dense(n_features))(decoded)

    autoencoder = Model(input_layer, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_layer, intermediate_layer)
    return autoencoder, encoder

def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # Gather past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
        output_y.append(y[i + lookback + 1])
    return output_X, output_y


algo = "Dense"#"Dense" # "LSTM", "GRU"

# load data

data = pd.read_csv("/home/knefati/Documents/MyWork-Ensai/python-environement/VisLing/data/df_sampleALE_allMetrics.csv")
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

Y = data["CECR.niveau"]

fs = features_selection()
important_variabels, features_score, df_corr, variables_high_corr = fs.selecte_relevant_feature(df, output=Y, method_correlation="spearman",#"pearson"
                                                                  )


df_clust = df[important_variabels]

n_clusters = len(set(data["CECR.niveau"]))
x_train = df[variables]
x_train.index = df[df.columns[0]]
data_size = x_train.shape[1]

if algo !="Dense":
    x_train1 = x_train.values
    xx_train = np.ndarray(shape=(x_train1.shape[0] + 2, x_train1.shape[1]))
    xx_train[0, :] = x_train1[0, :]
    xx_train[-1, :] = x_train1[-1, :]
    xx_train[1:-1, :] = x_train1
    timesteps = 1
    X, y = temporalize(X=xx_train, y=np.zeros(len(xx_train)), lookback=timesteps)

import itertools

list_dim_encoder = [2, 3, 4]
hidden_units1 = list(range(1, 100, 3))
epochs = 100
#hidden_units2 = list(range(1, 100, 10))
activations = ["linear"]
possibilities = [list_dim_encoder, activations, hidden_units1
                 #,hidden_units2
                 ]

list_results = []
possibs = list(itertools.product(*possibilities))
print("Ther are {} configurations".format(len(possibs)))
l = 0
for element in possibs:
    l += 1
    dim_encoder = element[0]
    activation = element[1]
    list_units = element[2:]
    if algo == "Dense":
        autoencoder, encoder = pca_autoencoder(input_size=x_train.shape[1],
                                               dim_encoder=dim_encoder,
                                               list_units=list_units,
                                               activation=activation)  # tanh relu linear

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
    else:
        n_features = df.shape[1]
        X = np.array(X)
        X = X.reshape(X.shape[0], timesteps, n_features)
        unit1 = element[2]
        autoencoder, encoder = lstm_autoencoder(unit1, X.shape, activation=activation)

        autoencoder.compile(optimizer='adam', loss='mae')  # , loss='mse' 'mae')

        autoencoder.fit(x=X,
                        y=X,
                        epochs=epochs,
                        verbose=0
                        # batch_size=150,
                        # validation_split=0.1
                        )

        yy = encoder.predict(X)
        intermediate_output = yy[:, 0, :]

    km = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=50, random_state=42).fit(intermediate_output)
    # data['kmeans_cluster'] = km.labels_
    # cross_tbl = pd.crosstab(data["kmeans_cluster"], data["CECR.niveau"], margins=False)
    cross_tbl = pd.crosstab(km.labels_, data["CECR.niveau"], margins=False)
    # print(cross_tbl.sum(0))
    # print(cross_tbl)
    mat = round(100 * cross_tbl / cross_tbl.sum(0), 2)
    # print(mat)

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

    print("config({},{},{}): {}/{}".format(dim_encoder, activation, [(unit) for unit in list_units], l, len(possibs)), val_max_percent)
    # print(mat)
    list_results.append(val_max_percent)
df_results = pd.DataFrame(list_results)
df_results.index = possibs


def f(v):
    return all([any(np.array(v) > 60),
                any(np.array(v) > 50),
                any(np.array(v) > 40)])


keep = df_results.apply(f, axis=1)
df_results = df_results[keep]

print("res",df_results)
if df_results.shape[0] > 0:
    i = df_results.sum(axis=1).argmax()
    best_config = df_results.index[i]
    best_val_max_percent = df_results.iloc[i]

    print("best config is {}".format(best_config))
    print("best vector is \n {}".format(best_val_max_percent))
    df_results.to_csv("csvs/results_{}configs_{}_epochs_{}variables.csv".format(df_results.shape[0], epochs, len(variables)))

    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#EEACC5', '#FFFC34', '#4E9ADD']

    dim_encoder = best_config[0]
    activation = best_config[1]
    list_units = best_config[2:]
    autoencoder, encoder = pca_autoencoder(input_size=x_train.shape[1],
                                           dim_encoder=dim_encoder,
                                           list_units=list_units,
                                           activation=activation)  # tanh relu linear

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
    if dim_encoder > 2:
        tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=3)
        np.set_printoptions(suppress=True)
        T = tsne.fit_transform(df_clust)
    else:
        T = intermediate_output
    df_T = pd.DataFrame(T)
    km = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=50, random_state=42).fit(intermediate_output)
    df_T['kmeans_cluster'] = km.labels_
    df_T.to_csv("csvs/dim_reduction_{}epochs_{}variables.csv".format(epochs, len(variables)))
    labels = df_T["kmeans_cluster"]
    import matplotlib.pyplot as plt


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
    plt.show()
else:
    print("\n\n[INFO] There are no results")
