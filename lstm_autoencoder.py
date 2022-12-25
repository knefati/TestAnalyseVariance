# lstm autoencoder reconstruct and predict sequence
from numpy import array
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Flatten, GRU
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.utils import plot_model
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd


def pca_autoencoder(input_size, list_units, pca_dim=2, activation='relu'):
    input_data = Input(shape=input_size)
    if len(list_units) > 0:
        layer = input_data
        i = 0
        while i < len(list_units):
            units = list_units[i]
            encoded = LSTM(units=units, activation=activation, input_shape=(input_size[0], units))(layer)
            # encoded = LSTM(100, activation='relu')(layer)

            layer = encoded
            i += 1
        # bottleneck
        intermediate_layer = Dense(pca_dim, activation=activation, input_shape=(input_size[0], pca_dim))(encoded)
        layer = intermediate_layer
        i = len(list_units) - 1
        while i >= 0:
            units = list_units[i]
            decoded = LSTM(units=units, activation=activation, input_shape=(input_size[0], units))(layer)
            layer = decoded
            i -= 1
        decoded = LSTM(input_size, activation=activation)(decoded)
    else:
        intermediate_layer = LSTM(pca_dim, activation=activation)(input_data)
        decoded = LSTM(input_size, activation=activation)(intermediate_layer)
    autoencoder = Model(input_data, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_data, intermediate_layer)
    return autoencoder, encoder


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

important_variabels = features_score["Features"][0:11]

df_interest = df[important_variabels]  # df

x_train = df_interest.values


seq_in = x_train
# reshape input into [samples, timesteps, features]
n_in = x_train.shape[0]
n_features = x_train.shape[1]
seq_in = seq_in.reshape((1, n_in, n_features))

# define encoder
visible = Input(shape=(n_in,1))
encoder = LSTM(100, activation='relu')(visible)
flat = Flatten()(encoder)
decoders = []
# define reconstruct decoder
#for j in range(n_features):
decoder1 = RepeatVector(n_in)(flat)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)
decoders.append(decoder1)

decoder2 = RepeatVector(n_in)(flat)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)
decoders.append(decoder1)

# tie it together
model = Model(inputs=visible, outputs=decoders)
model.compile(optimizer='adam', loss='mse')
# fit model
seq_out = [x_train[:,j] for j in range(n_features)]
model.fit(seq_in, seq_out, epochs=300, verbose=0)
# demonstrate prediction
yhat = model.predict(seq_in, verbose=0)




seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))
# define encoder
visible = Input(shape=(n_in,1))
encoder = LSTM(100, activation='relu')(visible)
# define reconstruct decoder
decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)
# define predict decoder
decoder2 = RepeatVector(n_in)(encoder)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)
# tie it together
model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse')
plot_model(model, show_shapes=True, to_file='composite_lstm_autoencoder.png')
# fit model
model.fit(seq_in, [seq_in,seq_in], epochs=300, verbose=0)
# demonstrate prediction
yhat = model.predict(seq_in, verbose=0)

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


def lstm_autoencoder(nb_units, x_shape, activation="linear"):
    timesteps = x_shape[1]
    n_features = x_shape[2]
    input_layer = Input(shape=(timesteps, n_features))
    encoded = LSTM(nb_units, activation=activation)(input_layer)
    layer_repeat_vec = RepeatVector(timesteps)(encoded)
    # bottleneck
    intermediate_layer = Dense(2, activation=activation)(layer_repeat_vec)

    decoded = LSTM(nb_units, activation=activation)(intermediate_layer)
    decoded = Dense(n_features)(decoded)

    # time_distib_layer = TimeDistributed(Dense(n_features))(decoded)

    autoencoder = Model(input_layer, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_layer, intermediate_layer)
    return autoencoder, encoder

xx_train = np.ndarray(shape=(x_train.shape[0] + 2, x_train.shape[1]))
xx_train[0, :] = x_train[0, :]
xx_train[-1, :] = x_train[-1, :]
xx_train[1:-1, :] = x_train
timesteps = 1
X, y = temporalize(X=xx_train, y=np.zeros(len(xx_train)), lookback=timesteps)

n_features = df_interest.shape[1]
X = np.array(X)
X = X.reshape(X.shape[0], timesteps, n_features)


autoencoder, encoder = lstm_autoencoder(62, X.shape)

autoencoder.compile(optimizer='adam', loss='mae')  # , loss='mse' 'mae')

autoencoder.fit(x=X,
                y=X,
                epochs=100,
                verbose=1
                # batch_size=150,
                # validation_split=0.1
                )

yy = encoder.predict(X)
intermediate_output = yy[:, 0, :]

df_clust = intermediate_output
n_clusters = len(set(data["CECR.niveau"]))

T = intermediate_output
df_T = pd.DataFrame(T)
km = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=50, random_state=42).fit(intermediate_output)

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

df_T['kmeans_cluster'] = km.labels_
# df_T.to_csv("csvs/dim_reduction_{}epochs_{}variables.csv".format(epochs, len(variables)))
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
# plt.savefig("imgs/dim_reduction_{}epochs_{}variables.png".format(epochs, len(variables)))
plt.show()
