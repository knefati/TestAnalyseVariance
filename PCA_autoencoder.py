import csv
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.python.ops.summary_ops_v2 import keras_model
#from ggplot import *
from plotnine import *
from sklearn.preprocessing import StandardScaler, scale
import matplotlib.pyplot as plt
filename = "data/df_sampleALE_allMetrics.csv"



df = pd.read_csv("data/df_sampleALE_allMetrics.csv")
df = df.dropna()

df["CECR.niveau"] = df["CECR.niveau"].map({"A1": "A1/A2",
                                           "A2": "A1/A2",
                                           "B1": "B1/B2",
                                           "B2": "B1/B2",
                                           "C1": "C1/C2",
                                           "C2": "C1/C2"})
df["K"] = 1 / df["K"]
df["Fucks"] = 1 / df["Fucks"]
df["Dickes.Steiwer"] = 1 / df["Dickes.Steiwer"]
df["DRP"] = 1 / df["DRP"]
df["Traenkle.Bailer"] = 1 / df["Traenkle.Bailer"]

df = df.rename(columns={'K': '1/K'})

# df = pd.read_csv("data/df_sampleALE_allMetrics-TTRWithStopwords.csv")

variables = ["CTTR",
             "NDW",
             "W",
             "T",
             "RIX",
             "MLT",
             "CN.T",
             "CP.T",
             "1/K",

             'Coleman', 'Coleman.C2', 'Coleman.Liau.ECP', 'Coleman.Liau.grade',
             'Coleman.Liau.short', 'Dale.Chall', 'Dale.Chall.old', 'Dale.Chall.PSK',
             'Danielson.Bryan', 'Danielson.Bryan.2', 'ELF',
             'Dickes.Steiwer', 'DRP',

             'Farr.Jenkins.Paterson', 'Flesch', 'Flesch.PSK', 'Flesch.Kincaid',
             'FOG', 'FOG.PSK',
             'FOG.NRI',  # important not increase quality after
             'FORCAST', 'FORCAST.RGL',
             'Fucks',

             'Linsear.Write', 'LIW',
             'nWS',
             'nWS.2', 'nWS.3', 'nWS.4',  # (decreasre quality)
             'Scrabble',


             # 'SMOG', 'SMOG.C', 'SMOG.simple', 'SMOG.de',
             # 'Traenkle.Bailer.2',
             # 'meanSentenceLength',
             # 'S.1',
             # 'Traenkle.Bailer.2',
             # 'Wheeler.Smith',


             'Spache',
             'Spache.old', 'Strain',  # 'Traenkle.Bailer',

             'Wheeler.Smith',
             'meanWordSyllables',

             'Danielson.Bryan', 'Danielson.Bryan.2',
             'Dickes.Steiwer',
             'DRP',
             'ELF',

             'Farr.Jenkins.Paterson',  # 'Flesch',
             'Flesch.PSK', 'Flesch.Kincaid',

             'FOG', 'FOG.PSK',

             'FOG.NRI', 'FORCAST', 'FORCAST.RGL',

             'Linsear.Write', 'LIW', 'nWS', 'nWS.2', 'nWS.3',
             'nWS.4',

             'Scrabble', 'SMOG', 'SMOG.C', 'SMOG.simple', 'SMOG.de',
             'Spache', 'Spache.old', 'Strain',
             'Traenkle.Bailer',  # (to see the inverse)
             'meanSentenceLength', 'meanWordSyllables',

             'VP.T', 'C.T', 'DC.C', 'DC.T', 'T.S', 'CT.T', 'CP.C'
             ]


# scaler = StandardScaler()
# scaler.fit(data)
# x_train = scaler.transform(data)
# x_train = pd.DataFrame(x_train)
df_clust = df[variables]
x_train = df_clust
x_train.index =df[df.columns[0]]
data_size = x_train.shape[1]

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=data_size)
mu = np.mean(x_train, axis=0)
mat = np.tile(mu, (x_train.shape[0], 1))
x = x_train - mat #scale(x_train)
# pca.fit(x)
principalComponents = pca.fit_transform(x_train)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC' + str(i+1) for i in range(data_size)])
# eignvalues = pca.explained_variance_
# principalComponents = pca.components_
finalDf = pd.concat([principalDf, df["CECR.niveau"]], axis = 1)
def hat_x_pca(k):
    eignvectors = np.matrix(pca.components_).T
    xhat = np.asmatrix(principalComponents[:,0:k]) * eignvectors[:,0:k].T + mat
    return xhat
mse_pca = []
for k in range(data_size):
    xhat = hat_x_pca(k+1)
    mse = np.mean(np.square(xhat - np.asmatrix(x_train)))
    mse_pca.append(mse)
print(mse_pca)
#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
# var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
#
# plt.plot(var1)
#
# fig1= plt.figure()
# plt.plot(mse_pca, ylab="Errors", xlab="number of variables")
#
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# etab = ['ASKORIA', 'IFPEK', 'PFPS']
# colors = ['r', 'g', 'b']
# for target, color in zip(etab, colors):
#     indicesToKeep = finalDf['etablissement'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
#                , finalDf.loc[indicesToKeep, 'PC2']
#                , c = color
#                , s = 50)
# ax.legend(etab)
# ax.grid()
# plt.show()

from keras.layers import Input, Dense
from keras.models import Model

def pca_autoencoder(input_size, list_units, pca_dim=2, activation='relu'):
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
        intermediate_layer = Dense(pca_dim, activation=activation)(encoded)
        layer = intermediate_layer
        i = len(list_units) -1
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


mse_autoencoder = []

for k in range(data_size):
    autoencoder, encoder = pca_autoencoder(input_size=x_train.shape[1],
                                           pca_dim=k+1,
                                           list_units=[6],
                                           activation='linear') # tanh relu

    autoencoder.compile(optimizer='adam', loss='mse')#, loss='mae')
    x_train = x_train.astype('float32')

    autoencoder.fit(x=x_train,
                    y=x_train,
                    epochs=100,
                    #batch_size=150,
                    #validation_split=0.1
                    )

    intermediate_output = encoder.predict(x_train)

    xtrain_autoenc_hat=autoencoder.predict(x_train)
    mse_auto = np.mean((xtrain_autoenc_hat - x_train) ** 2)
    mse_autoencoder.append(mse_auto)

print(mse_auto)

fig, ax = plt.subplots()
res = {"mse PCA": mse_pca,
       "mse_autoencoder": mse_autoencoder}
for key, mse_values in res.items():
    ax.plot(mse_values, label=key)
ax.legend()
plt.show()


exit(0)
dd={"PC1":intermediate_output[:,0],
    "PC2":intermediate_output[:,1],
    "etablissement": df["CECR.niveau"]}
print(autoencoder.summary())
print("***********************")
print("MSE Autoencoder: {}".format(mse_autoencoder))
print("***********************")
print(ggplot(pd.DataFrame(dd), aes(x='PC1', y='PC2', colour="etablissement")) + geom_point())
####################

# this is the size of our encoded representations
# pca_dim = 2  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# data_size = x_train.shape[1]
# # this is our input placeholder
# input_data = Input(shape=(data_size,))
# # "encoded" is the encoded representation of the input
# encoded = Dense(units=encoding_dim, activation='relu')(input_data)
# # "decoded" is the lossy reconstruction of the input
# decoded = Dense(units=data_size, activation='relu')(encoded)  #sigmoid
# # this model maps an input to its reconstruction
# autoencoder = Model(input_data, decoded)

# input_data = Input(shape=(data_size,))
# encoded = Dense(15, activation='relu')(input_data)
# encoded = Dense(5, activation='relu')(encoded)
#
# encoded = Dense(pca_dim, activation='relu')(encoded)
#
# decoded = Dense(5, activation='relu')(encoded)
# decoded = Dense(15, activation='relu')(decoded)
# decoded = Dense(data_size, activation='sigmoid')(decoded)
#
# autoencoder = Model(input_data, decoded)
# print(autoencoder.summary())
#
# # this model maps an input to its encoded representation
# encoder = Model(input_data, encoded)

# create a placeholder for an encoded  input
# encoded_input = Input(shape=(2,))
# decoder_layer = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer(encoded_input))

# encoded = Dense(units=10, activation='relu')(input_data)
# encoded = Dense(units=encoding_dim, activation='relu')(encoded)
# # "decoded" is the lossy reconstruction of the input
# decoded = Dense(units=10, activation='relu')(encoded)  #sigmoid
# decoded = Dense(units=data_size, activation='relu')(decoded)  #sigmoid

# retrieve the last layer of the autoencoder model
# decoder_layer2 = autoencoder.layers[-1]
# decoder_layer1 = autoencoder.layers[-2]
# create the decoder model

# decoder = Model(input=encoded_input, output=decoder_layer2(decoder_layer1))



# df_aisance=df[df.columns[11:17]]
# avis = {"Très difficile": -2,
#         "Difficile": -1,
#         "Ne me correspond pas":0,
#         "Abordable":1,
#         "Facile":2}
# df_aisance=df_aisance.replace(to_replace="Très difficile", value=-2)
# df_aisance=df_aisance.replace(to_replace="Difficile", value=-1)
# df_aisance=df_aisance.replace(to_replace="Ne me correspond pas", value=0)
# df_aisance=df_aisance.replace(to_replace="Abordable", value=1)
# df_aisance=df_aisance.replace(to_replace="Facile", value=2)
