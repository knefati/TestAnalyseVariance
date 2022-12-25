# Feature Selection with Univariate Statistical Tests
from math import gamma
from scipy.stats.stats import spearmanr
import pandas as pd
from numpy import set_printoptions
from scipy.special import psi
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# load data
import numpy as np
from sklearn.neighbors import NearestNeighbors

from bussiness.features_selection import features_selection

data = pd.read_csv("/home/knefati/Documents/MyWork-Ensai/python-environement/VisLing/data/df_sampleALE_allMetrics.csv")
data = data.dropna()

data["CECR.niveau"] = data["CECR.niveau"].map({"A1": "A1/A2",
                                               "A2": "A1/A2",
                                               "B1": "B1/B2",
                                               "B2": "B1/B2",
                                               "C1": "C1/C2",
                                               "C2": "C1/C2"})

raw_variables = ["CTTR",
                 "NDW",
                 "W",
                 # "NDWC",
                 "T",
                 "RIX",
                 "MLT",
                 "CN.T",
                 "CP.T",
                 "R",
                 'Coleman.C2',
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
data["NDWC"] = data["NDWC"] / data["NDW"]

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

Y = data["CECR.niveau"]
raw_df = data[raw_variables]

fs = features_selection()
# important_variabels, features_score = fs.selecte_relevant_feature(raw_df, output=Y, method_correlation="pearson")
important_variabels, features_score, corr_variables, variables_high_corr = fs.selecte_relevant_feature(raw_df,
                                                                                                       output=Y,
                                                                                                       method_correlation="spearman")

for i, v in enumerate(important_variabels):
    # print("variable {}: {}".format(i+1,v))
    print(v)

import matplotlib.pyplot as plt

# plt.figure(figsize=(14, 8))
# features_score = features_score.sort_values('p-value', ascending=True)
# plt.bar(features_score["Features"], features_score["p-value"], color='black')
# plt.axhline(y=0.05, color='r', linestyle='-')
# plt.xlabel('Features')
# plt.ylabel('p-value')
# plt.margins(x=0, y=0.1)
# plt.title(title)
# plt.subplots_adjust(bottom=0.23)
# plt.xticks(features_score["Features"], rotation=90)
# plt.show()

features_score = features_score.sort_values('scores', ascending=False)

freq_series = features_score["scores"]

# Plot the figure.
plt.figure(figsize=(14, 8))
ax = freq_series.plot(kind='bar', fontsize=16)
ax.set_title('Features importance', fontweight='bold', fontsize=20)
ax.set_xlabel('Feature', fontweight='bold', fontsize=16)
ax.set_ylabel('Score', fontweight='bold', fontsize=16)
ax.set_xticklabels(features_score["Features"], fontweight='bold', fontsize=13)

# Make labels.
scores = features_score["p-value"].to_list()
labels_num = np.round(scores, 4) * 100
labels = ["{:.2f}".format(lbl) + " %" if lbl > 1e-6 else "0 %" for lbl in labels_num]
rects = ax.patches

n_features = len(important_variabels)
pos_feature_n = rects[n_features - 1].get_x() + rects[n_features - 1].get_width() / 2
pos_feature_n1 = rects[n_features].get_x() + rects[n_features].get_width() / 2
plt.axvline(x=(pos_feature_n + pos_feature_n1) / 2, color='r', linestyle='-')

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2,
            height + 0.5, label,
            ha='center', va='bottom', rotation=90, fontweight='bold')
ax.text(rects[6].get_x(),
        rects[6].get_height() + 25,
        r'p < .05', color='b',
        fontsize=20, fontweight='bold', rotation=90)
ax.text(rects[20].get_x(),
        rects[6].get_height() + 25,
        r'p >= .05', color='b',
        fontsize=20, fontweight='bold', rotation=90)
plt.subplots_adjust(bottom=0.23)


#

# Plot the figure.
features_score1 = features_score[features_score['p-value'] < 0.05]
freq_series1 = features_score1["scores"]

plt.figure(figsize=(14, 8))
ax = freq_series1.plot(kind='barh', fontsize=16)
ax.set_title('Features importance', fontweight='bold', fontsize=20)
ax.set_ylabel('Feature', fontweight='bold', fontsize=20)
ax.set_xlabel('Score', fontweight='bold', fontsize=20)
ax.set_yticklabels(features_score1["Features"], fontweight='bold', fontsize=20)
plt.xticks(fontsize=20, fontweight='bold')

# Make labels.
scores = features_score["p-value"].to_list()
labels_num = np.round(scores, 4) * 100
labels = ["{:.2f}".format(lbl) + " %" if lbl > 1e-6 else "0 %" for lbl in labels_num]
plt.tight_layout()

plt.show()

# plt.bar(features_score["Features"], features_score["scores"], color='black')
# plt.axvline(x=11.5, color='r', linestyle='-')
# plt.xlabel('Features')
# plt.ylabel('score')
# plt.margins(x=0, y=0.1)
# plt.title(title)
# plt.subplots_adjust(bottom=0.23)
# plt.xticks(features_score["Features"], rotation=90)
