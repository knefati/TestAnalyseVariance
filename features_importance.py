# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd



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

             'Farr.Jenkins.Paterson', 'Flesch', 'Flesch.PSK', 'Flesch.Kincaid',
             'FOG', 'FOG.PSK',
             'FOG.NRI',  # important not increase quality after
             'FORCAST', 'FORCAST.RGL',

             'Linsear.Write', 'LIW',
             'nWS',
             'nWS.2', 'nWS.3', 'nWS.4',  # (decreasre quality)
             'Scrabble',


             'SMOG', 'SMOG.C', 'SMOG.simple', 'SMOG.de',
             'Traenkle.Bailer.2',
             'meanSentenceLength',
             'S.1',
             'Traenkle.Bailer.2',
             'Wheeler.Smith',
             'Flesch',

             'Spache',
             'Spache.old', 'Strain',

             'Wheeler.Smith',
             'meanWordSyllables',

             'Danielson.Bryan', 'Danielson.Bryan.2',


             'ELF',

             'Farr.Jenkins.Paterson',
             'Flesch.PSK', 'Flesch.Kincaid',

             'FOG', 'FOG.PSK',

             'FOG.NRI', 'FORCAST', 'FORCAST.RGL',

             'Linsear.Write', 'LIW', 'nWS', 'nWS.2', 'nWS.3',
             'nWS.4',

             'Scrabble', 'SMOG', 'SMOG.C', 'SMOG.simple', 'SMOG.de',
             'Spache', 'Spache.old', 'Strain',
             'meanSentenceLength', 'meanWordSyllables',

             'VP.T', 'C.T', 'DC.C', 'DC.T', 'T.S', 'CT.T', 'CP.C'
             ]
features_inverse = ["K", "Fucks", "Dickes.Steiwer", "DRP", "Traenkle.Bailer"]
for var in features_inverse:
    data[var] = 1 / data[var]
    new_var_name = '1/{}'.format(var)
    data = data.rename(columns={var: new_var_name})
    variables.append(new_var_name)

df = data[variables]

X = df.values
Y = data["CECR.niveau"]

# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
features_importance = model.feature_importances_

features_score = pd.DataFrame({"Features":df.columns,
                               "features_importance":features_importance})
features_score = features_score.sort_values('features_importance', ascending=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(14, 8))
plt.bar(features_score["Features"], features_score["features_importance"], color='black')
plt.xlabel('Features')
plt.ylabel('Features importance')
plt.xticks(features_score["Features"], rotation=90)
plt.show()