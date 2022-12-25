from scipy.stats import kendalltau
# Feature Selection with Univariate Statistical Tests
import pandas as pd
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# load data
import numpy as np

data = pd.read_csv("/home/knefati/Documents/MyWork-Ensai/python-environement/VisLing/data/df_sampleALE_allMetrics2.csv")
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
                 "NDWC",
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
data["NDWC"] = data["NDWC"]/data["NDW"]

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
        if corr_variables[var1][var2] > 0.99:
            variables_high_corr.append((var1, var2))
            j = len(raw_variables)
        j += 1
var_to_remove = []
for var1, var2 in variables_high_corr:
    var_to_remove.append(var2)

print("number of removed variables: {}".format(len(var_to_remove)))

variables = list(set(raw_variables) - set(var_to_remove))
df = raw_df[variables]

X = df.values
Y = data["CECR.niveau"]

res = {}
for k, v in df.items():
    tau, p_value = kendalltau(v, Y)
    res.update({k: [tau, p_value]})
df_res = pd.DataFrame(res)
df_res.index = ["Spearmans correlation coefficient","p_value"]
df_res1 = df_res.sort_values(by="p_value", axis=1)