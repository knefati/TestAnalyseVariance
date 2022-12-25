from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
class features_selection:
    def __init__(self):
        pass

    def selecte_relevant_feature(self, df_features, output, method_correlation="pearson", alpha=0.05):
        """

        :param df_features:
        :param output:
        :param method_correlation:
        :param alpha: test level
        :return: list of relevant variables
        """
        variables_name = df_features.columns
        corr_variables = df_features.corr(method=method_correlation)
        variables_high_corr = []
        # find one correlation variable
        for i, var1 in enumerate(variables_name):
            j = i + 1
            while (j < len(variables_name)):
                var2 = variables_name[j]
                if corr_variables[var1][var2] >= 0.99 or corr_variables[var1][var2] <= -0.99:
                    variables_high_corr.append((var1, var2, corr_variables[var1][var2]))
                    j = len(variables_name)
                j += 1
        var_to_remove = []

        for var1, var2, corr in variables_high_corr:
            var_to_remove.append(var2)

        print("number of removed variables: {}".format(len(var_to_remove)))

        variables = list(set(variables_name) - set(var_to_remove))
        df = df_features[variables]

        X = df.values

        # feature extraction
        test = SelectKBest(score_func=f_classif, k=4)
        fit = test.fit(X, output)
        # summarize scores
        set_printoptions(precision=3)
        scores_features = fit.scores_
        # print(scores_features)
        features = fit.transform(X)
        # summarize selected features
        # print(features[0:5,:])
        features_score = pd.DataFrame({"Features": df.columns,
                                       "p-value": test.pvalues_,
                                       "scores": scores_features})
        features_score = features_score.sort_values('scores', ascending=False)
        nb_important_variabels = len(np.array(variables)[test.pvalues_ < alpha])

        important_variabels = features_score["Features"][0:nb_important_variabels]
        return important_variabels, features_score, corr_variables, variables_high_corr