
df0 = read.csv("/home/knefati/Documents/MyWork-Ensai/python-environement/VisLing/data/df_sampleALE_allMetrics.csv", 
              encoding="UTF-8", dec=",", stringsAsFactors=FALSE)

variables = c("CTTR",
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
)


features_inverse = c("K", "Fucks", "Dickes.Steiwer", "DRP", "Traenkle.Bailer")
inverse_variables = FALSE
for (variable in features_inverse){
  if (inverse_variables){
  df0[variable] = 1/sapply(df0[variable], as.numeric)
  new_var_name = paste0('1/',variable)
  colnames(df0)[which(colnames(df0)==variable)]=new_var_name
  }else{
    new_var_name = variable
  }
  variables = c(variables, new_var_name)
}

df = df0[variables]
df = df[complete.cases(df), ] # remove rows having missing values  
df = as.data.frame(sapply(df, as.numeric))

PCA_data = df
library(FactoMineR)

# 

pca_facto = PCA(PCA_data
                #quanti.sup = 9:10, 
                #quali.sup = 11
)
#plot(pca_facto#, habillage = 11, invisible = "quali"
#)
# show the students whome the cos2 (contribution) > 0.7
#plot(pca_facto, cex=0.5,  invisible = "quali", select = "cos2 0.7")
# show the  40 student which have the biggest contribution 
plot(pca_facto, cex=0.5,  invisible = "quali", select = "contrib 40")
dimdesc(pca_facto)
summary(pca_facto)

df_conrib = pca_facto$var$contrib

df_conrib[
  order(df_conrib[,1]+ df_conrib[,2], decreasing=TRUE ),
]


