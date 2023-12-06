import pandas as pd
import csv
import matplotlib.pyplot as plt


## Transformation du premier fichier de données en DF

file2 = r'donnees_01.txt'
df2 = pd.read_csv(file2, sep=';', encoding=open(file2).encoding)

print("transformation des donnees en dataframe")

df2 = df2.drop(columns=["DATE_INTERVENTION", "P_IPP", "NOM_COMMERCIAL"]) # On enlève les informations inutiles pour la suite


## Homogénéisation du poids en kg

for line in df2.index: 
    if float(df2["POIDS"][line]) >= 1000:
        df2["POIDS"][line] = float(df2["POIDS"][line])/1000

df2 = df2.drop(columns=["POIDS_UNITE"])


## Affichage des quantitéq prises de Lorazepam 1mg cp

df = df2.loc[df2["CODE_UCD"] == 9090884]
L = [0,0,0,0,0,0,0,0]
for i in df.index:
    if df["QTE_PRESC"][i] == 0.25:
        L[0] = L[0]+1
    elif df["QTE_PRESC"][i] == 0.5:
        L[1] = L[1]+1
    elif df["QTE_PRESC"][i] == 1.:
        L[2] = L[2]+1
    elif df["QTE_PRESC"][i] == 1.5:
        L[3] = L[3]+1
    elif df["QTE_PRESC"][i] == 2.:
        L[4] = L[4]+1
    elif df["QTE_PRESC"][i] == 2.5:
        L[5] = L[5]+1
    elif df["QTE_PRESC"][i] == 3.:
        L[6] = L[6]+1
    elif df["QTE_PRESC"][i] == 10.:
        L[7] = L[7]+1
print(L)

#Duree =[]
#Duree.append()
## Affichage des quantités prises pour une forme de médicament en fonction du poids, ici le ceftriaxone 1g inj

UCD = df2["CODE_UCD"].unique()
CDI = df2["DCI"].unique()
ceftriaxone = df2.loc[df2["CODE_UCD"] == UCD[7]]
X_cef = ceftriaxone["POIDS"]
Y_cef = ceftriaxone["QTE_PRESC"]
plt.scatter(X_cef,Y_cef)
plt.show()