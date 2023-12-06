import pandas as pd
import csv
import datetime

"""
i = 0

with open("donnees.txt","r") as donnees:
    part1 = open("donnees_01.txt", "a")
    part2 = open("donnees_02.txt", "a")
    part3 = open("donnees_03.txt", "a")
    for line in donnees:
        x = line.replace(",",".")
        if i <= 1000000:
                #with open("donnees_01.txt", "a") as part1:
            part1.write(x)
        elif 1000001 <= i and i <= 2000000:
                #with open("donnees_02.txt", "a") as part2:
            part2.write(x)
        else:
                #with open("donnees_03.txt", "a") as part3:
            part3.write(x)
        i = i + 1
    part1.close()
    part2.close()  
    part3.close()
"""


file2 = r'donnees_01.txt'
#file2 = r'echantillon.txt'
#print(open(file2).encoding)
df2 = pd.read_csv(file2, sep=';', encoding=open(file2).encoding)

print("transformation des donnees en dataframe")
#print(df2)

df2 = df2.drop(columns=["DATE_INTERVENTION", "P_IPP", "NOM_COMMERCIAL"])
#print(df2['DATE_FIN'])

#df2['duree'] = df2['DATE_FIN']-df2['DATE_DEBUT']

#print(df2["POIDS"])

for line in df2.index:    
    
#    if df2["POIDS"].isna()[line]:
#        1+1
#    else:
    if float(df2["POIDS"][line]) >= 1000:
        df2["POIDS"][line] = float(df2["POIDS"][line])/1000
    

print(df2[["QTE_PRESC","UNITE_PRESC","CODE_UCD"]])

mycolumns = ['QTE_PRESC','UNITE_PRESC']

        
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