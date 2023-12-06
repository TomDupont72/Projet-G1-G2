import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


import gc
gc.collect()

#Nettoyage des données et des séparateurs

nbsep = 100
#linea
#lineb
nblineprec = 0
i = 0

donnees = open("donnees.txt", "a")
donnees.close()

with open("donnees.txt","a") as donnees:
    with open("securimed.csv", "r") as tf:
        for line in tf: 
            #print(line)
            nbline = line.count(';')
            if nbline == nbsep :
                donnees.write(line)
            elif nblineprec == 0:
                lineprec = line
                nblineprec = nbline
            elif nblineprec + nbline < nbsep :
                lineprec = lineprec.replace("\n", line)
                nblineprec = nblineprec + nbline
            elif nblineprec + nbline == nbsep :
                lineprec = lineprec.replace("\n", line)
                donnees.write(lineprec)
                nblineprec = 0
            
            i = i + 1
			

#On continue le data cleaning, avec par exemple les colonnes où il manque des valeurs

df = pd.read_csv('G:\Data\prevention_iatrogenie_centrale_5\securimed.csv',encoding='ISO-8859-1',sep=';',error_bad_lines=False,skiprows=[i for i in range(300000,2900000)], usecols=['P_ORD_C_NUM', 'LIBELLE_MEDICAMENT'])
ddf=df[0:1000000]
del df
#input_fd = open('G:\Data\prevention_iatrogenie_centrale_5\securimed.csv')
 #df = pd.read_csv(input_fd, engine='python', sep=',', quotechar= '"', error_bad_lines=False, quoting=3)
#df=pd.read_csv('G:\Data\prevention_iatrogenie_centrale_5\securimed.csv')
##### Filration des données ####
Df=ddf.drop(['P_IPP','LIBELLE_MEDICAMENT', 'VOIE_ADMIN', 'UNITE_PRESC', 'TYPE_FREQUENCE', 'HEURE_PRISE', 'MIN_PRISE', 'FREQUENCE', 'NON_QUOTIDIEN','UNITE_PRESC','DATE_INTERVENTION','P_ORD_C_NUM', 'P_LPM_C_NUM', 'P_LCP_C_NUM', 'P_LPO_C_NUM', 'DATE_DEBUT', 'DATE_FIN', 'CODE_UCD', 'CODE_ATC', 'NOM_COMMERCIAL', 'DCI','COMMENTAIRE_POSO','MOMENT','CONDITION','POIDS_DATE','TAILLE_DATE','MDRD_DATE','CKDEPI_DATE', 'POIDS_UNITE','TAILLE_UNITE','MDRD_UNITE', 'MDRD_DATE','MDRD_UNITE', 'MDRD_DATE','CKDEPI_UNITE', 'CKDEPI_DATE','BILIT_UNITE', 'BILIT_DATE','PAL_UNITE', 'PAL_DATE','TGO_UNITE', 'TGO_DATE', 'TGP_UNITE', 'TGP_DATE','LEUCC_UNITE', 'LEUCC_DATE', 'ERYTH_UNITE', 'ERYTH_DATE', 'HHB_UNITE', 'HHB_DATE', 'HTE_UNITE', 'HTE_DATE', 'VGM_UNITE', 'VGM_DATE',  'TCMH_UNITE', 'TCMH_DATE', 'CCMH_UNITE', 'CCMH_DATE', 'NPLAQ_UNITE', 'NPLAQ_DATE', 'VPM_UNITE', 'VPM_DATE','TCAM_UNITE', 'TCAM_DATE', 'TCAT_UNITE', 'TCAT_DATE',  'RTCAMT_UNITE', 'RTCAMT_DATE',  'TQM_UNITE', 'TQM_DATE', 'TQT_UNITE', 'TQT_DATE',  'RTQMT_UNITE', 'RTQMT_DATE',  'TP_UNITE', 'TP_DATE', 'FIB_UNITE', 'FIB_DATE'],axis=1)
del ddf
list(ddf.columns)
size=ddf.shape
print("le nombre de colonne est " , size[1])
print("le nombre de  de ligne est  " , size[0])
list(df['LIBELLE_MEDICAMENT'].isnull()).count(True)
list(df['SEXE'].isnull()).count(True)
# exploration des données :
#Trouver des colonnes avec des données manquantes
df.isnull().any()
# obtenir une liste des colonnes avec des données manquantes 
df.columns[df.isnull().any]
#nombre de données manquantes par colonnes :
df.isnull().sum()
#nombre de données manquantes par une colonne donnée :
df['cln'].isull().sum()
#Obtenir des colonnes avec le plus grand nombre de donné manquantes :
df.isnull().sum().nlargest(1)
#Nombre de données manquantes dans la data frame :
df.isnull().sum()
#1-Suprimer les colonnes avec plus de 50% de donnée manquantes :
column_with_nan=Df.columns[Df.isnull().any()]
column_with_nan
for columns in column_with_nan :
    if Df[columns].isnull().sum()*100/Df.shape[0] > 50 :
        Df.drop(columns , 1 ,inplace=True)
Df.shape

#2-remplir les valeurs non null  :

Df['SEXE']=Df['SEXE'].map({'M' : 1 ,'F' : 0} ,na_action=None)
def isnumber(s):
    try:
        a=float(str(s).replace(',','.'))

        return a
    except ValueError :
        return np.nan



def iscaracter(s):
    try:

        a=str(s).replace(',','.').replace('/','').replace(':','').replace('.','').replace(' ','')
        float(a)

        return np.nan

    except ValueError :
        
        return a

#Etape suivante de preprocessing

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



# aprés la filtration on va faire le clustering 




# cols sont les données conservés pour le clustering

cols=['SEXE', 'AGE',  'BILIT', 'PAL', 'TGO', 'TGP', 'LEUCC','ERYTH', 'HHB', 'HTE', 'VGM', 'TCMH', 'CCMH', 'NPLAQ', 'VPM', 'TCAM','TCAT', 'RTCAMT', 'TQM',  'TP', 'FIB']

## ici on va extraire les données biologiques - les filtrer - apres faire le clustring 
df = pd.read_csv('G:\Data\prevention_iatrogenie_centrale_5\securimed.csv',encoding='ISO-8859-1',sep=';',error_bad_lines=False , skiprows=[i for i in range(300000,2900000)] , usecols=cols[1:])

# cette fonction ( isnumber) permet de garder que les valeurs numérique pour une colonne de type float

def isnumber(s):
    try:
        a=float(str(s).replace(',','.'))

        return a
    except ValueError :
        return np.nan

## les colonnes de la biologies
cols=['SEXE', 'AGE',  'BILIT', 'PAL', 'TGO', 'TGP', 'LEUCC','ERYTH', 'HHB', 'HTE', 'VGM', 'TCMH', 'CCMH', 'NPLAQ', 'VPM', 'TCAM','TCAT', 'RTCAMT', 'TQM',  'TP', 'FIB']
Df=df
# libération de mémoire 

del df
Df['SEXE']=Df['SEXE'].map({'M' : 1 ,'F' : 0} ,na_action=None)

## eliminer les outliers :
for l in cols[1:] :
    Df[l]=Df[l].apply(isnumber)
    upper_lim = Df[l].quantile(.95)
    under_lim=Df[l].quantile(.05)
    Df[l][~(Df[l]<under_lim)&(Df[l]>upper_lim)]=np.nan

## remplacer les valeurs manquantes par une loi gaussienne 
n=Df.shape[0]
for l in cols[1:] :
    normale=np.abs(np.random.normal(Df[l].mean(),Df[l].std(),n))
    index_with_nan=Df[l].index[Df[l].isnull()]
    Df[l].iloc[index_with_nan] = normale[index_with_nan]
    
1

## Normalisation :

min_max_scaler=preprocessing.MinMaxScaler()
np_scaler= min_max_scaler.fit_transform(Df)
Df_normalized=pd.DataFrame(np_scaler,columns=cols[1:])
Df_normalized

del Df
## faire le clustering selon la biologie du patient 

## 1ére étape : determiner le nombre de cluster par la méthode blowmethode



from sklearn.cluster import KMeans 
inertie=[]

for i in range(1,15):
    model = KMeans(n_clusters=i)
    model.fit(Df_normalized) 
    inertie.append(model.inertia_)





plt.plot(range(1,15),inertie)
plt.title("La méthode Eblow")
plt.xlabel("nombre de clusters")
plt.ylabel("Inertie")
plt.show()

# donc par cette méthode on voie que le nombre de cluster doit etre 2

model=KMeans(n_clusters=4)
model.fit(Df_normalized)



## Visualiser les données :

colormap= np.array(["Red" , "green","blue","yellow"])


# Compression du dimension par la méthode PCA

from sklearn.decomposition import PCA

models=PCA(n_components=2)
Df_reduce=models.fit_transform(Df_normalized)
plt.scatter(Df_reduce[:,0] ,Df_reduce[:,1] , c= colormap[model.labels_])
plt.show()


## eliminer des anomalies :

### visualsier la 1ere partie rouge




Df_normalized["classe"]= np.nan
Df_normalized["classe"][model.labels_==0] =0
Df_normalized["classe"][model.labels_==1] =1
Df_normalized["classe"][model.labels_==2] =2
Df_normalized["classe"][model.labels_==3] =3



sns.boxplot( x='classe' ,y='HTE' , data=Df_normalized)

sns.displot( Df_normalized , x='HTE' , hue='classe' , kind='kde')

plt.show()
# HTE  , VPM , FIB ,TP

#Après avoir fait le clustering par biologie, on fait l'embedding des médicaments : on va chercher à les vectoriser pour avoir une notion de distance entre eux
#On commence l'embedding basique en vectorisant simplement les ordonnances par individu statistique

tab_libelles_medicament_par_ord = []

ordonnance_act = ""
Df2 = ddf['P_ORD_C_NUM', 'LIBELLE_MEDICAMENT']
Df2['LIBELLE_MEDICAMENT']=Df2['LIBELLE_MEDICAMENT'].apply(iscaracter)
Df2=df
id_ord_act = Df2['P_ORD_C_NUM'][0]
Df2['P_ORD_C_NUM'] = Df2['P_ORD_C_NUM'].apply(isnumber)

for ind in Df2.index:
    if Df2['P_ORD_C_NUM'][ind] == id_ord_act:
        ordonnance_act += str(Df2['LIBELLE_MEDICAMENT'][ind])+ ' '
    else:
        tab_libelles_medicament_par_ord.append(ordonnance_act)
        ordonnance_act = str(Df2['LIBELLE_MEDICAMENT'][ind]) + ' '
        id_ord_act = Df2['P_ORD_C_NUM'][ind]

from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
vectorizer = CountVectorizer()
vector_des_medicaments = vectorizer.fit_transform(tab_libelles_medicament_par_ord)

#On améliore l'embedding en utilisant le tf-idf pour les ordonnances par individu statistique

tfidf_vectorizer = TfidfVectorizer()

vector_tfidf_medicaments = tfidf_vectorizer.fit_transform(tab_libelles_medicament_par_ord)

#On va maintenant utiliser l'embedding sur le libellé commercial d'un médicament
import collections 
contexts_medic = collections.defaultdict(list)
#Le contexte des médicaments est ici les médicaments avec lesquels ils sont présents sur les ordonnances

for ordonnance in tab_libelles_medicament_par_ord:
    medicaments = ordonnance.split()
    for i, medicament in enumerate(medicaments):
        medicaments_candidats = [i+j for j in range(-3, 3) if j != 0]
        medicaments_reels = [candidat for candidat in medicaments_candidats if 0 <= candidat < len(medicaments)]

        contexts_medic[medicament] += [medicaments[a] for a in medicaments_reels ]

#On dispose désormais d'un dictionnaire de médicaments. Pour chaque médicament, on dispose de son contexte, c'est-à-dire les méicaments avec lesquels il est proche dans les ordonnances

#On va maintenant chercher à embedder, c'est-à-dire vectoriser le contexte, pour pouvoir calculer des distances entre les libellés des médicaments

vecteurs_des_medicaments = {}

for medicament, vecteur in dict(contexts_medic).items():
    str_ = ' '.join(vecteur)
    vecteurs_des_medicaments[medicament] = np.around(vectorizer.transform([str_]).toarray(), 2)[0]

#On dispose maintenant d'un vecteur pour chaque libellé commercial de médicaments. Chaque médicament est donc maintenant situé dans un espace vectoriel et on peut calculer des distances entre deux médicaments, notamment pour voir s'il y a des anomalies entre les deux (s'ils sont trop espacés l'un de l'autre par ex)
#Maintena, on va faire la : Représentation graphique des vecteurs de deux médicaments 

#On prend deux médicaments (cas simple ici, uniquement pour voir)
ord0 = tab_libelles_medicament_par_ord[0]
medicaments_ord0 = ord0.split()

medicament_0 = medicaments_ord0[0]

ord1 = tab_libelles_medicament_par_ord[1]
medicaments_ord1 = ord1.split()

medicament_1 = medicaments_ord0[1]
medicament_1 
#On les représente avec matplotlib.plt
vecteurs_des_medicaments[medicament_0].shape

from sklearn.decomposition import PCA

models=PCA(n_components=2)
X=np.concatenate((vecteurs_des_medicaments[medicament_0] , vecteurs_des_medicaments[medicament_1])).reshape(1954,2).T
X_reduce=models.fit_transform(X)

plt.scatter(X_reduce[:,0] ,X_reduce[:,1] )
plt.show()