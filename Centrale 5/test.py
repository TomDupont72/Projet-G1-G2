import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from   sklearn.preprocessing import LabelEncoder
## Analyse statistique :
df = pd.read_csv('G:\Data\prevention_iatrogenie_centrale_5\securimed.csv',encoding='ISO-8859-1',sep=';',error_bad_lines=False , skiprows=[i for i in range(300000,2900000)] , usecols=['VOIE_ADMIN','TYPE_FREQUENCE','MOMENT','CONDITION','AGE' , 'POIDS' , 'TAILLE','QTE_PRESC' ,'BILIT', 'PAL', 'TGO', 'TGP', 'LEUCC','ERYTH', 'HHB', 'HTE', 'VGM', 'TCMH', 'CCMH', 'NPLAQ', 'VPM', 'TCAM','TCAT', 'RTCAMT', 'TQM',  'TP', 'FIB'])


df=df.drop(['P_IPP','LIBELLE_MEDICAMENT',  'UNITE_PRESC',  'HEURE_PRISE', 'MIN_PRISE',  'NON_QUOTIDIEN','UNITE_PRESC','DATE_INTERVENTION','P_ORD_C_NUM', 'P_LPM_C_NUM', 'P_LCP_C_NUM', 'P_LPO_C_NUM', 'DATE_DEBUT', 'DATE_FIN', 'CODE_UCD', 'CODE_ATC', 'NOM_COMMERCIAL', 'DCI','COMMENTAIRE_POSO','POIDS_DATE','TAILLE_DATE','MDRD_DATE','CKDEPI_DATE', 'POIDS_UNITE','TAILLE_UNITE','MDRD_UNITE', 'MDRD_DATE','MDRD_UNITE', 'MDRD_DATE','CKDEPI_UNITE', 'CKDEPI_DATE','BILIT_UNITE', 'BILIT_DATE','PAL_UNITE', 'PAL_DATE','TGO_UNITE', 'TGO_DATE', 'TGP_UNITE', 'TGP_DATE','LEUCC_UNITE', 'LEUCC_DATE', 'ERYTH_UNITE', 'ERYTH_DATE', 'HHB_UNITE', 'HHB_DATE', 'HTE_UNITE', 'HTE_DATE', 'VGM_UNITE', 'VGM_DATE',  'TCMH_UNITE', 'TCMH_DATE', 'CCMH_UNITE', 'CCMH_DATE', 'NPLAQ_UNITE', 'NPLAQ_DATE', 'VPM_UNITE', 'VPM_DATE','TCAM_UNITE', 'TCAM_DATE', 'TCAT_UNITE', 'TCAT_DATE',  'RTCAMT_UNITE', 'RTCAMT_DATE',  'TQM_UNITE', 'TQM_DATE', 'TQT_UNITE', 'TQT_DATE',  'RTQMT_UNITE', 'RTQMT_DATE',  'TP_UNITE', 'TP_DATE', 'FIB_UNITE', 'FIB_DATE'],axis=1)

col = df.columns 




def iscaracter(s):
    try:

        a=str(s).replace(',','.').replace('/','').replace(':','').replace('.','').replace(' ','')
        float(a)

        return np.nan

    except ValueError :
        
        return s

def isnumber(s):
    try:
        a=float(str(s).replace(',','.'))

        return a
    except ValueError :

        return np.nan



##  netoyage des donner et filtration des valeurs abérantes

1

## éliminer les outliers par la méthode "quantile"


for l in ['AGE' , 'POIDS' , 'TAILLE','QTE_PRESC' ,'BILIT', 'PAL', 'TGO', 'TGP', 'LEUCC','ERYTH', 'HHB', 'HTE', 'VGM', 'TCMH', 'CCMH', 'NPLAQ', 'VPM', 'TCAM','TCAT', 'RTCAMT', 'TQM',  'TP', 'FIB'] :
    df[l]=df[l].apply(isnumber)
    upper_lim = df[l].quantile(.95)
    under_lim=df[l].quantile(.05)
    df[l][~(df[l]<under_lim)&(df[l]>upper_lim)]=np.nan

1
##  remplacment des valeurs manquantes 
n=df.shape[0]
for l in ['AGE' , 'POIDS' , 'TAILLE','QTE_PRESC','BILIT', 'PAL', 'TGO', 'TGP', 'LEUCC','ERYTH', 'HHB', 'HTE', 'VGM', 'TCMH', 'CCMH', 'NPLAQ', 'VPM', 'TCAM','TCAT', 'RTCAMT', 'TQM',  'TP', 'FIB'] :
    normale=np.abs(np.random.normal(df[l].mean(),df[l].std(),n))
    index_with_nan=df[l].index[df[l].isnull()]
    df[l].iloc[index_with_nan] = normale[index_with_nan]
## garder que les caractéres
for l in ['VOIE_ADMIN','TYPE_FREQUENCE','MOMENT','CONDITION'] :
        df[l]=df[l].apply(iscaracter)
        

## suprission des lignes qui contient des valeurs manquantes 
#Supprimer les lignes avec des données manquantes 
1
for l in  ['VOIE_ADMIN','TYPE_FREQUENCE','MOMENT','CONDITION'] :
    le = LabelEncoder()
    df[l].fillna("None" , inplace = True)
    le.fit(df[l])
    df[l]=le.transform(df[l])

1
sns.displot(df , x='VOIE_ADMIN' , kind='kde')
sns.displot(df , x='CONDITION' , kind='kde')

plt.show()
## l'application du Anomalie detection pour eliminer les prescrit non usuelle en fonction du contexte de prescription 

data=df['QTE_PRESC']

from scipy.cluster.vq import kmeans , vq
colormap= np.array(["Red" , "green","blue"])

X=data.to_numpy()
cluster , _ = kmeans(X , 2)
cluster_indices , _ =vq(X , cluster)
X=X.reshape(-1,1)

df['classe']=cluster_indices

sns.boxplot( x='classe' ,y='QTE_PRESC' , data=df)
sns.displot(D, x='FIB' , hue='classe' ,kind='kde' )
plt.show()
col
## échantilloner les données :
indice1=df['classe'][df['classe']==1].index
indice0=df['classe'][df['classe']==0].index     
D=pd.concat([df.iloc[indice1],df.iloc[indice0[:10283]]])
D

from sklearn.model_selection import train_test_split
from sklearn import datasets , svm , metrics
from sklearn.ensemble import RandomForestRegressor as model
from sklearn.metrics import r2_score
from sklearn import preprocessing
## Nomalisation :
min_max_scaler=preprocessing.MinMaxScaler()
np_scaler= min_max_scaler.fit_transform(df)
Df_normalized=pd.DataFrame(np_scaler,columns=col)
X=Df_normalized[['VOIE_ADMIN', 'MOMENT' ,'AGE', 'POIDS', 'TAILLE','FIB' ,'TCAT','VPM','HHB','RTCAMT','CONDITION']]
y=Df_normalized['QTE_PRESC']

X_train ,X_test, y_train , y_test = train_test_split(X , y , test_size=0.2 ,random_state=42)

clf=model()
clf.fit(X_train , y_train)
y_pred=clf.predict(X_test)
r2_score(y_test , y_pred)


