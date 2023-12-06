import pandas as pd
import csv
import matplotlib.pyplot as plt

file2 = 'donnees_01.txt'
df2 = pd.read_csv(file2, sep=';', encoding=open(file2).encoding)

df2.head()