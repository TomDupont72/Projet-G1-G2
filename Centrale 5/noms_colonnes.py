import pandas as pd
import csv

with open('donnees.txt') as donnees:
    noms_colonnes = donnees.readline()

print(noms_colonnes)