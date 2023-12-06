import pandas as pd
import csv



nbsep = 100
#linea
#lineb
nblineprec = 1
i = 0

donnees = open("donnees.txt", "a")
donnees.close()

with open("donnees.txt","r") as donnees:
    for line in donnees: 
        #if line.count(';') != nbsep:
        print(i)
        i = i + 1
            

