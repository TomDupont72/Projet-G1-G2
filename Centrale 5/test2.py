import pandas as pd
import csv



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

