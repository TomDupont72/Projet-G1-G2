## Correction des erreurs de saut à la ligne

nbsep = 100
nbline = 0
nblineprec = 0

with open("donnees.txt","a") as donnees:
    with open("securimed.csv", "r") as tf:
        for line in tf: 
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


## Partage du fichier de données en 3 fichiers pour éviter les problèmes de mémoire

i = 0

with open("donnees.txt","r") as donnees:
    part1 = open("donnees_01.txt", "a")
    part2 = open("donnees_02.txt", "a")
    part3 = open("donnees_03.txt", "a")
    for line in donnees:
        x = line.replace(",",".") # On remplace les ',' par des '.' pour reconnaître des valeurs numériques par la suite
        if i <= 1000000:
            part1.write(x)
        elif 1000001 <= i and i <= 2000000:
            part2.write(x)
        else:
            part3.write(x)
        i = i + 1
    part1.close()
    part2.close()  
    part3.close()
