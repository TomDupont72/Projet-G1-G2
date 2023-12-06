import pandas as pd

file = r'echantillon.txt'
df = pd.read_csv(file, sep=';')

df.describe()
print(df)

#file2 = r'securimed.csv'
#print(open(file2).encoding)
#df2 = pd.read_csv(file2, sep=';', encoding=open(file2).encoding)


"""
for i in range(100):
    print("Hello")

file = r'echantillon.txt'
df = pd.read_csv(file, sep=';')
"""
"""

df.describe()
print(df)

file2 = r'securimed.csv'
#print(open(file2).encoding)
#df2 = pd.read_csv(file2, sep=';', encoding=open(file2).encoding)
"""
print("!!!")


lines = [0, 263, 264, 265, 266, 267, 268, 269, 96419, 96420]
data = []
i = 0

with open("securimed.csv", "r") as tf:
    for line in tf:
        if i in lines:
            data.append(line)
        i = i + 1
print(data[0].count(';'))
print(data[1])
print(data[1].count(';'))
print(data[2])
print(data[2].count(';'))
print(data[3])
print(data[3].count(';'))
print(data[4])
print(data[4].count(';'))
print(data[5])
print(data[5].count(';'))
print(data[6])
print(data[6].count(';'))
print(data[7])
print(data[7].count(';'))
