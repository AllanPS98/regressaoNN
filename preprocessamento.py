import math
import matplotlib.pyplot as plt
import csv

x = -1
a = 2.5
list_x = []
list_y = []
for i in range(1000):
    list_x.append(x)
    y = math.sqrt(2 * math.sqrt(math.pow(a, 3) * (a + (2 * x))) + (2 * math.pow(a, 2)) + (2 * a * x) - math.pow(x, 2))
    list_y.append(y)
    x += 0.01

plt.plot(list_x, list_y)
plt.show()


def split_dataset(list_x, list_y):
    list_treino = []
    list_teste = []
    count = 0
    for i in range(1000):
        elm_x = list_x.__getitem__(i)
        elm_y = list_y.__getitem__(i)
        if count % 3 == 0:
            list_teste.append([elm_x, elm_y])
        else:
            list_treino.append([elm_x, elm_y])
        count += 1
    return list_treino, list_teste


treino, teste = split_dataset(list_x, list_y)

y_treino = []
y_teste = []
x_tr = []
x_tst = []
for e in treino:
    y_treino.append(e[1])
    x_tr.append(e[0])
for e in teste:
    y_teste.append(e[1])
    x_tst.append(e[0])
plt.plot(x_tr, y_treino, 'r-')
plt.title('Treino')
plt.show()
plt.plot(x_tst, y_teste, 'b-')
plt.title('Teste')
plt.show()


escrita_treino = csv.writer(open('cardioide-treino.csv', 'w', newline=''))
escrita_teste = csv.writer(open('cardioide-teste.csv', 'w', newline=''))

for elm in treino:
    escrita_treino.writerow(elm)

for elm in teste:
    escrita_teste.writerow(elm)


