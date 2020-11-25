from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.datasets import SupervisedDataSet
import csv
import matplotlib.pyplot as plt
import sklearn.metrics as mtr
import math
import numpy as num

dataset_treino = csv.reader(open("cardioide-treino.csv", "r"))
dataset_teste = csv.reader(open("cardioide-teste.csv", "r"))

print("Leu os arquivos")
ds = SupervisedDataSet(1, 1)
for row in dataset_treino:
    row = [float(val) for val in row]
    ds.addSample(row[0], row[1])
print("Adicionou os exemplos de treino")

nn = buildNetwork(ds.indim, 15, 5, ds.outdim, bias=True)
print(nn['in'])
print(nn['hidden0'])
print(nn['hidden1'])
print(nn['out'])
treinador = BackpropTrainer(nn, ds)
print("Construiu a rede neural")
for i in range(50):
    print(treinador.train())

print("terminou de treinar")

y_tst = []
y_resultado = []
x = []
for row in dataset_teste:
    row = [float(val) for val in row]
    x.append(row[0])
    elm = nn.activate([row[0]])[0]
    y_resultado.append(elm)
    y_tst.append(row[1])

print("terminou de testar")


def mape(y_tst, y_resultado):
    count = 0
    somatorio = 0
    for i in range(0, len(y_tst)):
        somatorio = somatorio + (num.abs((y_tst[i] - y_resultado[i]) / y_tst[i]))
        count += 1
    return (somatorio / count) * 100


r_quad = mtr.r2_score(y_true=y_tst, y_pred=y_resultado) * 100
r_quad_ajust = (1 - ((1 - mtr.r2_score(y_tst, y_resultado)) * (len(y_resultado) - 1)) / (len(y_tst) - 1 - 1)) *100
MSE = mtr.mean_squared_error(y_tst, y_resultado)
RMSE = math.sqrt(MSE)
MAE = mtr.mean_absolute_error(y_tst, y_resultado)
MAPE = mape(y_tst, y_resultado)
RMSLE = mtr.mean_squared_log_error(y_tst, y_resultado)

print("R-Quadrado = {:.5f}% "
      "\nR-Quadrado Ajustado = {:.5f}%"
      "\nErro Quadrático Médio = {:.5f}"
      "\nRaiz do erro quadrático médio = {:.5f}"
      "\nErro Absoluto Médio = {:.5f}"
      "\nErro Percentual Absoluto Médio = {:.5f}%"
      "\nRaiz do erro médio quadrático e logarítmico = {:.5f}"
      .format(r_quad, r_quad_ajust, MSE, RMSE, MAE, MAPE, RMSLE))

plt.plot(x, y_tst, 'r-')
plt.plot(x, y_resultado, 'b-')
plt.show()
