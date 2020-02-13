import math
import numpy as np
lista = []

lista.append(10)
lista.append(11)
lista.append(15)
x=float('nan')

if(math.isnan(x)):
    print("True")
    lista.append(np.mean(lista))
print(lista)