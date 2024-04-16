import numpy as np
import matplotlib.pyplot as plt
from neuralNetwork import train

params = {}
cost_history = []

x = 10
y = 11

params, cost_history = train(x, y, params, 10000, 0.01)

plt.plot(cost_history)