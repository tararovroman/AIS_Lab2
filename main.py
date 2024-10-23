import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Генерация данных
x1 = np.linspace(0, 10, 500)
x2 = np.linspace(5, 15, 500)
y = x1**6 + x2**2 + x1**3 + 4*x2 + 5

# Запись данных в graph.csv
df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
df.to_csv('graph.csv', index=False)

# Используем meshgrid для 3D графика
X1, X2 = np.meshgrid(x1, x2)
Y = X1**6 + X2**2 + X1**3 + 4*X2 + 5

# Создаем окно с несколькими подграфиками
fig = plt.figure(figsize=(15, 10))

# Первый 2D график: y(x1) при фиксированном x2
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x1, y, label='y(x1) при фиксированном x2')
ax1.scatter(x1, y, color='red')
ax1.set_title('График y от x1 при фиксированном x2')
ax1.set_xlabel('x1')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True)

# Второй 2D график: y(x2) при фиксированном x1
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x2, y, label='y(x2) при фиксированном x1')
ax2.scatter(x2, y, color='blue')
ax2.set_title('График y от x2 при фиксированном x1')
ax2.set_xlabel('x2')
ax2.set_ylabel('y')
ax2.legend()
ax2.grid(True)

# 3D график
ax3 = fig.add_subplot(2, 1, 2, projection='3d')
ax3.plot_surface(X1, X2, Y, cmap='viridis')
ax3.set_title('3D график функции y = x1^6 + x2^2 + x1^3 + 4x2 + 5')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')

# Отображение всех графиков в одном окне
plt.tight_layout()
plt.show()

