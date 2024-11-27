import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


# Целевая функция
def f(x):
    x1, x2 = x
    return x1**2 - 3 * x1 * x2 + 10 * x2**2 + 5 * x1 - 3 * x2

# Градиент целевой функции
def grad_f(x):
    x1, x2 = x
    df_dx1 = 2 * x1 - 3 * x2 + 5
    df_dx2 = -3 * x1 + 20 * x2 - 3
    return np.array([df_dx1, df_dx2])

# Метод наискорейшего спуска с оптимальным шагом
def steepest_descent(f, grad_f, x0, epsilon=1e-4):
    x = np.array(x0, dtype=float)
    iter_count = 0
    trajectory = [x.copy()]  # Для сохранения траектории точек

    while np.linalg.norm(grad_f(x)) > epsilon:
        iter_count += 1

        # Определение функции φk(λ) = f(x - λ * grad_f(x)) для минимизации по λ
        def phi(lmbda):
            return f(x - lmbda * grad_f(x))

        # Поиск оптимального λ с уменьшенной точностью (tol) для ускорения с помощью встроенного метода золотогго сечения библиотеки SciPy
        res = minimize_scalar(phi, bounds=(0, 1), method='bounded', options={'xatol': 1e-3})
        lmbda_opt = res.x  # Оптимальный шаг

        # Обновление точки
        x = x - lmbda_opt * grad_f(x)
        trajectory.append(x.copy())  # Сохраняем точку траектории

    return x, f(x), iter_count, trajectory

# Функция для отрисовки 3D графика
def plot_3d_trajectory(f, trajectory, x_range=(-5, 5), y_range=(-5, 5)):
    # Сетка точек
    x1 = np.linspace(x_range[0], x_range[1], 400)
    x2 = np.linspace(y_range[0], y_range[1], 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f([X1, X2])

    # Создаем фигуру и 3D-ось
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Отрисовка 3D поверхности функции
    ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor='none', alpha=0.7)

    # Отображение траектории спуска
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], [f(point) for point in trajectory], color="red", marker="o", label="Траектория спуска")
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], f(trajectory[-1]), color="blue", s=100, label="Минимум")

    # Настройки графика
    ax.set_title("3D график функции и траектория наискорейшего спуска")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("Значение функции")
    ax.legend()

    plt.show()

# Начальная точка
x0 = [2, 1]

# Запуск метода наискорейшего спуска
min_point, min_value, iterations, trajectory = steepest_descent(f, grad_f, x0)

# Вывод результатов
print("Минимум найден в точке:", min_point)
print("Значение функции в минимуме:", min_value)
print("Число итераций:", iterations)

# Построение 3D графика
plot_3d_trajectory(f, trajectory)
