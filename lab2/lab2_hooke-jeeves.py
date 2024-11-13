import matplotlib.pyplot as plt
import numpy as np


# Целевая функция
def f(x):
    x1, x2 = x
    return x1 ** 2 - 3 * x1 * x2 + 10 * x2 ** 2 + 5 * x1 - 3 * x2


# Метод Хука-Дживса с оптимизациями
def hooke_jeeves_optimized(f, x0, delta=0.01, epsilon=1e-6, alpha=1):
    x = np.array(x0, dtype=float)
    n = len(x)
    step = np.array([delta] * n)  # Шаг по каждому направлению
    iteration = 0  # Счетчик итераций

    while max(step) > epsilon:
        iteration += 1  # Увеличиваем счетчик итераций

        # 1. Исследующий поиск
        x_base = x.copy()
        for i in range(n):
            improved = False  # Флаг улучшения
            for direction in [+1, -1]:  # Два направления: положительное и отрицательное
                x_test = x_base.copy()
                x_test[i] += direction * step[i]
                if f(x_test) < f(x_base):  # Улучшение найдено
                    x_base = x_test
                    improved = True
                    break  # Прерываем дальнейший поиск в этом направлении
            if improved:
                break  # Прерываем перебор направлений, если найдено улучшение

        # 2. Шаг приближения
        if np.any(x_base != x):
            x = x + alpha * (x_base - x)  # Делаем шаг в направлении улучшения
        else:
            # 3. Уменьшение шага
            step *= 0.5

    return x, f(x), iteration  # Возвращаем количество итераций


# Функция для отрисовки 3D графика
def plot_function_and_min(f, x_min, x_range=(-5, 5), y_range=(-5, 5)):
    # Сетка точек
    x1 = np.linspace(x_range[0], x_range[1], 400)
    x2 = np.linspace(y_range[0], y_range[1], 400)
    X1, X2 = np.meshgrid(x1, x2)

    # Вычисляем значения функции на сетке
    Z = f([X1, X2])

    # Отрисовка 3D поверхности
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor='none')

    # Отмечаем точку минимума
    ax.scatter(x_min[0], x_min[1], f(x_min), color="red", label="Минимум", s=100)
    ax.set_title("3D график функции и точка минимума")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("Значение функции")
    ax.legend()

    plt.show()


# Начальная точка
x0 = [2, 1]

# Запуск метода
min_point, min_value, iterations = hooke_jeeves_optimized(f, x0)

# Вывод результатов
print("Минимум найден в точке:", min_point)
print("Значение функции в минимуме:", min_value)
print("Число итераций:", iterations)  # Выводим количество итераций

# Построение 3D графика
plot_function_and_min(f, min_point)
