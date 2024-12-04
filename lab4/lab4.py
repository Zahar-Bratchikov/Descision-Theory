import matplotlib.pyplot as plt
import numpy as np

# Целевая функция
def f(x):
    x1, x2 = x
    return x1 ** 2 - 3 * x1 * x2 + 10 * x2 ** 2 + 5 * x1 - 3 * x2

# Ограничения
def constraints(x):
    x1, x2 = x
    # Ограничение: x1 + 2x2 <= 4
    return [x1 + 2 * x2 - 4]  # Ограничение: x1 + 2x2 <= 4

# Штрафная функция
def penalty(x, q=2):
    penalties = [max(0, g) ** q for g in constraints(x)]
    return sum(penalties)

# Вспомогательная функция
def phi_k(x, k):
    return f(x) + k * penalty(x)

# Метод Хука-Дживса для функции φ_k(x)
def hooke_jeeves_phi(phi, x0, lambd=0.1, epsilon=1e-6, alpha=1):
    x = np.array(x0, dtype=float)
    n = len(x)
    step = np.array([lambd] * n)  # Шаг по каждому направлению
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
                if phi(x_test) < phi(x_base):  # Улучшение найдено
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

    return x, phi(x), iteration  # Возвращаем x, значение φ и число итераций

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

    # Отображение плоскости ограничения x1 + 2x2 = 4
    X1_plane, X2_plane = np.meshgrid(np.linspace(x_range[0], x_range[1], 400),
                                     np.linspace(y_range[0], y_range[1], 400))
    Z_plane = 4 - X1_plane - 2 * X2_plane  # Решение уравнения x1 + 2x2 = 4 для плоскости
    ax.plot_surface(X1_plane, X2_plane, Z_plane, color='r', alpha=0.5, label='Ограничение: x1 + 2x2 = 4')

    # Отмечаем точку минимума
    ax.scatter(x_min[0], x_min[1], f(x_min), color="red", label="Минимум", s=100)
    ax.set_title("3D график функции и точка минимума")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("Значение функции")
    ax.legend()

    plt.show()

# Начальная точка
x0 = [5, 5]  # Начальная точка, лежащая за пределами ограничения

# Итеративный процесс метода штрафных функций
k = 1
A_k = 1  # Начальный штрафной коэффициент
epsilon = 1e-6
x_prev = None

while True:
    # Создаём вспомогательную функцию с текущим k
    phi = lambda x: phi_k(x, A_k)

    # Решаем задачу минимизации вспомогательной функции
    x_min, phi_min, iterations = hooke_jeeves_phi(phi, x0)

    # Проверяем условие остановки
    if x_prev is not None and abs(phi_min - phi_k(x_prev, A_k)) < epsilon:
        break

    # Увеличиваем штрафной коэффициент и сохраняем текущую точку
    x_prev = x_min
    A_k *= 10  # Увеличиваем штрафной коэффициент для более строгого учета ограничения

# Вывод результатов
print("Минимум найден в точке:", x_min)
print("Значение функции в минимуме:", f(x_min))
print("Число итераций:", iterations)

# Построение 3D графика с ограничением
plot_function_and_min(f, x_min)
