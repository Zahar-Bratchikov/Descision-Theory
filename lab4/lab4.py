import matplotlib.pyplot as plt
import numpy as np

# Целевая функция
def f(x):
    x1, x2 = x
    return x1**2 - 3 * x1 * x2 + 10 * x2**2 + 5 * x1 - 3 * x2

# Ограничение
def g(x):
    x1, x2 = x
    return (1 / 2) * x1 - (1 / 3) * x2 + 2  # Плоскость ограничения

# Штрафная функция
def penalty(x, q=2):
    return max(g(x), 0) ** q

# Вспомогательная функция с учетом штрафа
def phi(x, A_k):
    return f(x) + A_k * penalty(x)

# Метод Хука-Дживса с использованием вспомогательной функции
def hooke_jeeves_penalty(phi, x0, lambd=0.5, epsilon=1e-6, alpha=1):
    x = np.array(x0, dtype=float)
    n = len(x)
    step = np.array([lambd] * n)  # Шаг по каждому направлению
    trajectory = [x.copy()]  # Сохраняем путь оптимизации
    iterations = 0  # Счетчик итераций

    while max(step) > epsilon:
        iterations += 1

        # 1. Исследующий поиск
        x_base = x.copy()
        for i in range(n):
            for direction in [+1, -1]:  # Два направления
                x_test = x_base.copy()
                x_test[i] += direction * step[i]
                if phi(x_test) < phi(x_base):  # Улучшение найдено
                    x_base = x_test

        # 2. Шаг приближения
        if np.any(x_base != x):
            x = x + alpha * (x_base - x)  # Шаг к улучшению
        else:
            # 3. Уменьшение шага
            step *= 0.5

        trajectory.append(x.copy())  # Сохраняем траекторию

    return x, phi(x), iterations, trajectory

# Основной метод штрафных функций
def penalty_method(f, g, x0, initial_A_k=1, epsilon=1e-6):
    x_k = x0
    A_k = initial_A_k
    trajectory = []
    phi_k_prev = float('inf')  # Для проверки условия остановки
    total_iterations = 0  # Общее количество итераций

    while True:
        # Оборачиваем phi в функцию, фиксируя текущий A_k
        phi_with_penalty = lambda x: phi(x, A_k)
        x_k, phi_k, iterations, path = hooke_jeeves_penalty(phi_with_penalty, x_k, epsilon=epsilon)
        trajectory.extend(path)
        total_iterations += iterations  # Суммируем внутренние итерации

        # Условие остановки
        if abs(phi_k - phi_k_prev) < epsilon:
            break

        phi_k_prev = phi_k
        A_k *= 10  # Увеличиваем штрафной коэффициент

    return x_k, f(x_k), A_k, trajectory, total_iterations

# Функция для отрисовки 3D графика с траекторией и ограничением
def plot_function_with_constraint(f, g, x_min, trajectory, x_range=(-5, 5), y_range=(-5, 5)):
    # Сетка точек
    x1 = np.linspace(x_range[0], x_range[1], 400)
    x2 = np.linspace(y_range[0], y_range[1], 400)
    X1, X2 = np.meshgrid(x1, x2)

    # Вычисляем значения функции
    Z = f([X1, X2])

    # Значения плоскости ограничения
    constraint_Z = g([X1, X2])

    # Отрисовка 3D поверхности
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor='none', alpha=0.8, label="f(x)")

    # Плоскость ограничения
    ax.plot_surface(X1, X2, constraint_Z, color="orange", alpha=0.5, label="Ограничение")

    # Траектория оптимизации
    trajectory = np.array(trajectory)
    z_values = np.array([f(point) for point in trajectory])
    ax.plot(trajectory[:, 0], trajectory[:, 1], z_values, color="red", marker='o', label="Траектория")

    # Точка минимума
    ax.scatter(x_min[0], x_min[1], f(x_min), color="blue", s=100, label="Минимум")

    ax.set_title("3D график функции с ограничением и траекторией")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")
    ax.legend()
    plt.show()

# Начальная точка
x0 = [2, -3]  # Начальная точка

# Запуск метода
min_point, min_value, final_A_k, trajectory, total_iterations = penalty_method(f, g, x0)

# Вывод результатов
print("Минимум найден в точке:", min_point)
print("Значение функции в минимуме:", min_value)
print("Штрафной коэффициент A_k:", final_A_k)
print("Общее количество итераций:", total_iterations)

# Построение 3D графика
plot_function_with_constraint(f, g, min_point, trajectory)
