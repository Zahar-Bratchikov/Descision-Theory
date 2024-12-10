import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


# Целевая функция
def f(x):
    return x[0] ** 2 + 3 * x[1] ** 2 + np.cos(x[0] + x[1])


# Новое ограничение
def g(x):
    return x[0] - x[1] + 1  # Новая плоскость ограничения


# Штрафная функция
def penalty(x, q=2):
    return max(g(x), 0) ** q


# Функция с учетом штрафной функции
def phi(x, penalty_coeff):
    return f(x) + penalty_coeff * penalty(x)


# Метод наискорейшего спуска
def steepest_descent_with_penalty(phi, grad_phi, x0, penalty_coeff, tol=1e-6):
    """Метод наискорейшего спуска для функции с штрафной функцией."""
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    phi_prev = phi(x, penalty_coeff)  # Предыдущее значение функции
    n_iter = 0  # Счетчик итераций

    while True:
        grad = grad_phi(x, penalty_coeff)  # Градиент функции
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:  # Условие остановки по норме градиента
            break

        # Найти оптимальный шаг (линейный поиск)
        def phi_lambda(lmbd):
            return phi(x - lmbd * grad, penalty_coeff)

        res = minimize_scalar(phi_lambda, bounds=(0, 1), method='bounded')
        if not res.success:
            break

        lambda_k = res.x
        x = x - lambda_k * grad  # Обновление x
        trajectory.append(x.copy())

        # Проверка изменения значения функции
        phi_current = phi(x, penalty_coeff)
        if abs(phi_current - phi_prev) < tol:  # Условие остановки
            break

        phi_prev = phi_current  # Обновление предыдущего значения функции
        n_iter += 1  # Увеличиваем счетчик итераций

    return x, phi(x, penalty_coeff), n_iter, trajectory


# Метод штрафных функций
def penalty_method(f, g, x0, initial_penalty_coeff=1, tol=1e-6):
    """Метод штрафных функций."""
    penalty_coeff = initial_penalty_coeff
    x_k = np.array(x0)
    trajectory = []
    phi_prev = float("inf")  # Для проверки сходимости
    total_iterations = 0  # Счетчик общего числа итераций

    while True:
        # Градиент функции
        def grad_phi(x, penalty_coeff):
            grad_f = np.array([2 * x[0] - np.sin(x[0] + x[1]),
                               6 * x[1] - np.sin(x[0] + x[1])])
            grad_g = np.array([1, -1])  # Градиент нового ограничения
            if g(x) > 0:
                grad_penalty = penalty_coeff * 2 * g(x) * grad_g
            else:
                grad_penalty = np.array([0, 0])
            return grad_f + grad_penalty

        # Используем метод наискорейшего спуска
        x_k, phi_k, sub_iterations, sub_trajectory = steepest_descent_with_penalty(
            lambda x, _: f(x) + penalty_coeff * penalty(x),  # Обертка для phi
            grad_phi,
            x_k,
            penalty_coeff,
            tol=tol
        )
        trajectory.extend(sub_trajectory)
        total_iterations += sub_iterations  # Увеличиваем общее число итераций

        # Условие остановки
        if abs(phi_k - phi_prev) < tol:  # Проверка изменения значения функции
            break

        phi_prev = phi_k
        penalty_coeff *= 10  # Увеличиваем штрафной коэффициент

    return x_k, f(x_k), penalty_coeff, trajectory, total_iterations


# График функции и траектории
def plot_function_with_constraint(f, g, trajectory, x_min, x_range=(-1, 2), y_range=(-1, 2)):
    # Сетка точек
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f([X, Y])

    # Значения ограничения
    G = g([X, Y])

    # Построение графика
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

    # Ограничение
    constraint_plane = np.maximum(0, G) + Z.min()
    ax.plot_surface(X, Y, constraint_plane, color='yellow', alpha=0.3)

    # Траектория оптимизации
    trajectory = np.array(trajectory)
    Z_traj = np.array([f(point) for point in trajectory])
    ax.plot(trajectory[:, 0], trajectory[:, 1], Z_traj, color='red', marker='o', label='Траектория', lw=2)

    # Минимум
    ax.scatter(x_min[0], x_min[1], f(x_min), color='blue', s=100, label='Минимум', zorder=5)

    ax.set_title('3D график функции с новым ограничением и траекторией')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x)$')
    ax.legend()
    plt.show()


# Запуск метода штрафных функций с новым ограничением
x0 = [0.5, 0.5]
x_min, f_min, final_penalty_coeff, trajectory, total_iterations = penalty_method(f, g, x0)

# Вывод результатов
print("Минимум найден в точке:", x_min)
print("Значение функции в минимуме:", f_min)
print("Штрафной коэффициент:", final_penalty_coeff)
print("Общее количество итераций:", total_iterations)

# Построение графика
plot_function_with_constraint(f, g, trajectory, x_min)
