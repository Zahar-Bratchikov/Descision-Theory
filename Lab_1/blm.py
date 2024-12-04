import numpy as np


# Функция
def f(x):
    return -x ** 3 + 3 * (1 + x) * (np.log(x + 1) - 1)


# Вычисление константы Липшица
def compute_lipschitz_constant(a, b):
    x_values = np.linspace(a, b, 10000)
    derivatives = np.abs(-3 * x_values ** 2 + 3 * (np.log(x_values + 1) - 1) + 3 * (1 + x_values) / (x_values + 1))
    return np.max(derivatives)


# Функция g(x, x_i)
def g(x, x_i, L):
    return f(x_i) - L * abs(x - x_i)


# Метод золотого сечения для минимизации
def golden_section_search(func, a, b, tol=1e-8):
    phi = (1 + np.sqrt(5)) / 2  # Золотое сечение
    inv_phi = 1 / phi
    x1 = b - inv_phi * (b - a)
    x2 = a + inv_phi * (b - a)
    f1, f2 = func(x1), func(x2)

    while abs(b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - inv_phi * (b - a)
            f1 = func(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + inv_phi * (b - a)
            f2 = func(x2)

    return (a + b) / 2


# Метод ломанных
def broken_line_method(a, b, x0, epsilon):
    # Вычисляем константу Липшица
    L = compute_lipschitz_constant(a, b)
    x_points = [x0]  # Начальная точка
    iterations = 0

    while True:
        # Шаг 2: Строим функцию p_k(x)
        def p_k(x):
            return max(g(x, x_i, L) for x_i in x_points)

        # Шаг 3: Минимизируем p_k(x) на интервале [a, b] с помощью золотого сечения
        x_next = golden_section_search(p_k, a, b)

        # Условие останова
        if len(x_points) > 1 and abs(p_k(x_next) - p_k(x_points[-1])) < epsilon:
            break

        # Обновляем список точек
        x_points.append(x_next)
        iterations += 1

    # Возвращаем точку минимума, значение функции в этой точке и количество итераций
    return x_next, f(x_next), iterations


# Параметры
a = -0.5
b = 0.5
x0 = 0.0  # Начальная точка
epsilon = 1e-6  # Точность

# Вычисляем минимум
result, min_value, steps = broken_line_method(a, b, x0, epsilon)
print(f"Минимум функции: {min_value}, точка минимума: {result}, число итераций: {steps}")
