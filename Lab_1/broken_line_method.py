import numpy as np


# Функция и её производная
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

        # Шаг 3: Минимизируем p_k(x) на интервале [a, b]
        x_candidates = np.linspace(a, b, 1000)
        min_x = None
        min_value = float('inf')

        for x in x_candidates:
            p_value = p_k(x)
            if p_value < min_value:
                min_value = p_value
                min_x = x

        # Условие останова
        if len(x_points) > 1 and abs(p_k(min_x) - p_k(x_points[-1])) < epsilon:
            break

        # Обновляем список точек
        x_points.append(min_x)
        iterations += 1

    # Возвращаем точку минимума, значение функции в этой точке и количество итераций
    return min_x, f(min_x), iterations


# Параметры
a = -0.5
b = 0.5
x0 = 0.0  # Начальная точка
epsilon = 1e-6  # Точность

# Вычисляем минимум
result, min_value, steps = broken_line_method(a, b, x0, epsilon)
print(f"Минимум функции: {min_value}, точка минимума: {result}, число итераций: {steps}")
