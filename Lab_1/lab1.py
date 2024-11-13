import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Задаем параметры точности и интервала поиска
tolerance = 1e-6
delta = 1e-7
a = 0.5
b = 1

# Определяем функцию f(x) и ее производные
def f(x):
    return x ** 2 - 2 * x - 2 * math.cos(x)

def f_prime(x):
    return 2 * x - 2 + 2 * math.sin(x)

def f_second_derivative(x):
    return 2 + 2 * math.cos(x)

# Метод половинного деления
def bisection_method(a, b, tolerance):
    iteration = 1
    while abs(b - a) >= tolerance:
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2
        if f(x1) >= f(x2):
            a = x1
        else:
            b = x2
        iteration += 1
    x_min = (a + b) / 2
    return x_min, f(x_min), iteration

# Метод секущих
def secant_method(f_prime, x0, x1, epsilon=1e-6, max_iter=100):
    iteration = 1
    for _ in range(max_iter):
        iteration += 1
        f_prime_x0 = f_prime(x0)
        f_prime_x1 = f_prime(x1)
        if abs(f_prime_x1 - f_prime_x0) < epsilon:
            return x1, f(x1), iteration
        x2 = x1 - f_prime_x1 * (x1 - x0) / (f_prime_x1 - f_prime_x0)
        if abs(x2 - x1) < epsilon:
            return x2, f(x2), iteration
        x0, x1 = x1, x2
    return x1, f(x1), iteration

# Метод Ньютона
def newton_method(a, tolerance):
    x_k = a
    x_k_n = x_k - (f_prime(x_k) / f_second_derivative(x_k))
    iterations_x = [x_k]
    iteration = 1

    while abs(f(x_k_n) - f(x_k)) >= tolerance and abs((2 * x_k_n - 2 + 2 * math.sin(x_k_n)) - (2 * x_k - 2 + 2 * math.sin(x_k))) >= tolerance:
        x_k = x_k_n
        x_k_n = x_k - (f_prime(x_k) / f_second_derivative(x_k))
        iterations_x.append(x_k_n)
        iteration += 1

    return x_k_n, f(x_k_n), iteration

# Запуск всех методов
x_bisect, f_bisect, iter_bisect = bisection_method(a, b, tolerance)

# Начальные точки для метода секущих
x0 = np.random.uniform(a, b)
x1 = np.random.uniform(a, b)
x_secant, f_secant, iter_secant = secant_method(f_prime, x0, x1)

x_newton, f_newton, iter_newton = newton_method(a, tolerance)

# Вывод результатов
print(f"Метод половинного деления: x_min = {x_bisect}, f(x_min) = {f_bisect}, итерации = {iter_bisect}, f'(x_min) = {f_prime(x_bisect)}")
print(f"Метод секущих: x_min = {x_secant}, f(x_min) = {f_secant}, итерации = {iter_secant}, f'(x_min) = {f_prime(x_secant)}")
print(f"Метод Ньютона: x_min = {x_newton}, f(x_min) = {f_newton}, итерации = {iter_newton}, f'(x_min) = {f_prime(x_newton)}")

# Построение одного графика с тремя точками минимума
x_vals = np.linspace(a, b, 4000)
y_vals = [f(x) for x in x_vals]

plt.plot(x_vals, y_vals, label="f(x)", color="blue")

# Отметка минимумов, найденных разными методами
plt.scatter(x_bisect, f_bisect, color="red", label=f"Половинное деление (x={x_bisect:.6f}, f(x)={f_bisect:.6f})")
plt.scatter(x_secant, f_secant, color="green", label=f"Секущие (x={x_secant:.6f}, f(x)={f_secant:.6f})")
plt.scatter(x_newton, f_newton, color="orange", label=f"Ньютон (x={x_newton:.6f}, f(x)={f_newton:.6f})")

# Оформление графика
plt.title("Поиск минимума разными методами")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
