"""
Метод касательных

Параметры:
f - заданная функция
f_der - производная функции
g - функция из файла [g(x, x0) = f(x0) + f'(x0) * (x - x0)]
a, b - интервал для поиска минимума
x0 - ачальная точка
epsilon - точность
iterations - счетчик итераций
"""
import numpy as np


def f(x):
    return -x ** 3 + 3 * (1 + x) * (np.log(x + 1) - 1)


def f_der(x):
    return -3 * x**2 + 3 * (np.log(x + 1) - 1) + 3 * (1 + x) / (x + 1)


def g(x, y):
    return f(x) + f_der(x) * (y - x)


def tangent_method(a, b, x0, epsilon=10 ** -6):
    x = x0
    iterations = 0
    while True:
        min_y = np.linspace(a, b, 10)  # сетка точек для поиска минимума функции на отрезке [a, b]
        min_value = float(
            'inf')  # начальное значение минимума, взятое очень большим, что бы любое значение функции было меньше его
        new_x = None

        for y in min_y:
            g_value = g(x, y)
            if g_value < min_value:
                min_value = g_value
                new_x = y

        if abs(f(new_x) - g(x, new_x)) < epsilon:
            break

        x = new_x
        iterations += 1

    return x, iterations


a = -0.5
b = 0.5
x0 = 0  # выбрал случайно, можешь поменять сама, комментарий потом удалить
epsilon = 10 ** -4
result, steps = tangent_method(a, b, x0, epsilon)
print(f"Минимум функции: {result}, число итераций: {steps}")