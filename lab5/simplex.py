import numpy as np

def simplex_method(c, A, b):
    """
    Симплекс-метод для задачи максимизации с отладочными выводами.
    """
    # Приведение к канонической форме
    m, n = A.shape
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:-1, :-1] = np.hstack((A, np.eye(m)))  # Добавляем базисные переменные
    tableau[:-1, -1] = b  # Правая часть ограничений
    tableau[-1, :-1] = np.hstack((-c, np.zeros(m)))  # Коэффициенты целевой функции
    basis = list(range(n, n + m))  # Индексы базисных переменных

    print("Начальная симплекс-таблица:")
    print(tableau)

    while True:
        # Шаг 4: Проверка оптимальности
        if np.all(tableau[-1, :-1] <= 0):  # Все оценки σn ≤ 0
            print("Решение оптимально.")
            break

        # Шаг 5: Выбор разрешающего столбца
        pivot_col = np.argmax(tableau[-1, :-1])  # Максимальный элемент в последней строке
        if tableau[-1, pivot_col] <= 0:
            print("Решение оптимально.")
            break

        # Шаг 6: Проверка неограниченности
        if np.all(tableau[:-1, pivot_col] <= 0):
            raise Exception("Целевая функция неограничена.")

        # Шаг 7: Выбор разрешающей строки
        ratios = [
            tableau[i, -1] / tableau[i, pivot_col]
            if tableau[i, pivot_col] > 0 else np.inf
            for i in range(m)
        ]
        pivot_row = np.argmin(ratios)

        print(f"Разрешающий элемент: строка {pivot_row}, столбец {pivot_col}")
        print(f"До пересчета таблицы:")
        print(tableau)

        # Шаг 8: Пересчет таблицы
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        print(f"После пересчета таблицы:")
        print(tableau)

        # Обновление базиса
        basis[pivot_row] = pivot_col

    # Извлечение результата
    solution = np.zeros(n)
    for i, col in enumerate(basis):
        if col < n:
            solution[col] = tableau[i, -1]

    return solution, -tableau[-1, -1]  # Максимальная целевая функция

# Исходные данные
c = np.array([2, 2], dtype=float)  # Коэффициенты целевой функции
A = np.array([
    [-3, 2],  # -3x1 + 2x2 <= 6
    [-1, -1],  # -x1 - x2 <= -3
    [1, 0],  # x1 <= 3
    [0, 1]  # x2 <= 5
], dtype=float)
b = np.array([6, -3, 3, 5], dtype=float)  # Правая часть ограничений

# Приведение ограничений к стандартной форме (все b >= 0)
for i in range(len(b)):
    if b[i] < 0:
        A[i, :] *= -1
        b[i] *= -1

print("Матрица ограничений (A):")
print(A)
print("Вектор ограничений (b):")
print(b)

# Решение
solution, max_value = simplex_method(c, A, b)

# Вывод результатов
print("\nОптимальное решение:")
print(f"x1 = {solution[0]:.2f}, x2 = {solution[1]:.2f}")
print(f"Максимальное значение целевой функции: F = {max_value:.2f}")
