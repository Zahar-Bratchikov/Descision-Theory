import numpy as np

# Функция для поиска решения симплекс-методом с выводом промежуточных таблиц
def simplex_method(c, A, b):
    m, n = A.shape

    # Создаем расширенную симплекс-таблицу
    table = np.zeros((m + 1, n + m + 1))
    table[:-1, :-1] = np.hstack((A, np.eye(m)))  # Добавляем искусственные переменные
    table[:-1, -1] = b  # Правая часть ограничений
    table[-1, :-1] = np.hstack((-c, np.zeros(m)))  # Коэффициенты целевой функции

    step = 0  # Счетчик шагов
    while True:
        print(f"\nШаг {step}: Симплекс-таблица")
        print(table)

        # Шаг 1: Определение входящей переменной
        col = np.argmin(table[-1, :-1])  # Входящая переменная (наименьший коэффициент)
        if table[-1, col] >= 0:
            break  # Решение оптимально

        # Шаг 2: Определение выходящей переменной
        ratios = []
        for i in range(m):
            if table[i, col] > 0:
                ratios.append(table[i, -1] / table[i, col])
            else:
                ratios.append(np.inf)
        row = np.argmin(ratios)
        if all(r == np.inf for r in ratios):
            raise Exception("Задача не имеет решения")

        print(f"Входящая переменная: x{col + 1}, выходящая переменная: строка {row}")

        # Шаг 3: Обновление таблицы (приведение ведущего элемента к 1)
        table[row, :] /= table[row, col]
        for i in range(m + 1):
            if i != row:
                table[i, :] -= table[i, col] * table[row, :]

        step += 1

    # Возвращаем решение
    solution = np.zeros(n)
    for j in range(n):
        col_values = table[:-1, j]
        if np.count_nonzero(col_values) == 1 and table[-1, j] == 0:  # Проверка канонического вектора
            row = np.where(col_values == 1)[0][0]
            solution[j] = table[row, -1]
    return solution, table[-1, -1]  # Решение и значение целевой функции


# Определение целевой функции и ограничений
c = np.array([2, 2], dtype=float)  # Коэффициенты целевой функции
A = np.array([
    [-3, 2],   # -3x1 + 2x2 <= 6
    [-1, -1],  # -x1 - x2 <= -3
    [1, 0],    # x1 <= 3
    [0, 1]     # x2 <= 5
], dtype=float)
b = np.array([6, -3, 3, 5], dtype=float)  # Правая часть ограничений

# Решение задачи
solution, max_value = simplex_method(c, A, b)

# Вывод результатов
print("\nОптимальное решение:")
print(f"x1 = {solution[0]:.2f}, x2 = {solution[1]:.2f}")
print(f"Максимальное значение целевой функции: F = {max_value:.2f}")
