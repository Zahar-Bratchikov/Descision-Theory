import numpy as np

def simplex_method(c, A, b):
    def pivot(table, row, col):
        """ Пивотирование: обновление симплекс-таблицы """
        table[row] /= table[row, col]
        for i in range(len(table)):
            if i != row:
                table[i] -= table[i, col] * table[row]

    def print_table(table, iteration):
        """ Печать текущей симплекс-таблицы с округлением до 2 знаков """
        rounded_table = np.round(table, 2)
        print(f"\nИтерация {iteration}:")
        print(rounded_table)

    num_vars = len(c)  # Количество переменных
    num_constraints = len(b)  # Количество ограничений

    # Инициализация симплекс-таблицы
    table = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))

    # Заполнение ограничений
    for i in range(num_constraints):
        table[i, :num_vars] = A[i]
        table[i, num_vars + i] = 1  # Базисные переменные
        table[i, -1] = b[i]

    # Заполнение целевой функции
    table[-1, :num_vars] = -np.array(c)

    # Итерации симплекс-метода
    iteration = 0
    print_table(table, iteration)
    while np.any(table[-1, :-1] < 0):  # Пока есть отрицательные коэффициенты в целевой функции
        iteration += 1

        # Ведущий столбец (наименьший элемент в последней строке)
        col = np.argmin(table[-1, :-1])

        # Проверка на неограниченность целевой функции
        if all(table[:-1, col] <= 0):
            raise ValueError("Целевая функция не ограничена!")

        # Ведущая строка (минимальное положительное отношение)
        ratios = np.divide(table[:-1, -1], table[:-1, col], where=table[:-1, col] > 0)
        ratios[table[:-1, col] <= 0] = np.inf  # Игнорируем некорректные строки
        row = np.argmin(ratios)

        # Выполнение пивотирования
        pivot(table, row, col)

        # Вывод текущей таблицы
        print_table(table, iteration)

    # Извлечение решения
    solution = np.zeros(num_vars)
    for i in range(num_constraints):
        basic_var = np.where(table[i, :num_vars] == 1)[0]
        if len(basic_var) == 1:
            solution[basic_var[0]] = table[i, -1]

    # Оптимальное значение целевой функции
    return solution, table[-1, -1]

# Данные задачи
c = [2, 2]  # Целевая функция: F = 2x1 + 2x2
A = [
    [3, -2],  # 3x1 - 2x2 <= 6
    [-1, -1],  # x1 + x2 >= 3 -> -x1 - x2 <= -3
    [1, 0],   # x1 <= 3
    [0, 1]    # x2 <= 5
]
b = [6, -3, 3, 5]  # Правая часть ограничений

# Решение задачи
try:
    solution, max_value = simplex_method(c, A, b)
    print("\nОптимальное решение:", np.round(solution, 2))
    print("Максимальное значение целевой функции:", round(max_value, 2))
except ValueError as e:
    print("Ошибка:", e)
