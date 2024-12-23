import numpy as np

# Константа большого значения M
M = 1e6

# Формирование начальной симплекс-таблицы
initial_table = np.array([
    [-3 * M, -2 - M, -2 - M, 0, M, 0, 0, 0],  # Целевая функция
    [6, -3, 2, 1, 0, 0, 0, 0],  # Ограничения
    [3, 1, 1, 0, -1, 0, 0, 0],
    [3, 1, 0, 0, 0, 1, 0, 0],
    [5, 0, 1, 0, 0, 0, 1, 0]
])

def get_pivot_row(table, pivot_col_index):
    """
    Определяет индекс разрешающей строки для текущего разрешающего столбца.
    """
    ratios = [
        (table[i][0] / table[i][pivot_col_index]) if table[i][pivot_col_index] > 0 else float('inf')
        for i in range(1, len(table))
    ]
    return 1 + np.argmin(ratios)

def update_tableau(table, pivot_row_index, pivot_col_index):
    """
    Пересчитывает симплекс-таблицу после выбора разрешающего элемента.
    """
    result_table = np.zeros_like(table, dtype=float)
    result_table[pivot_row_index] = table[pivot_row_index] / table[pivot_row_index][pivot_col_index]
    for row_index in range(len(table)):
        if row_index == pivot_row_index:
            continue
        result_table[row_index] = table[row_index] - (
            table[row_index][pivot_col_index] * result_table[pivot_row_index]
        )
    return result_table

# Хранилище для базисных переменных
basis_indices = []

# Симплекс-метод:
current_table = initial_table.copy()
while np.any(current_table[0] < 0):
    pivot_col_index = np.argmin(current_table[0])  # Выбор разрешающего столбца
    try:
        pivot_row_index = get_pivot_row(current_table, pivot_col_index)
    except ValueError:
        print("Оптимальное решение недостижимо: задача не ограничена.")
        break
    basis_indices.append((pivot_row_index, pivot_col_index))
    current_table = update_tableau(current_table, pivot_row_index, pivot_col_index)

# Вывод результатов
print("Итоговые результаты:")
for row_idx, col_idx in basis_indices:
    if col_idx > 0:  # Пропускаем вывод для x0
        print(f"x{col_idx} = {current_table[row_idx][0]}")
print(f"Оптимальное значение целевой функции F(x) = {current_table[0][0]}")

