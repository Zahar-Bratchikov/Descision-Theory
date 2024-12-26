class BasisCell:
    def __init__(self, row, col):
        self.row = row  # Номер строки ячейки
        self.col = col  # Номер столбца ячейки


class LoopState:
    def __init__(self, cell, previous_direction, next_cells):
        self.cell = cell  # Текущая ячейка в цикле
        self.previous_direction = previous_direction  # Направление, из которого пришли в эту ячейку
        self.next_cells = next_cells  # Список доступных соседних ячеек


def solve_transportation(supplies, demands, cost_matrix):
    # Основная функция для решения транспортной задачи

    # Шаг 1: Построение начального плана методом северо-западного угла
    allocation = [[0] * len(demands) for _ in range(len(supplies))]  # Таблица распределения
    remaining_supplies = supplies[:]  # Оставшиеся поставки
    remaining_demands = demands[:]  # Оставшиеся потребности
    basis_cells = []  # Базисные ячейки

    i, j = 0, 0
    while i < len(supplies) and j < len(demands):
        # Заполняем ячейку минимально возможным значением
        if remaining_supplies[i] < remaining_demands[j]:
            allocation[i][j] = remaining_supplies[i]
            basis_cells.append(BasisCell(i, j))
            remaining_demands[j] -= remaining_supplies[i]
            remaining_supplies[i] = 0
            i += 1  # Переход к следующему поставщику
        else:
            allocation[i][j] = remaining_demands[j]
            basis_cells.append(BasisCell(i, j))
            remaining_supplies[i] -= remaining_demands[j]
            remaining_demands[j] = 0
            j += 1  # Переход к следующему потребителю

    # Вычисление стоимости начального плана
    initial_cost = sum(allocation[cell.row][cell.col] * cost_matrix[cell.row][cell.col] for cell in basis_cells)
    print(f"Начальная стоимость = {initial_cost}")

    # Шаг 2: Оптимизация методом потенциалов
    optimize_potential_method(supplies, demands, allocation, cost_matrix, basis_cells)


def optimize_potential_method(supplies, demands, allocation, costs, basis_cells):
    # Метод потенциалов для оптимизации решения
    num_suppliers, num_consumers = len(supplies), len(demands)
    while True:
        # Сортируем базисные ячейки для корректной обработки
        basis_cells.sort(key=lambda cell: (cell.row, cell.col))

        # Вычисляем потенциалы (u для строк и v для столбцов)
        potentials_u = [None] * num_suppliers
        potentials_v = [None] * num_consumers
        potentials_u[0] = 0  # Начинаем с первой строки

        while any(p is None for p in potentials_u + potentials_v):
            for cell in basis_cells:
                i, j = cell.row, cell.col
                if potentials_u[i] is not None and potentials_v[j] is None:
                    potentials_v[j] = costs[i][j] - potentials_u[i]  # Рассчитываем потенциал столбца
                elif potentials_v[j] is not None and potentials_u[i] is None:
                    potentials_u[i] = costs[i][j] - potentials_v[j]  # Рассчитываем потенциал строки

        # Поиск внебазисных ячеек с отрицательной оценкой
        non_basis_cells = []
        reduced_costs = []

        for i in range(num_suppliers):
            for j in range(num_consumers):
                if all(cell.row != i or cell.col != j for cell in basis_cells):
                    reduced_cost = potentials_u[i] + potentials_v[j] - costs[i][j]
                    if reduced_cost > 0:  # Выбираем ячейки с положительной оценкой
                        non_basis_cells.append(BasisCell(i, j))
                        reduced_costs.append(reduced_cost)

        if not non_basis_cells:
            # Если нет подходящих ячеек, решение оптимально
            print("Найдено оптимальное решение.")
            break

        # Выбираем ячейку с максимальной оценкой
        max_reduced_cost = max(reduced_costs)
        entering_cell = next(cell for cell, cost in zip(non_basis_cells, reduced_costs) if cost == max_reduced_cost)
        basis_cells.append(entering_cell)

        # Строим замкнутый цикл и обновляем распределения
        loop_path = construct_closed_loop(entering_cell, basis_cells)
        minus_cells = loop_path[1::2]  # Ячейки, из которых вычитаем
        min_allocation = min(allocation[cell.row][cell.col] for cell in minus_cells)

        for idx, cell in enumerate(loop_path):
            if idx % 2 == 0:
                allocation[cell.row][cell.col] += min_allocation  # Прибавляем к чётным ячейкам
            else:
                allocation[cell.row][cell.col] -= min_allocation  # Вычитаем из нечётных ячеек

        # Удаляем нулевые распределения из базиса
        for cell in minus_cells:
            if allocation[cell.row][cell.col] == 0:
                basis_cells.remove(cell)

    # Вычисляем оптимальную стоимость
    optimal_cost = sum(allocation[cell.row][cell.col] * costs[cell.row][cell.col] for cell in basis_cells)
    print(f"Оптимальная стоимость = {optimal_cost}")
    # Вывод финального плана перевозок
    print("Финальный план перевозок:")
    for row in allocation:
        print("\t".join(f"{value:5}" for value in row))

def construct_closed_loop(start_cell, basis_cells):
    # Построение замкнутого цикла для пересчёта распределений
    stack = [LoopState(start_cell, 'vertical',
                       [cell for cell in basis_cells if cell.row == start_cell.row and cell.col != start_cell.col])]

    while stack:
        head = stack[-1]

        if len(stack) >= 4 and ((head.cell.row == start_cell.row) or (head.cell.col == start_cell.col)):
            # Если цикл замкнулся, выходим
            break

        if not head.next_cells:
            stack.pop()  # Удаляем ячейку, если больше нет путей
            continue

        next_cell = head.next_cells.pop()
        next_direction = 'horizontal' if head.previous_direction == 'vertical' else 'vertical'
        next_cells = [
            cell for cell in basis_cells
            if (cell.col == next_cell.col if next_direction == 'horizontal' else cell.row == next_cell.row)
               and (cell.row != next_cell.row if next_direction == 'horizontal' else cell.col != next_cell.col)
        ]

        stack.append(LoopState(next_cell, next_direction, next_cells))

    return [state.cell for state in stack]


if __name__ == "__main__":
    supplies = [30, 20, 40, 50]  # Запасы поставщиков
    demands = [35, 20, 55, 30]  # Потребности потребителей
    cost_matrix = [  # Матрица стоимости перевозки
        [2, 4, 1, 3],
        [5, 6, 5, 4],
        [3, 7, 9, 5],
        [1, 2, 2, 7]
    ]

    solve_transportation(supplies, demands, cost_matrix)