import numpy as np
import tkinter as tk
from tkinter import ttk


class GeneticAlgorithm:
    def __init__(self, num_generations, population_size, mutation_rate, encoding_type, lower_bound, upper_bound):
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.encoding_type = encoding_type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population = self.initialize_population()

    def evaluate_fitness(self, x, y):
        return 2 * (x ** 3) + 4 * x * (y ** 3) - 10 * x * y + (y ** 2)

    def initialize_population(self):
        if self.encoding_type == 'real':
            return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.population_size, 2))
        else:
            return np.random.uniform(0, 1, size=(self.population_size, 2))

    def enforce_bounds(self, individual):
        if self.encoding_type == 'real':
            return np.maximum(np.minimum(individual, self.upper_bound), self.lower_bound)
        else:
            return np.maximum(np.minimum(individual, 1), 0)

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual):
        mutation_mask = np.random.rand(*individual.shape) < self.mutation_rate
        individual[mutation_mask] = np.random.randn(*individual.shape)[mutation_mask]
        return self.enforce_bounds(individual)

    def select_parents(self, population, fitness_values):
        tournament_size = 3
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness_values[indices]
        parent_index = indices[np.argmin(tournament_fitness)]
        return population[parent_index]


def clear_entry(entry, value):
    entry.delete(0, tk.END)
    entry.insert(0, value)


def update_table(data):
    for row in tree.get_children():
        tree.delete(row)
    for row_data in data:
        tree.insert("", "end", values=row_data)


def run_genetic_algorithm(encoding_type):
    mutation_rate = float(probability_entry.get()) / 100.0
    population_size = int(chromosome_count_entry.get())
    num_generations = int(iteration_count_entry.get())
    lower_bound = float(min_gene_entry.get())
    upper_bound = float(max_gene_entry.get())

    genetic_algorithm = GeneticAlgorithm(num_generations, population_size, mutation_rate, encoding_type, lower_bound,
                                         upper_bound)
    generations_data = []

    for generation in range(num_generations):
        fitness_values = np.apply_along_axis(lambda x: genetic_algorithm.evaluate_fitness(x[0], x[1]), 1,
                                             genetic_algorithm.population)

        sorted_indices = np.argsort(fitness_values)
        genetic_algorithm.population = genetic_algorithm.population[sorted_indices]

        selected_population = genetic_algorithm.population[:genetic_algorithm.population_size // 2]

        new_population = []
        for i in range(genetic_algorithm.population_size // 2):
            parent1 = genetic_algorithm.select_parents(selected_population, fitness_values)
            parent2 = genetic_algorithm.select_parents(selected_population, fitness_values)
            child1, child2 = genetic_algorithm.crossover(parent1, parent2)
            child1 = genetic_algorithm.mutate(child1)
            child2 = genetic_algorithm.mutate(child2)
            new_population.extend([child1, child2])

        genetic_algorithm.population = np.array(new_population)
        best_individual = genetic_algorithm.population[0]
        best_individual = genetic_algorithm.enforce_bounds(best_individual)
        best_fitness = genetic_algorithm.evaluate_fitness(best_individual[0], best_individual[1])

        generations_data.append((generation + 1, best_fitness, best_individual[0], best_individual[1]))

    if encoding_type == 'logarithmic':
        x_log = lower_bound * 10 ** (best_individual[0] * np.log10(1 + upper_bound / max(lower_bound, 1e-10)))
        y_log = lower_bound * 10 ** (best_individual[1] * np.log10(1 + upper_bound / max(lower_bound, 1e-10)))
        coord_text = f"Координаты точки: ({-abs(x_log):.4f}, {y_log:.4f})"
        fitness_text = f"Функция: {-abs(genetic_algorithm.evaluate_fitness(x_log, y_log)):.4f}"
    else:
        coord_text = f"Координаты точки: ({best_individual[0]:.4f}, {best_individual[1]:.4f})"
        fitness_text = f"Функция: {best_fitness:.4f}"

    result_canvas.delete("all")
    result_canvas.itemconfig(result_canvas.create_text(197, 10, text=coord_text, font=("Arial", 10)),
                             tags="result_text_coord")
    result_canvas.itemconfig(result_canvas.create_text(197, 40, text=fitness_text, font=("Arial", 10)),
                             tags="result_text_fitness")
    update_table(generations_data)


root = tk.Tk()
root.title("Генетический алгоритм для поиска минимума функции")
root.geometry('1000x600')

# Создаем холст для отображения графики
canvas = tk.Canvas(root, width=1000, height=1000, borderwidth=0, highlightthickness=0)
canvas.place(relx=0, rely=0.0)

# Фрейм для таблицы результатов
frame_tree = ttk.Frame(root, borderwidth=2, relief="groove")
frame_tree.place(relx=0.49, rely=0.05)

# Таблица результатов
columns = ("Номер", "Результат", "Ген 1", "Ген 2")
tree = ttk.Treeview(frame_tree, columns=columns, show="headings", height=21)

style = ttk.Style()
style.configure("Treeview.Heading", font=("Arial", 10))
style.configure("Treeview", font=("Arial", 9), rowheight=25)
style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])

for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=120)

tree.pack(expand=True, fill="both")

# Фрейм для предварительных настроек
frame_settings = ttk.Frame(root, borderwidth=2, relief="groove")
frame_settings.place(relx=0.02, rely=0.05)

# Отображаем функцию и ее описание
lbl_func = tk.Label(frame_settings, text="Функция:", font=("Arial", 10))
lbl_func.grid(row=0, column=0, padx=5, pady=5, sticky="w")
lbl_func2 = tk.Label(frame_settings, text="2x[1]^3 + 4x[1]x[2]^3-10x[1]x[2]+x[2]^3", font=("Arial", 10))
lbl_func2.grid(row=0, column=1, padx=5, pady=5, sticky="w")

lbl_mutation = tk.Label(frame_settings, text="Вероятность мутации, %:", font=("Arial", 10))
lbl_mutation.grid(row=1, column=0, padx=5, pady=5, sticky="w")

probability_entry = tk.Entry(frame_settings, width=7)
probability_entry.insert(0, "20")
probability_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

lbl_chromosomes = tk.Label(frame_settings, text="Количество хромосом:", font=("Arial", 10))
lbl_chromosomes.grid(row=2, column=0, padx=5, pady=5, sticky="w")

chromosome_count_entry = tk.Entry(frame_settings, width=7)
chromosome_count_entry.insert(0, "50")
chromosome_count_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

lbl_min_gene = tk.Label(frame_settings, text="Минимальное значение гена:", font=("Arial", 10))
lbl_min_gene.grid(row=3, column=0, padx=5, pady=5, sticky="w")

min_gene_entry = tk.Entry(frame_settings, width=7)
min_gene_entry.insert(0, "-50")
min_gene_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

lbl_max_gene = tk.Label(frame_settings, text="Максимальное значение гена:", font=("Arial", 10))
lbl_max_gene.grid(row=4, column=0, padx=5, pady=5, sticky="w")

max_gene_entry = tk.Entry(frame_settings, width=7)
max_gene_entry.insert(0, "50")
max_gene_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")

# Фрейм для управления
frame_control = ttk.Frame(root, borderwidth=2, relief="groove")
frame_control.place(relx=0.02, rely=0.37)

lbl_control = tk.Label(frame_control, text="Управление", font=("Arial", 10))
lbl_control.grid(row=0, column=0, padx=5, pady=5, sticky="w")

lbl_iterations = tk.Label(frame_control, text="Количество поколений:", font=("Arial", 10))
lbl_iterations.grid(row=1, column=0, padx=5, pady=5, sticky="w")

iteration_count_entry = tk.Spinbox(frame_control, from_=1, to=5000, width=5)
iteration_count_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

button_1 = tk.Button(frame_control, text="1", width=8,
                     command=lambda: clear_entry(iteration_count_entry, 1),
                     bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
button_1.place(relx=0.02, rely=0.40)

button_10 = tk.Button(frame_control, text="10", width=8,
                      command=lambda: clear_entry(iteration_count_entry, 10),
                      bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
button_10.place(relx=0.27, rely=0.40)


button_100 = tk.Button(frame_control, text="100", width=8,
                       command=lambda: clear_entry(iteration_count_entry, 100),
                       bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
button_100.place(relx=0.52, rely=0.40)

button_1000 = tk.Button(frame_control, text="1000", width=8,
                        command=lambda: clear_entry(iteration_count_entry, 1000),
                        bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
button_1000.grid(row=2, column=3, padx=5, pady=5, sticky="w")

button_calculate_log = tk.Button(frame_control, text="Рассчитать (логарифмическое кодирование)", width=60,
                                 command=lambda: run_genetic_algorithm('logarithmic'),
                                 bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
button_calculate_log.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky="w")

button_calculate_real = tk.Button(frame_control, text="Рассчитать (вещественное кодирование)", width=60,
                                  command=lambda: run_genetic_algorithm('real'),
                                  bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
button_calculate_real.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky="w")

# Фрейм для результатов
frame_results = ttk.Frame(root, borderwidth=2, relief="groove")
frame_results.place(relx=0.02, rely=0.73)

lbl_results = tk.Label(frame_results, text="Результаты", font=("Arial", 10))
lbl_results.grid(row=0, column=0, pady=5, sticky="w")

lbl_solution = tk.Label(frame_results, text="Лучшее решение достигается в точке:", font=("Arial", 10))
lbl_solution.grid(row=1, column=0, pady=5, sticky="w")

result_canvas = tk.Canvas(frame_results, width=433, height=50, bg="white", borderwidth=1,
                          highlightbackground="#CCCCCC", highlightthickness=2)
result_canvas.grid(row=2, column=0, pady=5, sticky="w")

root.mainloop()