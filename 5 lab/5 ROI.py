import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class Particle:
    def __init__(self, swarm):
        self.current_position = np.random.rand(2) * (swarm.max_values - swarm.min_values) + swarm.min_values
        self.local_best_position = self.current_position
        self.local_best_fitness = swarm.get_final_fitness(self.current_position)
        self.velocity = self.initialize_velocity(swarm)

    def initialize_velocity(self, swarm):
        min_val = -(swarm.max_values - swarm.min_values)
        max_val = (swarm.max_values - swarm.min_values)
        return np.random.rand(2) * (max_val - min_val) + min_val

    def next_iteration(self, swarm):
        rnd_curr_best_position = np.random.rand(2)
        rnd_global_best_position = np.random.rand(2)
        velo_ratio = swarm.local_velocity_ratio + swarm.global_velocity_ratio
        common_ratio = (2.0 * swarm.curr_velocity_ratio /
                        (np.abs(2.0 - velo_ratio - np.sqrt(velo_ratio ** 2 - 4.0 * velo_ratio))))
        new_velocity_part2 = (common_ratio *
                              swarm.local_velocity_ratio *
                              rnd_curr_best_position *
                              (self.local_best_position - self.current_position))
        new_velocity_part3 = (common_ratio *
                              swarm.global_velocity_ratio *
                              rnd_global_best_position *
                              (swarm.global_best_position - self.current_position))
        self.velocity = common_ratio * self.velocity + new_velocity_part2 + new_velocity_part3
        self.current_position += self.velocity
        final_fitness = swarm.get_final_fitness(self.current_position)
        if self.local_best_fitness is None or final_fitness < self.local_best_fitness:
            self.local_best_position = self.current_position
            self.local_best_fitness = final_fitness
        if swarm.global_best_fitness is None or final_fitness < swarm.global_best_fitness:
            swarm.global_best_position = self.current_position
            swarm.global_best_fitness = final_fitness


class Swarm:
    def __init__(self, swarm_size, min_values, max_values, curr_velocity_ratio, local_velocity_ratio,
                 global_velocity_ratio):
        self.swarm_size = swarm_size
        self.min_values = np.array(min_values)
        self.max_values = np.array(max_values)
        self.curr_velocity_ratio = curr_velocity_ratio
        self.local_velocity_ratio = local_velocity_ratio
        self.global_velocity_ratio = global_velocity_ratio
        self.global_best_fitness = float('inf')
        self.global_best_position = None
        self.swarm = self.create_swarm()

    def create_swarm(self):
        return [Particle(self) for _ in range(self.swarm_size)]

    def next_iteration(self):
        for particle in self.swarm:
            particle.next_iteration(self)

    def final_fitness(self, position):
        result = 2 * (position[0] ** 3) + 4 * position[0] * (position[1] ** 3) - 10 * position[0] * position[1] + \
                 position[1] ** 2
        return result if result is not None else float('inf')

    def get_final_fitness(self, position):
        final_fitness = self.final_fitness(position)
        if self.global_best_fitness is None or final_fitness < self.global_best_fitness:
            self.global_best_fitness = final_fitness
            self.global_best_position = position
        return final_fitness


def insert_value(place, num):
    place.delete(0, tk.END)
    place.insert(0, str(num))


def create_swarm():
    global swarm, ax, scatter, canvas_widget
    curr_velocity_ratio = float(currVeloc.get())
    local_velocity_ratio = float(localBest.get())
    global_velocity_ratio = float(globalBest.get())
    swarm_size = int(cntPieces.get())
    swarm = Swarm(swarm_size, [-10, -10], [5, 5], curr_velocity_ratio, local_velocity_ratio, global_velocity_ratio)
    ax.clear()
    ax.set_xlim(-10, 5)
    ax.set_ylim(-10, 5)
    ax.set_title("Расположение роя")
    scatter = ax.scatter([], [], color='blue', marker='o')
    canvas_widget.get_tk_widget().update_idletasks()
    positions = np.array([particle.current_position for particle in swarm.swarm])
    scatter.set_offsets(positions)
    canvas_widget.draw()


def make_iterations(n):
    global swarm, ax, scatter, canvas_widget
    it = int(txt4.get())
    txt4.delete(0, tk.END)
    insert_value(txt4, str(it + n))
    for _ in range(n):
        swarm.next_iteration()
    positions = np.array([particle.current_position for particle in swarm.swarm])
    scatter.set_offsets(positions)
    canvas_widget.draw()
    best_position = swarm.global_best_position
    best_fitness = swarm.global_best_fitness
    canvas2.delete("all")
    canvas2.create_text(10, 10, anchor="nw", text=f"Лучшее решение: {best_position}\nЗначение функции: {best_fitness}",
                        font=("Arial", 10), fill="black")


root = tk.Tk()
root.title("Роевой интеллект для поиска минимума функции")
root.geometry('1500x800')  # Увеличил размер окна

canvas = tk.Canvas(root, width=1200, height=1000, borderwidth=0, highlightthickness=0)
canvas.place(relx=0, rely=0.0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_facecolor('white')
ax.tick_params(axis='both', labelsize=8)
ax.set_xlim(-10, 5)
ax.set_ylim(-10, 5)
ax.set_title("Расположение роя")

canvas_widget = FigureCanvasTkAgg(fig, master=root)
canvas_widget.get_tk_widget().place(relx=0.45, rely=0.42)

lbl = tk.Label(root, text="Предварительные настройки", font=("Arial", 14))
lbl.place(relx=0.1, rely=0.01)

lblfunc = tk.Label(root, text="Функция:", font=("Arial", 14))
lblfunc.place(relx=0.027, rely=0.075)

lblfunc2 = tk.Label(root, text="2x[1]^3 + 4x[1]x[2]^3-10x[1]x[2]+x[2]^3", font=("Arial", 14))
lblfunc2.place(relx=0.12, rely=0.075)

lblk = tk.Label(root, text="Коэффициент текущей скорости:", font=("Arial", 14))
lblk.place(relx=0.027, rely=0.155)

currVeloc = tk.Entry(root, width=7, font=("Arial", 14))
currVeloc.insert(0, "0.3")
currVeloc.place(relx=0.35, rely=0.16)

lblk2 = tk.Label(root, text="Коэф-т локального лучшего значения:", font=("Arial", 14))
lblk2.place(relx=0.027, rely=0.235)

localBest = tk.Entry(root, width=7, font=("Arial", 14))
localBest.insert(0, "2")
localBest.place(relx=0.35, rely=0.24)

lblk2 = tk.Label(root, text="Коэф-т глобального лучшего значения:", font=("Arial", 14))
lblk2.place(relx=0.027, rely=0.315)

globalBest = tk.Entry(root, width=7, font=("Arial", 14))
globalBest.insert(0, "5")
globalBest.place(relx=0.35, rely=0.32)

lblk3 = tk.Label(root, text="Количество частиц:", font=("Arial", 14))
lblk3.place(relx=0.027, rely=0.395)

cntPieces = tk.Spinbox(root, from_=10, to=500, width=5, font=("Arial", 14))
cntPieces.place(relx=0.35, rely=0.4)

lbl4 = tk.Label(root, text="Управление", font=("Arial", 14))
lbl4.place(relx=0.18, rely=0.5)

but = tk.Button(root, text="Создать частицы", width=50,
                command=create_swarm,
                bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE, font=("Arial", 14))
but.place(relx=0.03, rely=0.58)

lbl5 = tk.Label(root, text="Количество итераций:", font=("Arial", 14))
lbl5.place(relx=0.027, rely=0.67)

cntIt = tk.Spinbox(root, from_=1, to=5000, width=5, font=("Arial", 14))
cntIt.place(relx=0.35, rely=0.68)

but1 = tk.Button(root, text="1", width=8,
                 command=lambda: insert_value(cntIt, 1),
                 bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE, font=("Arial", 14))
but1.place(relx=0.03, rely=0.75)

but2 = tk.Button(root, text="10", width=8,
                 command=lambda: insert_value(cntIt, 10),
                 bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE, font=("Arial", 14))
but2.place(relx=0.13, rely=0.75)

but3 = tk.Button(root, text="100", width=8,
                 command=lambda: insert_value(cntIt, 100),
                 bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE, font=("Arial", 14))
but3.place(relx=0.23, rely=0.75)

but4 = tk.Button(root, text="1000", width=8,
                 command=lambda: insert_value(cntIt, 1000),
                 bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE, font=("Arial", 14))
but4.place(relx=0.328, rely=0.75)

but = tk.Button(root, text="Рассчитать", width=50,
                command=lambda: make_iterations(int(cntIt.get())),
                bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE, font=("Arial", 14))
but.place(relx=0.03, rely=0.85)

lbl5 = tk.Label(root, text="Количество выполненных итераций:", font=("Arial", 14))
lbl5.place(relx=0.027, rely=0.94)

cnt = 0
entry_var = tk.StringVar()
txt4 = tk.Entry(root, width=7, textvariable=entry_var, state='disabled', disabledbackground="white", fg="black",
                font=("Arial", 14))
txt4.place(relx=0.35, rely=0.94)
entry_var.set(str(cnt))

lbl6 = tk.Label(root, text="Результаты", font=("Arial", 14))
lbl6.place(relx=0.67, rely=0.01)

lbl5 = tk.Label(root, text="Лучшее решение достигается в точке:", font=("Arial", 14))
lbl5.place(relx=0.60, rely=0.075)

canvas2 = tk.Canvas(root, width=800, height=160, bg="white", borderwidth=1, highlightbackground="#CCCCCC",
                    highlightthickness=2)
canvas2.place(relx=0.449, rely=0.14)

root.mainloop()
