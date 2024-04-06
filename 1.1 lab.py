import tkinter as tk
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GraphWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Поиск кратчайшего пути методом ближайшего соседа")
        self.geometry("780x800")
        self.options = {
            'node_color': '#e6e6fa',  # color of node
            'node_size': 300,  # size of node
            'width': 1,  # line width of edges
            'arrowstyle': '-|>',  # array style for directed graph
            'arrowsize': 10,  # size of arrow
            'edge_color': 'black',  # edge color
            'font_size': 12,
        }

        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_frame, width=350, height=400, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.vertices = []
        self.arrows = {}

        self.canvas.bind("<Button-1>", self.on_mouse_pressed)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_released)

        self.selected_vertex = None
        self.start_pos = None
        self.vertex_count = 0

        self.control_frame = tk.Frame(self.main_frame, bg="#EAEAEA")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        lbl3 = tk.Label(self.control_frame, text="Вводите взвешенные ребра в формате:", font=("Arial", 10),
                        bg="#EAEAEA")
        lbl3.grid(row=0, column=3, padx=5, pady=5)

        lbl_1 = tk.Label(self.control_frame, text="(1, 2, 3); (2, 1, 8)", font=("Arial", 10), bg="#EAEAEA")
        lbl_1.grid(row=1, column=3, padx=5, pady=5)

        lbl_2 = tk.Label(self.control_frame, text="где первые два числа - вершины,", font=("Arial", 10), bg="#EAEAEA")
        lbl_2.grid(row=2, column=3, padx=5, pady=5)

        lbl_3 = tk.Label(self.control_frame, text="а третье - длина ребра", font=("Arial", 10), bg="#EAEAEA")
        lbl_3.grid(row=3, column=3, padx=5, pady=5)

        self.txt_edges = tk.Text(self.control_frame, width=37, height=5)
        self.txt_edges.grid(row=4, column=3, padx=5, pady=5)

        lbl5 = tk.Label(self.control_frame, text="Полученный путь", font=("Arial", 10), bg="#EAEAEA")
        lbl5.grid(row=6, column=3, padx=5, pady=5)

        self.txt_path = tk.Text(self.control_frame, width=37, height=3, state='disabled', background='#EAEAEA',
                                fg="black")
        self.txt_path.grid(row=7, column=3, padx=5, pady=5)

        lbl_5 = tk.Label(self.control_frame, text="Длина пути", font=("Arial", 10), bg="#EAEAEA")
        lbl_5.grid(row=8, column=3, padx=5, pady=5)

        self.txt_path_length = tk.Text(self.control_frame, width=37, height=1, state='disabled', background='#EAEAEA',
                                       fg="black")
        self.txt_path_length.grid(row=9, column=3, padx=5, pady=5)

        btn_compute_path = tk.Button(self.control_frame, text="Вычислить оптимальный путь", width=42,
                                     command=self.nearest_neighbour, bg="#DDDDDD", activebackground="#CCCCCC",
                                     relief=tk.GROOVE)
        btn_compute_path.grid(row=10, column=3, padx=5, pady=5)

        btn_clear_graph = tk.Button(self.control_frame, text="Очистить граф", width=42, command=self.clear_graph,
                                    bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
        btn_clear_graph.grid(row=11, column=3, padx=5, pady=5)

        self.graph_canvas = tk.Canvas(self.control_frame, width=300, height=300, bg="white")
        self.graph_canvas.grid(row=12, column=3, padx=5, pady=5)

    def on_mouse_pressed(self, event):
        x, y = event.x, event.y
        item = self.canvas.find_overlapping(x - 5, y - 5, x + 5, y + 5)
        if item:
            self.selected_vertex = item[0]
            self.start_pos = event.x, event.y

    def on_mouse_released(self, event):
        if self.selected_vertex:
            x, y = event.x, event.y
            item = self.canvas.find_overlapping(x - 5, y - 5, x + 5, y + 5)
            if item and item[0] != self.selected_vertex:
                vertex_id = item[0]
                arrow = self.canvas.create_line(self.start_pos[0], self.start_pos[1], event.x, event.y, arrow=tk.LAST)
                self.arrows[(self.selected_vertex, vertex_id)] = arrow
        self.selected_vertex = None
        self.start_pos = None

        if not self.selected_vertex and not self.start_pos:
            nearby_vertices = self.canvas.find_overlapping(event.x - 15, event.y - 15, event.x + 15, event.y + 15)
            if not nearby_vertices:
                vertex = self.canvas.create_oval(event.x - 15, event.y - 15, event.x + 15, event.y + 15, fill="light blue")
                label = self.canvas.create_text(event.x, event.y, text=str(self.vertex_count), fill="black",
                                                font=("Helvetica", 12))
                self.vertices.append((vertex, label))
                self.vertex_count += 1

    def nearest_neighbour(self):
        nodes = [vertice for vertice in range(self.vertex_count)]
        edges = self.txt_edges.get("1.0", tk.END)
        edges = ''.join(edges).split(';')
        edges = [edge.strip() for edge in edges]
        weighted_edges = [eval(edge) for edge in edges]

        self.ANS_LEN = float('inf')
        self.ANS_CYCLE = []

        for started_id in range(self.vertex_count):
            summ = 0
            curr = nodes[started_id]
            completed = [curr]

            for i in range(self.vertex_count - 1):
                result = float('inf')
                for edge in weighted_edges:
                    if edge[0] == curr and edge[1] not in completed and edge[2] < result:
                        result = edge[2]
                        new_vert = edge[1]
                completed.append(new_vert)
                summ += result
                curr = new_vert

            flag = False
            for edge in weighted_edges:
                if edge[0] == completed[-1] and edge[1] == completed[0]:
                    summ += edge[2]
                    flag = True
                    break

            if flag:
                if summ < self.ANS_LEN:
                    self.ANS_LEN = summ
                    self.ANS_CYCLE = completed

        self.txt_path_length.config(state='normal')
        self.txt_path_length.delete(1.0, tk.END)
        self.txt_path_length.insert(tk.END, self.ANS_LEN)
        self.txt_path_length.config(state='disabled')

        self.txt_path.config(state='normal')
        self.txt_path.delete(1.0, tk.END)
        self.txt_path.insert(tk.END, self.ANS_CYCLE)
        self.txt_path.config(state='disabled')

        self.ANS_CYCLE.append(self.ANS_CYCLE[0])
        F = nx.DiGraph(directed=True)

        for i, (vertex, label) in enumerate(self.vertices):
            F.add_node(i)
            pos = self.canvas.coords(vertex)
            pos = (pos[0] + 15, self.canvas.winfo_height() - pos[1] - 15)
            F.nodes[i]['pos'] = pos

        for i in range(len(self.ANS_CYCLE) - 1):
            F.add_edge(self.ANS_CYCLE[i], self.ANS_CYCLE[i + 1])

        fig = Figure(figsize=(3, 3), dpi=100)
        ax = fig.add_subplot(111)

        pos = nx.get_node_attributes(F, 'pos')
        nx.draw(F, pos=pos, ax=ax, with_labels=True, arrows=True, **self.options)

        ax.set_aspect('equal')
        ax.set_adjustable('box')
        canvas = FigureCanvasTkAgg(fig, master=self.control_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=12, column=3, padx=5, pady=0)

        self.graph_canvas.update()

    def clear_graph(self):
        self.canvas.delete("all")
        self.vertices = []
        self.arrows = {}
        self.vertex_count = 0

        self.graph_canvas.delete("all")


if __name__ == "__main__":
    app = GraphWindow()
    app.mainloop()