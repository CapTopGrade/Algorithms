import random
import math


class Annealing:
    def __init__(self, network, start_temp, cool_rate, max_iter):
        self.network = network
        self.start_temp = start_temp
        self.cool_rate = cool_rate
        self.max_iter = max_iter

    def generate_initial_solution(self):
        initial_vertex = random.choice(list(self.network.keys()))
        current_vertex = initial_vertex
        visited_vertices = {current_vertex}
        route = [current_vertex]

        while len(visited_vertices) < len(self.network):
            unvisited_neighbors = [neighbor for neighbor in self.network[current_vertex] if
                                   neighbor not in visited_vertices]
            if not unvisited_neighbors:
                initial_vertex = random.choice(list(self.network.keys()))
                current_vertex = initial_vertex
                visited_vertices = {current_vertex}
                route = [current_vertex]
            else:
                next_vertex = random.choice(unvisited_neighbors)
                route.append(next_vertex)
                visited_vertices.add(next_vertex)
                current_vertex = next_vertex

        if route[-1] not in self.network[route[0]]:
            return self.generate_initial_solution()
        return route

    def compute_distance(self, route):
        total_distance = 0
        route_length = len(route)
        for i in range(route_length):
            vertex1, vertex2 = route[i], route[(i + 1) % route_length]
            if vertex1 in self.network and vertex2 in self.network[vertex1]:
                total_distance += self.network[vertex1][vertex2]
            else:
                return None
        return total_distance

    def perform_simulated_annealing(self):
        current_route = self.generate_initial_solution()
        current_distance = self.compute_distance(current_route)

        optimal_route = current_route.copy()
        optimal_distance = current_distance
        temp = self.start_temp

        for _ in range(self.max_iter):
            if optimal_distance is None:
                break

            vertex_indices = list(self.network.keys())
            i, j = sorted(random.sample(range(len(vertex_indices)), 2))
            vertex1, vertex2 = vertex_indices[i], vertex_indices[j]

            if (current_route[(current_route.index(vertex1) - 1) % len(current_route)] not in self.network[vertex2]) or \
                    (current_route[(current_route.index(vertex1) + 1) % len(current_route)] not in self.network[
                        vertex2]) or \
                    (current_route[(current_route.index(vertex2) - 1) % len(current_route)] not in self.network[
                        vertex1]) or \
                    (current_route[(current_route.index(vertex2) + 1) % len(current_route)] not in self.network[
                        vertex1]):
                continue

            new_route = current_route[:current_route.index(vertex1)] + [vertex2] + current_route[
                                                                                   current_route.index(vertex1) + 1:]
            new_route = new_route[:new_route.index(vertex2)] + [vertex1] + new_route[new_route.index(vertex2) + 1:]
            new_distance = self.compute_distance(new_route)
            if new_distance is None:
                continue
            distance_diff = new_distance - current_distance

            if distance_diff < 0 or random.random() < math.exp(-distance_diff / temp):
                current_route = new_route
                current_distance = new_distance
                if current_distance < optimal_distance:
                    optimal_route = current_route.copy()
                    optimal_distance = current_distance

            temp *= self.cool_rate
        return optimal_route
