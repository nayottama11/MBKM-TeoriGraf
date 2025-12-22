import tkinter as tk
from tkinter import messagebox
from collections import defaultdict
import random
import math
import itertools 
import sys
import heapq 
import time

# Constants
NODE_RADIUS = 15
CANVAS_WIDTH = 650
CANVAS_HEIGHT = 650
NUM_CITIES = 7 
ANIMATION_DELAY_MS = 800 

# Colors
COLOR_DEFAULT = "#95a5a6"       # Grey (Untraversed)
COLOR_TRAVERSED = "#3498db"     # Blue (Traversed Path)
COLOR_CURRENT = "#f39c12"       # Yellow/Orange (Current Step)
COLOR_ODD_MATCH = "#e74c3c"      # Red (Odd Nodes / Duplicated Edges)

# =======================================================================
# SECTION 1: CHINESE POSTMAN PROBLEM SOLVER (CPP SOLVER)
# =======================================================================

class UnionFind:
    """Disjoint Set Union (DSU) for connectivity checking."""
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

class CPPSolver:
    """Stores graph data and executes the CPP algorithm steps."""
    def __init__(self):
        self.city_names = []
        self.adj = defaultdict(lambda: defaultdict(lambda: float('inf')))
        self.edges = []
        self.node_positions = {}
        self.V = 0

    def generate_random_graph(self, num_cities, max_weight=20, edge_probability=0.3):
        """Generates a connected random graph (Normal mode)."""
        self.reset()
        self.V = num_cities
        self.city_names = [str(i) for i in range(num_cities)]

        center_x = CANVAS_WIDTH / 2
        center_y = CANVAS_HEIGHT / 2
        radius = min(CANVAS_WIDTH, CANVAS_HEIGHT) / 2 - 50 

        for i in range(self.V):
            angle = 2 * math.pi * i / self.V
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.node_positions[i] = (x, y)
        
        all_possible_edges = []
        for u in range(self.V):
            for v in range(u + 1, self.V):
                weight = random.randint(1, max_weight)
                all_possible_edges.append((weight, u, v))
        
        uf_check = UnionFind(self.V)
        temp_edges = []

        # 1. Build initial graph (ensuring connectivity via MST-like approach)
        for w, u, v in sorted(all_possible_edges, key=lambda x: x[0]):
            if uf_check.find(u) != uf_check.find(v):
                uf_check.union(u, v)
                temp_edges.append((w, u, v))
            elif random.random() < edge_probability:
                temp_edges.append((w, u, v))
        
        # 2. Finalize Graph
        for w, u, v in temp_edges:
            if self.adj[u][v] == float('inf'):
                self.adj[u][v] = self.adj[v][u] = w
                self.edges.append((w, u, v))

    def reset(self):
        self.city_names = []
        self.adj = defaultdict(lambda: defaultdict(lambda: float('inf')))
        self.edges = []
        self.node_positions = {}
        self.V = 0

    def get_odd_degree_vertices(self):
        degrees = defaultdict(int)
        for u in range(self.V):
            degrees[u] = len(self.adj[u]) 
        odd_vertices = [i for i in range(self.V) if degrees[i] % 2 != 0]
        return odd_vertices

    def dijkstra(self, start_node):
        distances = {node: float('inf') for node in range(self.V)}
        distances[start_node] = 0
        priority_queue = [(0, start_node)]
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            if current_distance > distances[current_node]:
                continue
                
            for neighbor, weight in self.adj[current_node].items():
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
        return distances

    def find_all_shortest_paths(self, odd_vertices):
        dist = {}
        for u in odd_vertices:
            distances = self.dijkstra(u)
            dist[u] = {v: distances[v] for v in odd_vertices if u != v}
        return dist

    def find_min_matching(self, odd_vertices, dist_dict):
        num_odd = len(odd_vertices)
        if num_odd == 0: return [], 0
        if num_odd > 10: return None, "Too many odd degree vertices for brute force matching."
             
        def generate_matchings(nodes):
            if not nodes: yield []; return
            u = nodes[0]; remaining = nodes[1:]
            for i in range(0, len(remaining)):
                v = remaining[i]; rest = remaining[:i] + remaining[i+1:]
                for sub_matching in generate_matchings(rest):
                    yield [(u, v)] + sub_matching

        best_matching_cost = float('inf'); best_matching_edges = [] 
        odd_vertex_indices = list(range(num_odd))
        
        for matching_indices in generate_matchings(odd_vertex_indices):
            current_cost = 0; current_matching_edges = []
            for u_idx, v_idx in matching_indices:
                u = odd_vertices[u_idx]; v = odd_vertices[v_idx]
                weight = dist_dict[u][v]
                if weight == float('inf'): current_cost = float('inf'); break
                current_cost += weight
                current_matching_edges.append((u, v, weight))
                
            if current_cost < best_matching_cost:
                best_matching_cost = current_cost
                best_matching_edges = current_matching_edges
                
        final_augmented_edges = [(w, u, v) for u, v, w in best_matching_edges]
        return final_augmented_edges, best_matching_cost

    def find_eulerian_circuit(self, augmented_edges):
        """
        Hierholzer's Algorithm for the Eulerian Circuit.
        """
        adj_counter = defaultdict(lambda: defaultdict(int))
        for u in range(self.V):
            for v, w in self.adj[u].items():
                adj_counter[u][v] += 1
        for w, u, v in augmented_edges:
            adj_counter[u][v] += 1
            adj_counter[v][u] += 1 

        start_node = 0
        if self.V > 0:
            try:
                start_node = next(i for i in range(self.V) if adj_counter[i])
            except StopIteration: return [], "Graph has no edges."

        circuit = []; stack = [start_node]
        while stack:
            u = stack[-1]
            v = None
            if adj_counter[u]:
                v = next(iter(adj_counter[u].keys()))
            if v is not None:
                adj_counter[u][v] -= 1
                if adj_counter[u][v] == 0: del adj_counter[u][v]
                adj_counter[v][u] -= 1
                if adj_counter[v][u] == 0: del adj_counter[v][u]
                stack.append(v)
            else:
                circuit.append(stack.pop())
        circuit.reverse()
        remaining_edges = sum(sum(adj_counter[u].values()) for u in range(self.V))
        if remaining_edges == 0 and len(circuit) > 1:
            return circuit, "Eulerian Circuit successfully found."
        elif remaining_edges > 0 and len(circuit) > 1:
             return circuit, "Semi-Eulerian Path found (Incomplete circuit)."
        else:
            return [], "Graph is disconnected or has no edges."

    def solve(self):
        odd_vertices = self.get_odd_degree_vertices()
        total_original_cost = sum(w for w, u, v in self.edges)
        if len(odd_vertices) == 0:
            min_matching_cost = 0; augmented_edges = []
        else:
            dist_dict = self.find_all_shortest_paths(odd_vertices)
            augmented_edges, min_matching_cost = self.find_min_matching(odd_vertices, dist_dict)
            if augmented_edges is None: return None, min_matching_cost 

        total_cost = total_original_cost + min_matching_cost
        euler_circuit, message = self.find_eulerian_circuit(augmented_edges)
        return {
            'odd_vertices': odd_vertices,
            'augmented_edges': augmented_edges, 
            'total_original_cost': total_original_cost,
            'min_matching_cost': min_matching_cost,
            'total_cost': total_cost,
            'circuit': euler_circuit,
            'message': message
        }

# =======================================================================
# SECTION 2: TKINTER UI (INTERACTIVE VISUALIZATION)
# =======================================================================

class CPPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chinese Postman Problem (CPP) Optimal Visualization")
        self.geometry("1100x850") 
        self.configure(bg="#2c3e50") 
        self.solver = CPPSolver()
        self.results = {}
        self.is_animating = False
        self.animation_job = None
        self.animation_delay_ms = ANIMATION_DELAY_MS 
        self.postman_position = -1 
        self.route_step_index = -1 
        self.traversed_edges_list = [] 
        self.setup_ui()
        self.solver.generate_random_graph(NUM_CITIES)
        self.run_solver()

    def setup_ui(self):
        main_container = tk.Frame(self, bg="#2c3e50")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        right_control_frame = tk.Frame(main_container, bg="#34495e", padx=15, pady=15)
        right_control_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        tk.Label(right_control_frame, text="ANALYSIS & ROUTE CONTROL", font=("Helvetica", 14, "bold"), fg="#ecf0f1", bg="#34495e").pack(pady=(0, 10))
        control_frame = tk.Frame(right_control_frame, bg="#34495e")
        control_frame.pack(fill='x', pady=10)
        self.new_graph_button = tk.Button(control_frame, text="🎲 Generate New Graph & Solve", command=self.generate_and_run, bg="#f39c12", fg="white", activebackground="#e67e22", relief=tk.FLAT)
        self.new_graph_button.pack(fill='x', padx=(0, 5), pady=5)
        tk.Label(control_frame, text="ROUTE CONTROL:", fg="#bdc3c7", bg="#34495e").pack(anchor='w', pady=(10, 0))
        manual_frame = tk.Frame(control_frame, bg="#34495e")
        manual_frame.pack(fill='x')
        self.step_button = tk.Button(manual_frame, text="➡️ Next Route Step", command=lambda: self.next_route_step(manual=True), state=tk.DISABLED, bg="#3498db", fg="white", activebackground="#2980b9", relief=tk.FLAT)
        self.step_button.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 2), pady=5)
        self.auto_button = tk.Button(manual_frame, text="▶️ Auto Run", command=self.toggle_auto_run, state=tk.DISABLED, bg="#1abc9c", fg="white", activebackground="#16a085", relief=tk.FLAT)
        self.auto_button.pack(side=tk.LEFT, fill='x', expand=True, padx=(2, 0), pady=5)
        self.stop_button = tk.Button(control_frame, text="⏹️ Stop & Reset Route", command=self.stop_animation, state=tk.DISABLED, bg="#e74c3c", fg="white", activebackground="#c0392b", relief=tk.FLAT)
        self.stop_button.pack(fill='x', padx=(0, 5), pady=5)
        tk.Label(right_control_frame, text="ROUTE LOG (Eulerian Circuit)", fg="#bdc3c7", bg="#34495e").pack(anchor='w', pady=(15, 0))
        self.route_log_text = tk.Text(right_control_frame, height=8, width=40, font=("Courier", 10), bg="#34495e", fg="#ecf0f1", wrap=tk.WORD)
        self.route_log_text.pack(pady=5)
        tk.Label(right_control_frame, text="STATUS & ANALYSIS RESULTS", fg="#bdc3c7", bg="#34495e").pack(anchor='w', pady=(5, 0))
        self.steps_text = tk.Text(right_control_frame, height=10, width=40, font=("Courier", 10), bg="#34495e", fg="#ecf0f1", wrap=tk.WORD)
        self.steps_text.pack(pady=5)
        tk.Label(right_control_frame, text="OPTIMAL ROUTE COST:", fg="#bdc3c7", bg="#34495e").pack(anchor='w')
        self.current_cost_label = tk.Label(right_control_frame, text="0", fg="#2ecc71", bg="#34495e", font=("Helvetica", 24, "bold"))
        self.current_cost_label.pack(anchor='w', pady=(0, 10))
        self.show_final_button = tk.Button(right_control_frame, text="🖼️ Show Before/After Graph", command=self.show_final_graph_popup, state=tk.DISABLED, bg="#3498db", fg="white", activebackground="#2980b9", relief=tk.FLAT)
        self.show_final_button.pack(fill='x', pady=5)
        left_frame = tk.Frame(main_container, bg="#34495e", padx=15, pady=15)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        tk.Label(left_frame, text="ACTIVE GRAPH (Analysis & Postman Route)", font=("Helvetica", 16, "bold"), fg="#ecf0f1", bg="#34495e").pack(pady=(0, 10))
        self.process_canvas = tk.Canvas(left_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#ecf0f1", bd=0, highlightthickness=0)
        self.process_canvas.pack(pady=10)

    def run_solver(self):
        try:
            self.results = self.solver.solve()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run CPP Solver: {e}")
            self.results = {'circuit': [], 'odd_vertices': [], 'augmented_edges': [], 'total_cost': 0}
            return
        if self.results is None or not self.results.get('circuit'):
             self.results = {'circuit': [], 'odd_vertices': self.solver.get_odd_degree_vertices(), 'augmented_edges': [], 'total_cost': 0}
        self.stop_animation(reset_controls=False) 
        self.postman_position = self.results['circuit'][0] if self.results['circuit'] else -1
        self.route_step_index = 0
        self.traversed_edges_list = []
        self.update_route_log(initial=True) 
        self.draw_graph_step() 
        self.update_info_panel()
        if self.results.get('circuit') and len(self.results['circuit']) > 1:
             self.step_button.config(state=tk.NORMAL)
             self.auto_button.config(state=tk.NORMAL)
        self.show_final_button.config(state=tk.NORMAL)
        self.current_cost_label.config(text=str(int(self.results['total_cost'])))

    def generate_and_run(self):
        self.stop_animation()
        self.solver.generate_random_graph(NUM_CITIES)
        self.run_solver()
        
    def update_route_log(self, initial=False):
        circuit = self.results.get('circuit')
        if initial:
            if circuit:
                 self.route_log_text.delete(1.0, tk.END)
                 self.route_log_text.insert(tk.END, f"Route: {circuit[0]}")
            return
        if self.route_step_index > 0 and self.route_step_index < len(circuit):
            new_node = circuit[self.route_step_index]
            current_log = self.route_log_text.get(1.0, tk.END).strip()
            self.route_log_text.delete(1.0, tk.END)
            self.route_log_text.insert(tk.END, f"{current_log} -> {new_node}")
        
    def update_info_panel(self):
        odd_v_names = [self.solver.city_names[i] for i in self.results['odd_vertices']]
        text = "========================================\n"
        text += f"       CPP ANALYSIS - {self.solver.V} CITIES\n"
        text += "========================================\n"
        text += f"1. PROBLEM IDENTIFICATION:\n   Odd Vertices: {', '.join(odd_v_names) if odd_v_names else 'None'}\n"
        text += f"2. OPTIMIZATION:\n   Optimal Duplication Cost: {int(self.results['min_matching_cost'])}\n"
        text += f"\n3. ROUTE RESULTS:\n   Optimal Total Route Cost: {int(self.results['total_cost'])}\n"
        self.steps_text.delete(1.0, tk.END); self.steps_text.insert(tk.END, text)

    def toggle_auto_run(self):
        if self.is_animating:
            self.is_animating = False
            self.auto_button.config(text="▶️ Auto Run", bg="#1abc9c")
            if self.animation_job: self.after_cancel(self.animation_job)
        else:
            self.is_animating = True
            self.auto_button.config(text="⏸️ Pause Auto", bg="#e67e22")
            self.animate_route()
    
    def stop_animation(self, reset_controls=True):
        if self.animation_job: self.after_cancel(self.animation_job)
        self.is_animating = False
        self.traversed_edges_list = []
        self.route_step_index = 0
        if reset_controls:
            self.postman_position = self.results['circuit'][0] if self.results['circuit'] else -1
            self.auto_button.config(text="▶️ Auto Run", bg="#1abc9c")
            self.stop_button.config(state=tk.DISABLED)
            self.update_route_log(initial=True)
        self.draw_graph_step() 

    def next_route_step(self, manual=False):
        circuit = self.results.get('circuit')
        if not circuit or self.route_step_index >= len(circuit) - 1:
            if manual: messagebox.showinfo("Finished", "The route has been completed.")
            self.stop_animation(reset_controls=True); return
        u, v = circuit[self.route_step_index], circuit[self.route_step_index + 1]
        self.traversed_edges_list.append((u, v))
        self.route_step_index += 1
        self.postman_position = v 
        self.update_route_log()
        self.draw_graph_step(current_edge=(u, v)) 
        if self.is_animating and not manual: self.animation_job = self.after(self.animation_delay_ms, self.next_route_step)
            
    def animate_route(self):
        if self.is_animating: self.next_route_step()

    def draw_graph_step(self, current_edge=None):
        self.process_canvas.delete("all")
        odd_vertices = self.results.get('odd_vertices', [])
        augmented_edges = self.results.get('augmented_edges', [])
        for w_all, u_all, v_all in self.solver.edges:
            x1, y1 = self.solver.node_positions[u_all]; x2, y2 = self.solver.node_positions[v_all]
            edge_count = self.traversed_edges_list.count((u_all, v_all)) + self.traversed_edges_list.count((v_all, u_all))
            color = COLOR_TRAVERSED if edge_count >= 1 else COLOR_DEFAULT
            width = 2 if edge_count >= 1 else 1
            if current_edge and tuple(sorted((u_all, v_all))) == tuple(sorted(current_edge)):
                 color, width = COLOR_CURRENT, 4
            self.process_canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            self.process_canvas.create_rectangle(mid_x-10, mid_y-8, mid_x+10, mid_y+8, fill="#ecf0f1", outline=color if color != COLOR_DEFAULT else "#34495e")
            self.process_canvas.create_text(mid_x, mid_y, text=str(w_all), fill=color if color != COLOR_DEFAULT else "#34495e", font=("Helvetica", 9, "bold"))
        for w_aug, u_aug, v_aug in augmented_edges:
            x1, y1 = self.solver.node_positions[u_aug]; x2, y2 = self.solver.node_positions[v_aug]
            angle = math.atan2(y2 - y1, x2 - x1); offset = 6
            x1_o, y1_o = x1 - offset * math.sin(angle), y1 + offset * math.cos(angle)
            x2_o, y2_o = x2 - offset * math.sin(angle), y2 + offset * math.cos(angle)
            color = COLOR_ODD_MATCH
            if current_edge and tuple(sorted((u_aug, v_aug))) == tuple(sorted(current_edge)): color = COLOR_CURRENT
            self.process_canvas.create_line(x1_o, y1_o, x2_o, y2_o, fill=color, width=2, dash=(4, 2))
        for index, name in enumerate(self.solver.city_names):
            x, y = self.solver.node_positions[index]
            fill_color = COLOR_ODD_MATCH if index in odd_vertices else "#3498db"
            if self.postman_position == index: fill_color = "black"
            self.process_canvas.create_oval(x-NODE_RADIUS, y-NODE_RADIUS, x+NODE_RADIUS, y+NODE_RADIUS, fill=fill_color, outline="white", width=2)
            self.process_canvas.create_text(x, y, text=name, fill="white", font=("Helvetica", 10, "bold"))

    def show_final_graph_popup(self):
        popup = tk.Toplevel(self)
        popup.title(f"CPP Graph Comparison | Optimal Total Cost: {int(self.results['total_cost'])}")
        popup.geometry(f"{CANVAS_WIDTH*2 + 80}x{CANVAS_HEIGHT + 100}")
        popup.configure(bg="#34495e")
        container = tk.Frame(popup, bg="#34495e", padx=10, pady=10); container.pack(fill=tk.BOTH, expand=True)
        original_frame = tk.Frame(container, bg="#34495e"); original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(original_frame, text="BEFORE (Original)", font=("Helvetica", 12, "bold"), fg="#f39c12", bg="#34495e").pack(pady=5)
        original_canvas = tk.Canvas(original_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#ecf0f1", bd=0); original_canvas.pack()
        eulerian_frame = tk.Frame(container, bg="#34495e"); eulerian_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        tk.Label(eulerian_frame, text="AFTER (Eulerian)", font=("Helvetica", 12, "bold"), fg="#2ecc71", bg="#34495e").pack(pady=5)
        eulerian_canvas = tk.Canvas(eulerian_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#ecf0f1", bd=0); eulerian_canvas.pack()
        self._draw_static_graph(original_canvas, self.solver.edges, self.results['odd_vertices'], [])
        self._draw_static_graph(eulerian_canvas, self.solver.edges, [], self.results['augmented_edges'])

    def _draw_static_graph(self, canvas, original_edges, odd_vertices, augmented_edges):
        for w, u, v in original_edges:
            x1, y1 = self.solver.node_positions[u]; x2, y2 = self.solver.node_positions[v]
            canvas.create_line(x1, y1, x2, y2, fill="#34495e", width=1)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            canvas.create_text(mid_x, mid_y, text=str(w), fill="#34495e", font=("Helvetica", 9, "bold"))
        for w, u, v in augmented_edges:
            x1, y1 = self.solver.node_positions[u]; x2, y2 = self.solver.node_positions[v]
            angle = math.atan2(y2 - y1, x2 - x1); offset = 6
            x1_o, y1_o = x1 - offset * math.sin(angle), y1 + offset * math.cos(angle)
            x2_o, y2_o = x2 - offset * math.sin(angle), y2 + offset * math.cos(angle)
            canvas.create_line(x1_o, y1_o, x2_o, y2_o, fill=COLOR_ODD_MATCH, width=3, dash=(4, 2))
        for index, name in enumerate(self.solver.city_names):
            x, y = self.solver.node_positions[index]
            fill_color = COLOR_ODD_MATCH if index in odd_vertices else "#3498db"
            canvas.create_oval(x-NODE_RADIUS, y-NODE_RADIUS, x+NODE_RADIUS, y+NODE_RADIUS, fill=fill_color, outline="white", width=2)
            canvas.create_text(x, y, text=name, fill="white", font=("Helvetica", 10, "bold"))

if __name__ == "__main__":
    app = CPPApp(); app.mainloop()