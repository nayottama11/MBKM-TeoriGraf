import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx
import random
import heapq
import string
import math

class PrimsVisualizer:
    def __init__(self):
        # Setup the plot window
        self.fig = plt.figure(figsize=(16, 8)) 
        self.fig.canvas.manager.set_window_title("Prim's Algorithm - Step by Step")
        
        # State variables
        self.G = None
        self.pos = None
        self.mst_edges = []
        self.start_node = 'A'
        self.nodes_list = sorted(list(string.ascii_uppercase[:6])) # A, B, C, D, E, F
        self.algo_generator = None
        self.total_cost = 0
        self.finished = False

        # Start the application
        self.generate_new_graph(None)
        plt.show()

    def setup_running_layout(self):
        """Sets up the layout for the interactive step-by-step phase."""
        self.fig.clear()
        
        # 1. Main Graph Area (Left)
        self.ax_graph = self.fig.add_axes([0.05, 0.15, 0.70, 0.8]) 
        
        # 2. Cost Log Area (Right)
        self.ax_log = self.fig.add_axes([0.80, 0.15, 0.15, 0.8])
        self.ax_log.axis('off')

        # 3. Controls
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_node_click)

        # Button: New Graph
        ax_new = plt.axes([0.25, 0.03, 0.15, 0.06])
        self.btn_new = Button(ax_new, 'New Random Graph')
        self.btn_new.on_clicked(self.generate_new_graph)

        # Button: Next Step
        ax_step = plt.axes([0.45, 0.03, 0.15, 0.06])
        self.btn_step = Button(ax_step, 'Next Step')
        self.btn_step.on_clicked(self.next_step)

    def generate_new_graph(self, event):
        """Creates a new graph with randomized positions."""
        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes_list)
        
        # 1. Create a random path to ensure connectivity
        # This acts as the "spine" of the graph so no node is left behind
        shuffled = self.nodes_list[:]
        random.shuffle(shuffled)
        for i in range(len(shuffled) - 1):
            self.G.add_edge(shuffled[i], shuffled[i+1], weight=random.randint(10, 99))

        # 2. Add Random Extra Edges (Cycles)
        # Limit to 4 extra edges to keep it clean but interesting
        num_extra = 4
        attempts = 0
        while num_extra > 0 and attempts < 50:
            u, v = random.sample(self.nodes_list, 2)
            if u != v and not self.G.has_edge(u, v):
                self.G.add_edge(u, v, weight=random.randint(10, 99))
                num_extra -= 1
            attempts += 1
        
        # 3. Position Layout - SPRING (Randomized Physics)
        # k=2.0 pushes nodes apart aggressively to prevent label overlap
        # iterations=100 gives the physics engine time to untangle knots
        self.pos = nx.spring_layout(self.G, seed=random.randint(1, 10000), k=2.0, iterations=100)
        
        # Reset everything
        self.setup_running_layout()
        self.reset_algorithm()

    def on_node_click(self, event):
        """Detects clicks to change start node (only in running mode)."""
        if self.finished: return 
        if event.inaxes != self.ax_graph: return

        if self.pos:
            for node, (x, y) in self.pos.items():
                distance = math.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
                if distance < 0.15:
                    self.start_node = node
                    self.reset_algorithm()
                    return

    def reset_algorithm(self):
        self.mst_edges = []
        self.total_cost = 0
        self.finished = False
        self.algo_generator = self.prims_logic_generator()
        self.draw_running_view(title=f"Start Node: {self.start_node} (Click any node to change)")

    def prims_logic_generator(self):
        visited = set()
        min_heap = [(0, self.start_node, None)]
        
        while min_heap:
            cost, current_node, prev_node = heapq.heappop(min_heap)
            if current_node in visited: continue
            visited.add(current_node)

            if prev_node is not None:
                self.mst_edges.append((prev_node, current_node, cost))
                self.total_cost += cost
                yield f"Connected {prev_node}-{current_node} (Cost: {cost})"
            else:
                yield f"Started at Node {self.start_node}"

            for neighbor in self.G.neighbors(current_node):
                if neighbor not in visited:
                    weight = self.G[current_node][neighbor]['weight']
                    heapq.heappush(min_heap, (weight, neighbor, current_node))

    def next_step(self, event):
        if self.finished: return

        try:
            message = next(self.algo_generator)
            self.draw_running_view(title=message)
        except StopIteration:
            self.finished = True
            self.show_comparison_view()

    def draw_running_view(self, title):
        """Draws the standard step-by-step view."""
        self.ax_graph.clear()
        self.ax_log.clear()
        self.ax_log.axis('off')
        
        # 1. Edges
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax_graph, width=1, alpha=0.4, edge_color='gray', style='dashed')
        mst_lines = [(u, v) for u, v, w in self.mst_edges]
        if mst_lines:
            nx.draw_networkx_edges(self.G, self.pos, ax=self.ax_graph, edgelist=mst_lines, width=4, edge_color='#2ecc71')
        
        # 2. Nodes
        visited_nodes = {n for u, v, w in self.mst_edges for n in (u, v)}
        if self.start_node not in visited_nodes: visited_nodes.add(self.start_node)
        
        node_colors = []
        for n in self.G.nodes():
            if n == self.start_node: node_colors.append('#f39c12')
            elif n in visited_nodes: node_colors.append('#2ecc71')
            else: node_colors.append('#3498db')

        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax_graph, node_size=1000, node_color=node_colors, edgecolors='black')
        
        # 3. Labels
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax_graph, font_size=14, font_family='sans-serif', font_color='white', font_weight='bold')
        all_weights = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, ax=self.ax_graph, edge_labels=all_weights, font_color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'), font_size=11)

        self.ax_graph.set_title(title, fontsize=16, fontweight='bold', pad=20)
        self.ax_graph.axis('off')

        # 4. Log
        log_text = "COST LOG\n" + "="*15 + "\n"
        for u, v, w in self.mst_edges:
            log_text += f"{u} -- {v} : {w}\n"
        log_text += "-"*15 + "\n"
        log_text += f"Sum: {self.total_cost}"
        self.ax_log.text(0, 1.0, log_text, transform=self.ax_log.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        self.fig.canvas.draw()

    def show_comparison_view(self):
        """Switch to the final comparison layout."""
        self.fig.clear()
        
        # --- LEFT SUBPLOT: Original Network ---
        ax_orig = self.fig.add_subplot(1, 2, 1)
        ax_orig.set_title("Original Network (All Possible Routes)", fontsize=14, fontweight='bold')
        
        nx.draw_networkx_edges(self.G, self.pos, ax=ax_orig, width=1, alpha=0.6, edge_color='gray', style='dashed')
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax_orig, node_size=800, node_color='lightblue', edgecolors='black')
        nx.draw_networkx_labels(self.G, self.pos, ax=ax_orig, font_size=12, font_family='sans-serif')
        all_weights = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, ax=ax_orig, edge_labels=all_weights, font_size=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        ax_orig.axis('off')

        # --- RIGHT SUBPLOT: MST Result ---
        ax_mst = self.fig.add_subplot(1, 2, 2)
        ax_mst.set_title(f"Minimum Spanning Tree (Total: {self.total_cost})", fontsize=14, fontweight='bold')
        
        mst_lines = [(u, v) for u, v, w in self.mst_edges]
        nx.draw_networkx_edges(self.G, self.pos, ax=ax_mst, edgelist=mst_lines, width=4, edge_color='#2ecc71')
        
        node_colors = []
        for n in self.G.nodes():
            if n == self.start_node: node_colors.append('#f39c12')
            else: node_colors.append('#90ee90') 
            
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax_mst, node_size=800, node_color=node_colors, edgecolors='black')
        nx.draw_networkx_labels(self.G, self.pos, ax=ax_mst, font_size=12, font_family='sans-serif')
        
        active_weights = {(u,v): w for u,v,w in self.mst_edges}
        nx.draw_networkx_edge_labels(self.G, self.pos, ax=ax_mst, edge_labels=active_weights, font_color='green', font_size=10, font_weight='bold', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.2'))
        ax_mst.axis('off')

        # --- Re-add 'New Graph' Button (centered at bottom) ---
        ax_new = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.btn_new = Button(ax_new, 'Generate New Graph')
        self.btn_new.on_clicked(self.generate_new_graph)

        self.fig.canvas.draw()

if __name__ == "__main__":
    try:
        app = PrimsVisualizer()
    except KeyboardInterrupt:
        print("Application closed.")