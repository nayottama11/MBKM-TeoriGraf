import tkinter as tk
from tkinter import ttk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import string

# --- THE LOGIC (Unchanged) ---
def welsh_powell(graph_adj_list):
    # Sort nodes by degree (descending)
    sorted_nodes = sorted(graph_adj_list.keys(), key=lambda x: len(graph_adj_list[x]), reverse=True)
    
    color_map = {}  
    current_color = 0
    
    while len(color_map) < len(graph_adj_list):
        nodes_colored_this_round = []
        for node in sorted_nodes:
            if node in color_map: continue
            
            is_conflicted = False
            for colored_node in nodes_colored_this_round:
                if colored_node in graph_adj_list[node]:
                    is_conflicted = True
                    break
            
            if not is_conflicted:
                color_map[node] = current_color
                nodes_colored_this_round.append(node)
        current_color += 1
        
    return color_map

# --- THE GUI APP ---
class GraphColoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Welsh-Powell Graph Coloring GUI")
        self.root.geometry("1100x650")

        # --- Data Containers ---
        self.current_G = None
        self.color_results = None
        self.pos = None

        # --- Layout Configuration ---
        left_frame = tk.Frame(root, width=300, bg="#f0f0f0", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        right_frame = tk.Frame(root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Left Panel Content ---
        btn_generate = tk.Button(left_frame, text="Generate Random Graph", 
                                 command=self.generate_and_update,
                                 bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), pady=10)
        btn_generate.pack(fill=tk.X, pady=(0, 20))

        # Table Setup
        columns = ("vertex", "degree", "color_id")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=15)
        
        style = ttk.Style()
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
        
        self.tree.heading("vertex", text="Vertex")
        self.tree.column("vertex", anchor=tk.CENTER, width=70)
        self.tree.heading("degree", text="Degree")
        self.tree.column("degree", anchor=tk.CENTER, width=70)
        self.tree.heading("color_id", text="Color ID")
        self.tree.column("color_id", anchor=tk.CENTER, width=70)
        
        self.tree.pack(fill=tk.X)
        
        self.chromatic_label = tk.Label(left_frame, text="Chromatic Number = N/A", 
                                        font=("Arial", 14, "bold"), bg="#f0f0f0", pady=20)
        self.chromatic_label.pack()

        # --- Right Panel Content ---
        self.figure = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.generate_and_update()

    def generate_and_update(self):
        # 1. Generate Graph
        num_nodes = 10
        num_edges = 20
        G_int = nx.gnm_random_graph(num_nodes, num_edges) # Removed seed here for random structure
        
        while nx.number_of_edges(G_int) < num_edges - 2:
             G_int = nx.gnm_random_graph(num_nodes, num_edges)

        mapping = {i: string.ascii_uppercase[i] for i in range(num_nodes)}
        self.current_G = nx.relabel_nodes(G_int, mapping)
        
        # 2. CALCULATE POSITIONS (The Fix)
        # k=0.8 pushes nodes apart (optimal distance). 
        # iterations=50 allows the simulation to run longer to find the best spot.
        # scale=2 spreads the entire graph out over a larger coordinate area.
        self.pos = nx.spring_layout(self.current_G, k=0.8, iterations=50, scale=2)
        
        adj_dict = nx.to_dict_of_lists(self.current_G)
        self.color_results = welsh_powell(adj_dict)
        
        self.update_table(adj_dict)
        self.draw_graph()

    def update_table(self, adj_dict):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        table_data = []
        for node, neighbors in adj_dict.items():
            degree = len(neighbors)
            color_id = self.color_results[node]
            table_data.append((node, degree, color_id))
            
        table_data.sort(key=lambda x: x[1], reverse=True)
        
        for item in table_data:
            self.tree.insert("", tk.END, values=item)

        chrom_num = max(self.color_results.values()) + 1
        self.chromatic_label.config(text=f"Chromatic Number = {chrom_num}")

    def draw_graph(self):
        self.ax.clear()
        
        node_colors = [self.color_results[node] for node in self.current_G.nodes()]
        
        # Draw with the new positions (self.pos)
        nx.draw(self.current_G, self.pos, ax=self.ax,
                with_labels=True,
                node_color=node_colors,
                cmap=plt.cm.Set3, 
                node_size=1200,
                font_weight='bold',
                edge_color='gray', 
                width=1.5)
        
        self.ax.set_title("Graph Visualization", fontsize=14)
        self.ax.axis('off')
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphColoringApp(root)
    root.mainloop()