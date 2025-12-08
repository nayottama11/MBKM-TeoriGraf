"""
Interactive Hungarian Algorithm Visualization (Tkinter)
Complete runnable file — visualizes Hungarian Algorithm step-by-step.
Save as: hungarian_matching_tkinter.py
Run: python hungarian_matching_tkinter.py
Requires: Python 3 (no external packages)
FIX: Ensures user-edited matrix is used when running/stepping.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import copy
import itertools

# ------------------ Helpers ------------------

def copy_matrix(m):
    return [row[:] for row in m]

# Define colors for final visualization (Graph Coloring)
MATCH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# ------------------ Hungarian Algorithm (step-by-step generator) ------------------
class HungarianStepGenerator:
    def __init__(self, cost_matrix):
        # original matrix (for final cost calculation)
        self.orig = [row[:] for row in cost_matrix]
        self.n_rows = len(cost_matrix)
        self.n_cols = len(cost_matrix[0])
        self.n = max(self.n_rows, self.n_cols)

        # padded working matrix
        self.mat = [[0.0]*self.n for _ in range(self.n)]
        # Use a large number for padding (inf is safer but float input is complex, so a large multiple of max is used)
        large = 0
        if cost_matrix:
            large = max(max(row) for row in cost_matrix) * 1000 + 1
            if large == 1: large = 1000 # Handle 0 cost matrix case

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self.mat[i][j] = float(cost_matrix[i][j])
        
        # Fill padded cells
        for i in range(self.n_rows, self.n):
            for j in range(self.n):
                self.mat[i][j] = large
        for i in range(self.n):
            for j in range(self.n_cols, self.n):
                self.mat[i][j] = large

    def generate(self):
        mat = self.mat
        n = self.n
        
        yield {"phase":"start", "mat":copy_matrix(mat), 
               "explain":f"Initial matrix (size {self.n_rows}x{self.n_cols}) padded to {n}x{n} with a large cost."}

        # 1. Row reduction
        for i in range(n):
            rmin = min(mat[i])
            for j in range(n):
                mat[i][j] -= rmin
        yield {"phase":"row_reduction", "mat":copy_matrix(mat), "explain":"Step 1: Row reduction — subtract the minimum of each row."}

        # 2. Column reduction
        for j in range(n):
            cmin = min(mat[i][j] for i in range(n))
            for i in range(n):
                mat[i][j] -= cmin
        yield {"phase":"col_reduction", "mat":copy_matrix(mat), "explain":"Step 2: Column reduction — subtract the minimum of each column."}

        # ------------------ Main Loop (Covering and Adjustment) ------------------
        while True:
            zeros = [[mat[i][j] <= 1e-9 for j in range(n)] for i in range(n)] # Use tolerance for float comparison
            yield {"phase":"zeros_identified", "mat":copy_matrix(mat), "zeros":zeros, 
                   "explain":"Step 3: Identify current zeros."}

            # 3. Find maximum matching on zero graph (Using max flow/matching algorithm, approximated here by simple DFS matching)
            match_row = [-1]*n # match_row[r] = c
            
            def try_match(r, seen_cols):
                for c in range(n):
                    if zeros[r][c] and not seen_cols[c]:
                        seen_cols[c] = True
                        # If column c is unmatched, or its current match r_prev can be rematched
                        r_prev = next((i for i in range(n) if match_row[i] == c), -1)
                        if r_prev == -1 or try_match(r_prev, seen_cols):
                            match_row[r] = c
                            return True
                return False

            matched = 0
            # Need to clear previous matching results for each iteration
            match_row = [-1] * n 
            
            for r in range(n):
                if try_match(r, [False]*n):
                    matched += 1

            pairs = [(r, match_row[r]) for r in range(n) if match_row[r] != -1]
            yield {"phase":"matching_attempt", "mat":copy_matrix(mat), "pairs":pairs, "matched":matched,
                   "explain":f"Step 4: Attempt to find a zero assignment. Found {matched} pairs."}

            if matched == n:
                # Optimal assignment found
                assignment = []
                total_cost = 0.0
                for r, c in pairs:
                    if r < self.n_rows and c < self.n_cols:
                        cost = self.orig[r][c]
                        assignment.append((r, c, cost))
                        total_cost += cost
                yield {"phase":"optimal_found", "mat":copy_matrix(mat), "pairs":pairs, "assignment":assignment,
                       "total_cost":total_cost, "explain":"Optimal assignment found. Maximum matching on zeros achieved."}
                break

            # 4. Minimum line cover (Konig's theorem construction)
            # a. Mark all rows without a starred zero (unmatched rows)
            marked_r = [False]*n
            for r in range(n):
                if match_row[r] == -1:
                    marked_r[r] = True
            
            marked_c = [False]*n
            
            # b. Iterate: 
            #   Mark column c if it contains a zero in a marked row r.
            #   Mark row r' if it contains a starred zero in a marked column c.
            changed = True
            while changed:
                changed = False
                for r in range(n):
                    if marked_r[r]:
                        for c in range(n):
                            if zeros[r][c] and not marked_c[c]: # Zero in marked row
                                marked_c[c] = True
                                changed = True
                
                for c in range(n):
                    if marked_c[c]:
                        # Find the row r' that is matched to c
                        r_prime = next((r_i for r_i in range(n) if match_row[r_i] == c), -1)
                        if r_prime != -1 and not marked_r[r_prime]: # If column has a match and the row is unmarked
                            marked_r[r_prime] = True
                            changed = True

            # c. Cover rows that are UNMARKED and columns that are MARKED
            cover_rows = [i for i in range(n) if not marked_r[i]]
            cover_cols = [j for j in range(n) if marked_c[j]]
            
            yield {"phase":"min_cover", "mat":copy_matrix(mat), "cover_rows":cover_rows, "cover_cols":cover_cols,
                   "explain":"Step 5: Minimum line cover (Konig's Theorem). Lines cover all zeros."}

            # 5. Adjust matrix
            # Find minimum uncovered value
            uncovered_vals = [mat[i][j] for i in range(n) for j in range(n) 
                              if i not in cover_rows and j not in cover_cols]
            
            delta = min(uncovered_vals) if uncovered_vals else 0

            # Subtract delta from all uncovered elements
            for i in range(n):
                for j in range(n):
                    if i not in cover_rows and j not in cover_cols:
                        mat[i][j] -= delta
                    # Add delta to all elements covered twice (intersection)
                    elif i in cover_rows and j in cover_cols:
                        mat[i][j] += delta
            
            yield {"phase":"adjust_matrix", "mat":copy_matrix(mat), "delta":delta, "cover_rows":cover_rows,
                   "cover_cols":cover_cols, "explain":f"Step 6: Adjust matrix by $\delta={round(delta, 2)}$ to create new zeros."}

# ------------------ Tkinter GUI ------------------
class HungarianApp:
    def __init__(self, root):
        self.root = root
        root.title("Hungarian Algorithm Visualizer (Bipartite Assignment)")
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)

        # Default example (Bulldozer Distances)
        self.default_matrix = [
            [90, 75, 75, 80],
            [35, 85, 55, 65],
            [125, 95, 90, 105],
            [45, 110, 95, 115]
        ]
        self.size_var = tk.IntVar(value=len(self.default_matrix))

        self._setup_ui()
        self.build_entries()
        self.use_example()

    def _setup_ui(self):
        # Top controls
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.grid(row=0, column=0, sticky="ew")

        ttk.Label(ctrl, text="Matrix size (N):").grid(row=0, column=0, sticky="w", padx=5)
        self.size_spin = ttk.Spinbox(ctrl, from_=2, to=8, textvariable=self.size_var, width=4, command=self.build_entries)
        self.size_spin.grid(row=0, column=1, padx=6)

        ttk.Button(ctrl, text="Load Example", command=self.use_example).grid(row=0, column=2, padx=15)
        ttk.Button(ctrl, text="Run (Auto)", command=self.run_full).grid(row=0, column=3, padx=15)
        ttk.Button(ctrl, text="Next Step", command=self.next_step).grid(row=0, column=4, padx=15)
        ttk.Button(ctrl, text="Restart/Clear", command=self.reset).grid(row=0, column=5, padx=15)

        # Middle layout: matrix editor + canvas
        middle = ttk.Frame(self.root, padding=8)
        middle.grid(row=1, column=0, sticky="nsew")
        middle.columnconfigure(1, weight=1)
        
        self.matrix_frame = ttk.LabelFrame(middle, text="Cost Matrix Input (Bulldozers/Sites)")
        self.matrix_frame.grid(row=0, column=0, sticky="nw")

        canvas_frame = ttk.LabelFrame(middle, text="Step Visualization")
        canvas_frame.grid(row=0, column=1, sticky="nsew", padx=10)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, width=800, height=480, bg="#F5F5F5")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Bottom: explanations and summary
        bottom = ttk.Frame(self.root, padding=8)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self.explain_label = ttk.Label(bottom, text="Ready. Load example then use Next Step or Run.", font=('Helvetica', 10, 'italic'))
        self.explain_label.grid(row=0, column=0, sticky="w")

        self.summary_text = tk.Text(bottom, height=6, font=('Courier', 10))
        self.summary_text.grid(row=1, column=0, sticky="ew", pady=5)

        # internal state
        self.matrix_entries = []
        self.generator = None
        self.current_state = None
        self.user_matrix_input = [] # Store the initial read matrix for the generator

    def build_entries(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        
        size = self.size_var.get()
        self.matrix_entries = []
        
        # Add column labels
        for j in range(size):
            ttk.Label(self.matrix_frame, text=f"Site {j+1}", font=('Arial', 9, 'bold')).grid(row=0, column=j+1, padx=2, pady=2)

        for i in range(size):
            row = []
            # Add row labels
            ttk.Label(self.matrix_frame, text=f"B {i+1}", font=('Arial', 9, 'bold')).grid(row=i+1, column=0, padx=2, pady=2)
            for j in range(size):
                e = ttk.Entry(self.matrix_frame, width=6, justify='center')
                e.grid(row=i+1, column=j+1, padx=2, pady=2)
                row.append(e)
            self.matrix_entries.append(row)

    def use_example(self):
        m = self.default_matrix
        n = len(m)
        self.size_var.set(n)
        self.build_entries()
        for i in range(n):
            for j in range(n):
                self.matrix_entries[i][j].delete(0, tk.END)
                self.matrix_entries[i][j].insert(0, str(m[i][j]))
        
        self.reset()
        self.explain_label.config(text="Loaded default example. Click 'Next Step' to begin.")

    def read_matrix(self):
        size = self.size_var.get()
        mat = [[0.0]*size for _ in range(size)]
        
        # Use min(size, actual_rows) to handle size change before entries are updated
        num_rows = min(size, len(self.matrix_entries))
        num_cols = min(size, len(self.matrix_entries[0]) if num_rows > 0 else 0)

        try:
            for i in range(num_rows):
                for j in range(num_cols):
                    v = float(self.matrix_entries[i][j].get())
                    if v < 0:
                        raise ValueError("Negative not allowed")
                    mat[i][j] = v
        except Exception as e:
            messagebox.showerror("Invalid input", f"Please enter valid non-negative numbers. {e}")
            return None
        
        # Return only the N_rows x N_cols part for the generator initialization
        return [row[:num_cols] for row in mat[:num_rows]]


    def reset(self):
        self.generator = None
        self.current_state = None
        self.user_matrix_input = []
        self.summary_text.delete('1.0', tk.END)
        self.canvas.delete('all')
        self.explain_label.config(text="Reset. Edit matrix or load example.")

    def run_full(self):
        if self.generator is None:
            self.user_matrix_input = self.read_matrix()
            if self.user_matrix_input is None: return
            self.generator = HungarianStepGenerator(self.user_matrix_input).generate()
        
        try:
            while True:
                state = next(self.generator)
                self.apply_state(state)
                self.root.update()
            
        except StopIteration:
            self.generator = None
            self.explain_label.config(text="Algorithm Complete.", font=('Helvetica', 12, 'bold'), foreground='green')

    def next_step(self):
        if self.generator is None:
            # Initialize generator using current user input
            self.user_matrix_input = self.read_matrix()
            if self.user_matrix_input is None: return
            self.generator = HungarianStepGenerator(self.user_matrix_input).generate()
            
        try:
            state = next(self.generator)
            self.apply_state(state)
        except StopIteration:
            self.generator = None
            self.explain_label.config(text="Algorithm Complete.", font=('Helvetica', 12, 'bold'), foreground='green')

    def apply_state(self, state):
        self.current_state = state
        self.explain_label.config(text=state.get('explain',''))
        mat = state.get('mat')
        if mat is not None:
            self.draw_matrix_on_canvas(mat, state)
        
        if state.get('phase') == 'optimal_found':
            assign = state.get('assignment', [])
            total = state.get('total_cost', 0)
            
            txt = f"--- Optimal Matching Found ---\nTotal Minimum Cost: {total}\n\nAssignment Details (Cost, Color):\n"
            
            for i, (r, c, v) in enumerate(assign):
                color = MATCH_COLORS[i % len(MATCH_COLORS)]
                txt += f"Bulldozer {r+1} -> Site {c+1} (Cost {v}, Color: {color})\n"
            
            self.summary_text.delete('1.0', tk.END)
            self.summary_text.insert(tk.END, txt)

    def draw_matrix_on_canvas(self, mat, state):
        self.canvas.delete('all')
        n = len(mat)
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # Layout parameters
        grid_left = 40
        grid_top = 60
        cell = min(max((w-500)//n, 40), 90) # Adjust grid size based on canvas width
        
        # --- Draw Reduced Matrix ---
        self.canvas.create_text(grid_left + n*cell/2, 20, text='Reduced Cost Matrix', font=('Helvetica', 14, 'bold'))

        # Draw grid and numbers
        for i in range(n):
            for j in range(n):
                x = grid_left + j*cell
                y = grid_top + i*cell
                
                # Cell background (Highlight cells affected by adjustment)
                fill_color = 'white'
                if state.get('phase') == 'adjust_matrix':
                    cover_rows = state.get('cover_rows', [])
                    cover_cols = state.get('cover_cols', [])
                    if i in cover_rows and j in cover_cols:
                        fill_color = '#ADD8E6' # Covered twice
                    elif (i not in cover_rows and j not in cover_cols):
                        fill_color = '#FFB6C1' # Uncovered
                
                self.canvas.create_rectangle(x, y, x+cell, y+cell, fill=fill_color, outline='black')
                
                val = mat[i][j]
                # Format to integer if possible, else 2 decimal places
                txt = str(int(val) if abs(val - round(val)) < 1e-9 else round(val, 2)) 
                
                self.canvas.create_text(x+cell/2, y+cell/2, text=txt, font=('Helvetica', 11))

        # Labels for rows (Bulldozer) and columns (Site)
        for i in range(n):
            y = grid_top + i*cell + cell/2
            self.canvas.create_text(grid_left-10, y, text=f'B {i+1}', anchor='e', font=('Helvetica', 10, 'bold'))
        for j in range(n):
            x = grid_left + j*cell + cell/2
            self.canvas.create_text(x, grid_top-10, text=f'S {j+1}', font=('Helvetica', 10, 'bold'))

        # Highlight zeros
        zeros = state.get('zeros')
        if zeros:
            for i in range(n):
                for j in range(n):
                    if zeros[i][j]:
                        x = grid_left + j*cell
                        y = grid_top + i*cell
                        # Circle the zero
                        self.canvas.create_oval(x+cell*0.15, y+cell*0.15, x+cell*0.85, y+cell*0.85, outline='blue', width=2)

        # Draw cover lines (Step 5)
        cover_rows = state.get('cover_rows')
        cover_cols = state.get('cover_cols')
        line_offset = 5 # Offset for drawing line over cells
        if cover_rows:
            for i in cover_rows:
                y = grid_top + i*cell + cell/2
                self.canvas.create_line(grid_left - line_offset, y, grid_left + n*cell + line_offset, y, 
                                        fill='red', width=3, dash=(5, 5))
        if cover_cols:
            for j in cover_cols:
                x = grid_left + j*cell + cell/2
                self.canvas.create_line(x, grid_top - line_offset, x, grid_top + n*cell + line_offset, 
                                        fill='red', width=3, dash=(5, 5))

        # --- Draw Bipartite Graph for Matching ---
        
        # Determine the start position for the bipartite graph
        bip_start_x = grid_left + n*cell + 120
        bip_center_y = grid_top + n*cell / 2

        # Draw nodes
        node_radius = 15
        L_nodes, R_nodes = [], []
        
        # Left Set (Bulldozers)
        for i in range(n):
            ly = grid_top + i * cell + cell/2
            node = self.canvas.create_oval(bip_start_x - node_radius, ly - node_radius, 
                                           bip_start_x + node_radius, ly + node_radius, 
                                           fill='#ADD8E6', outline='blue', tags=f'BipL{i}')
            self.canvas.create_text(bip_start_x, ly, text=f'B{i+1}', font=('Arial', 10, 'bold'))
            L_nodes.append((bip_start_x, ly, node))
            
        # Right Set (Sites)
        bip_end_x = bip_start_x + 200
        for j in range(n):
            ry = grid_top + j * cell + cell/2
            node = self.canvas.create_oval(bip_end_x - node_radius, ry - node_radius, 
                                           bip_end_x + node_radius, ry + node_radius, 
                                           fill='#FFB6C1', outline='red', tags=f'BipR{j}')
            self.canvas.create_text(bip_end_x, ry, text=f'S{j+1}', font=('Arial', 10, 'bold'))
            R_nodes.append((bip_end_x, ry, node))
            
        # Draw Matched Pairs (Edges + Coloring)
        pairs = state.get('pairs')
        if pairs:
            for idx, (r, c) in enumerate(pairs):
                # Graph Coloring: Use a distinct color for each matched pair
                color = MATCH_COLORS[idx % len(MATCH_COLORS)]
                
                lx, ly, l_obj = L_nodes[r]
                rx, ry, r_obj = R_nodes[c]
                
                # Highlight the zero match in the matrix (optional)
                # No need to highlight matrix cell itself, done by circle
                
                # Draw the matching edge
                self.canvas.create_line(lx + node_radius, ly, rx - node_radius, ry, 
                                        width=3, fill=color, tags='MatchEdge')
                
                # Color the nodes
                self.canvas.itemconfig(l_obj, fill=color, outline='black', width=2)
                self.canvas.itemconfig(r_obj, fill=color, outline='black', width=2)


# ------------------ Run ------------------

def main():
    root = tk.Tk()
    root.geometry('1100x700')
    app = HungarianApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()