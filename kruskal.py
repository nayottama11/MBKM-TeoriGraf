import tkinter as tk
from tkinter import messagebox
from collections import defaultdict
import random
import time
import math # Diperlukan untuk perhitungan trigonometri (penempatan melingkar)

# Konstanta UI
NODE_RADIUS = 12
CANVAS_WIDTH = 700 # Tetap 700x700
CANVAS_HEIGHT = 700
NUM_CITIES = 8 # Jumlah kota yang akan dibuat secara acak

# =======================================================================
# BAGIAN 1: IMPLEMENTASI ALGORITMA KRUSKAL
# =======================================================================

class UnionFind:
    """Disjoint Set Union (DSU) untuk deteksi siklus."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

class KruskalSolver:
    """Menyimpan data graf dan menjalankan algoritma Kruskal."""
    def __init__(self):
        self.city_names = []
        self.edges = []  # Format: (bobot, simpul_u, simpul_v)
        self.node_positions = {} # {city_index: (x, y)} untuk visualisasi

    def generate_random_graph(self, num_cities, max_weight=100, edge_probability=0.3):
        """Membuat kota dan sisi secara acak, menggunakan penempatan melingkar."""
        self.city_names = [f"Kota {i+1}" for i in range(num_cities)]
        self.edges = []
        self.node_positions = {}
        V = num_cities

        # 1. Posisi Melingkar untuk penyebaran yang seragam dan bersih
        center_x = CANVAS_WIDTH / 2
        center_y = CANVAS_HEIGHT / 2
        # Radius, disesuaikan agar simpul tidak terlalu dekat dengan tepi kanvas
        radius = min(CANVAS_WIDTH, CANVAS_HEIGHT) / 2 - 50 

        for i in range(V):
            # Hitung sudut untuk penempatan melingkar
            angle = 2 * math.pi * i / V
            
            # Hitung posisi (x, y) menggunakan trigonometri
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            self.node_positions[i] = (x, y)
        
        # 2. Sisi Acak
        for u in range(V):
            for v in range(u + 1, V):
                if random.random() < edge_probability: 
                    weight = random.randint(1, max_weight)
                    self.edges.append((weight, u, v)) # Format: (w, u, v)
        
        # Pastikan graf terhubung (minimal)
        if not self.edges and V > 1:
            u, v = random.sample(range(V), 2)
            weight = random.randint(1, max_weight)
            self.edges.append((weight, u, v))


    def kruskal_mst_stepwise(self):
        """
        Menjalankan Kruskal's Algorithm dan merekam setiap langkah untuk visualisasi.
        Returns: (final_mst, total_cost, message, steps)
        """
        V = len(self.city_names)
        if V < 2:
            return [], 0, "Minimal 2 kota diperlukan.", []

        # 1. Sortir Sisi
        sorted_edges = sorted(self.edges) # Format: (w, u, v)

        # 2. Inisialisasi DSU dan Variabel
        uf = UnionFind(V) 
        result_mst = [] # Format: (w, u, v)
        total_cost = 0
        edges_in_mst = 0
        steps = [] # Untuk menyimpan histori visualisasi

        # 3. Iterasi dan Bangun MST
        for w, u, v in sorted_edges: # Unpack: (w, u, v)
            
            # Simpan state sebelum mencoba sisi
            step_data = {
                # Sisi yang sedang diproses. Simpan dalam format (u, v, w) untuk memudahkan display
                'edge': (u, v, w), 
                'action': 'REJECT', 
                'current_cost': total_cost,
                # Salinan MST saat ini (berisi (w, u, v))
                'mst_so_far': list(result_mst) 
            }
            
            if uf.find(u) != uf.find(v):
                # Sisi ini tidak membuat siklus
                uf.union(u, v)
                # Simpan MST dalam format (w, u, v)
                result_mst.append((w, u, v)) 
                total_cost += w
                edges_in_mst += 1
                step_data['action'] = 'ADD'
                # Perbarui state setelah penambahan
                step_data['current_cost'] = total_cost
                step_data['mst_so_far'] = list(result_mst)

            steps.append(step_data)
            
            if edges_in_mst == V - 1:
                break
            
        # 4. Verifikasi Konektivitas Akhir
        if V > 0 and edges_in_mst < V - 1:
            message = "MST ditemukan, tetapi graf mungkin tidak terhubung sepenuhnya."
        else:
            message = "MST berhasil ditemukan."
            
        return result_mst, total_cost, message, steps
    
    def reset(self):
        """Mereset semua data kota dan rute."""
        self.city_names = []
        self.edges = []
        self.node_positions = {}


# =======================================================================
# BAGIAN 2: APLIKASI TKINTER (VISUALISASI)
# =======================================================================

class KruskalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kruskal MST Visualisasi (Simulasi Jaringan Minimal)")
        # Ukuran disesuaikan untuk kanvas yang lebih besar
        self.geometry("1000x750") 
        self.configure(bg="#2c3e50") 
        
        self.solver = KruskalSolver()
        
        # State Visualisasi
        self.steps = []
        self.final_mst = []
        self.final_cost = 0
        self.current_step_index = -1
        self.is_animating = False
        self.animation_job = None
        self.animation_delay_ms = 800 # 0.8 detik per langkah
        
        self.setup_ui()
        self.setup_initial_data()

    def setup_ui(self):
        """Mengatur tata letak dan widget utama."""
        # Frame Utama (Dua Kolom)
        main_container = tk.Frame(self, bg="#2c3e50")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ------------------- KOLOM KIRI: VISUALISASI PROSES -------------------
        left_frame = tk.Frame(main_container, bg="#34495e", padx=15, pady=15)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(left_frame, text="PROSES ALGORITMA KRUSKAL", font=("Helvetica", 16, "bold"), fg="#ecf0f1", bg="#34495e").pack(pady=(0, 10))

        # Canvas Visualisasi Proses
        self.process_canvas = tk.Canvas(left_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#ecf0f1", bd=0, highlightthickness=0)
        self.process_canvas.pack(pady=10)
        self.process_canvas.bind("<Button-1>", self.start_drag)
        self.process_canvas.bind("<B1-Motion>", self.drag)
        self.process_canvas.bind("<ButtonRelease-1>", self.stop_drag)
        
        # ------------------- KOLOM KANAN: KONTROL DAN HASIL -------------------
        right_control_frame = tk.Frame(main_container, bg="#34495e", padx=15, pady=15)
        right_control_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        
        tk.Label(right_control_frame, text="KONTROL & HASIL", font=("Helvetica", 14, "bold"), fg="#ecf0f1", bg="#34495e").pack(pady=(0, 10))
        
        # --- Kontrol Visualisasi ---
        control_frame = tk.Frame(right_control_frame, bg="#34495e")
        control_frame.pack(fill='x', pady=5)

        self.step_button = tk.Button(control_frame, text="➡️ Langkah Berikutnya", command=self.next_step, state=tk.DISABLED, bg="#3498db", fg="white", activebackground="#2980b9", relief=tk.FLAT)
        self.step_button.pack(fill='x', padx=(0, 5), pady=5)

        self.auto_step_button = tk.Button(control_frame, text="⚡ Mode Otomatis", command=self.toggle_auto_step, state=tk.DISABLED, bg="#1abc9c", fg="white", activebackground="#16a085", relief=tk.FLAT)
        self.auto_step_button.pack(fill='x', padx=(0, 5), pady=5)
        
        # --- Info Langkah ---
        self.step_info_label = tk.Label(right_control_frame, text="Siap membuat graf acak baru.", fg="#ecf0f1", bg="#34495e", justify=tk.LEFT, wraplength=200)
        self.step_info_label.pack(fill='x', anchor='w', pady=(10, 5))

        # --- Biaya Saat Ini ---
        tk.Label(right_control_frame, text="Biaya MST Saat Ini:", fg="#bdc3c7", bg="#34495e").pack(anchor='w')
        self.current_cost_label = tk.Label(right_control_frame, text="0", fg="#2ecc71", bg="#34495e", font=("Helvetica", 18, "bold"))
        self.current_cost_label.pack(anchor='w', pady=(0, 20))
        
        # --- Hasil Akhir ---
        result_frame = tk.Frame(right_control_frame, bg="#2c3e50", padx=10, pady=10)
        result_frame.pack(fill='x', pady=(20, 5))
        
        tk.Label(result_frame, text="BIAYA MST TOTAL:", fg="#bdc3c7", bg="#2c3e50").pack(anchor='w')
        self.result_cost_label = tk.Label(result_frame, text="0", fg="#f39c12", bg="#2c3e50", font=("Helvetica", 24, "bold"))
        self.result_cost_label.pack(anchor='w', pady=(0, 5))
        
        self.show_final_button = tk.Button(result_frame, text="🖼️ Tampilkan Graf Akhir", command=self.show_final_graph_popup, state=tk.DISABLED, bg="#2ecc71", fg="white", activebackground="#27ae60", relief=tk.FLAT, font=("Helvetica", 10, "bold"))
        self.show_final_button.pack(fill='x', pady=5)
        
        tk.Button(right_control_frame, text="🎲 Buat Graf Acak Baru", command=self.generate_and_run, bg="#f39c12", fg="white", activebackground="#e67e22", relief=tk.FLAT, font=("Helvetica", 10, "bold")).pack(fill='x', pady=(15, 5))


    # --- FUNGSI DATA & AKSI ---
    def setup_initial_data(self):
        """Menyiapkan data acak awal dan menjalankan Kruskal."""
        self.solver.generate_random_graph(NUM_CITIES)
        self.run_kruskal(initial_run=True)

    def generate_and_run(self):
        """Membuat graf acak baru, mereset visualisasi, dan menjalankan Kruskal."""
        if self.is_animating:
            self.toggle_auto_step()
            
        self.solver.generate_random_graph(NUM_CITIES)
        self.reset_visualization_state()
        self.run_kruskal()

    def run_kruskal(self, initial_run=False):
        """Menjalankan algoritma Kruskal, merekam langkah, dan menyiapkan visualisasi."""
        
        # Dapatkan semua langkah visualisasi
        final_mst, total_cost, message, self.steps = self.solver.kruskal_mst_stepwise()
        self.final_mst = final_mst # Format: (w, u, v)
        self.final_cost = total_cost
        
        if not self.steps and len(self.solver.city_names) > 1 and len(self.solver.edges) == 0:
            messagebox.showwarning("Peringatan", "Graf yang dihasilkan tidak memiliki sisi. Coba buat graf baru.")
            return

        if not self.steps and len(self.solver.city_names) > 1:
            messagebox.showerror("Error", "Gagal menghitung langkah Kruskal.")
            return

        # Siapkan UI untuk visualisasi
        self.current_step_index = -1
        self.update_results_final(total_cost, message)
        self.draw_graph_step() # Gambar state awal
        
        # Aktifkan kontrol visualisasi
        self.step_button.config(state=tk.NORMAL)
        self.auto_step_button.config(state=tk.NORMAL)
        self.show_final_button.config(state=tk.DISABLED) # Nonaktifkan sampai selesai
        
        if not initial_run:
             messagebox.showinfo("Siap", f"Algoritma Kruskal siap dengan {len(self.steps)} langkah. Klik 'Langkah Berikutnya' atau 'Mode Otomatis'.")


    def reset_visualization_state(self):
        """Mereset variabel visualisasi."""
        if self.is_animating:
            self.toggle_auto_step()
            
        self.current_step_index = -1
        self.steps = []
        self.current_cost_label.config(text="0")
        self.step_info_label.config(text="Graf acak baru siap. Klik 'Langkah Berikutnya' untuk memulai.")
        self.step_button.config(state=tk.DISABLED)
        self.auto_step_button.config(state=tk.DISABLED)
        self.show_final_button.config(state=tk.DISABLED)

    def update_results_final(self, total_cost=0, message=""):
        """Memperbarui label biaya hasil akhir."""
        self.result_cost_label.config(text=str(total_cost))
        
        if total_cost > 0:
            self.step_info_label.config(text=f"Siap visualisasi Kruskal. Total Sisi: {len(self.solver.edges)}. {message}")
        else:
             self.step_info_label.config(text="Graf acak baru telah dibuat. Klik 'Langkah Berikutnya' untuk memulai visualisasi.")
             
    def update_step_info(self, step):
        """Memperbarui informasi di panel kontrol untuk langkah saat ini."""
        u, v, w = step['edge']
        action = step['action']
        cost = step['current_cost']
        
        action_text = "DITAMBAHKAN ke MST (tidak ada siklus)" if action == 'ADD' else "DITOLAK (membuat siklus)"
        action_color = "#2ecc71" if action == 'ADD' else "#e74c3c"
        
        status_text = f"Langkah {self.current_step_index + 1}/{len(self.steps)} (Bobot: {w})\n"
        status_text += f"Memproses: {self.solver.city_names[u]} <-> {self.solver.city_names[v]}\n"
        status_text += f"Aksi: {action_text}"
        
        self.step_info_label.config(text=status_text)
        self.current_cost_label.config(text=str(cost), fg=action_color)

    # --- FUNGSI VISUALISASI KONTROL ---
    def next_step(self):
        """Pindah ke langkah visualisasi berikutnya."""
        if not self.steps:
            return

        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            self.draw_graph_step()
            self.update_step_info(self.steps[self.current_step_index])
            
            # Cek apakah langkah ini adalah langkah terakhir
            if self.current_step_index == len(self.steps) - 1:
                self.step_button.config(state=tk.DISABLED)
                messagebox.showinfo("Selesai", "Visualisasi selesai. MST telah ditemukan. Graf akhir ditampilkan di jendela baru.")
                self.show_final_button.config(state=tk.NORMAL)
                self.show_final_graph_popup()
                if self.is_animating:
                    self.toggle_auto_step()
                
        else:
            # Ini tidak akan tercapai jika cek di atas sudah benar
            pass 

    def toggle_auto_step(self):
        """Mengaktifkan/menonaktifkan mode otomatis."""
        if not self.steps:
            messagebox.showwarning("Peringatan", "Jalankan Kruskal terlebih dahulu.")
            return

        self.is_animating = not self.is_animating
        if self.is_animating:
            self.auto_step_button.config(text="⏸️ JEDA Otomatis", bg="#e67e22")
            self.step_button.config(state=tk.DISABLED)
            self.auto_step()
        else:
            self.auto_step_button.config(text="⚡ Mode Otomatis", bg="#1abc9c")
            if self.current_step_index < len(self.steps) - 1:
                self.step_button.config(state=tk.NORMAL)
            if self.animation_job:
                self.after_cancel(self.animation_job)
                self.animation_job = None

    def auto_step(self):
        """Logika untuk langkah otomatis."""
        if self.is_animating and self.current_step_index < len(self.steps) - 1:
            self.next_step()
            self.animation_job = self.after(self.animation_delay_ms, self.auto_step)
        else:
            if self.is_animating:
                 # Jika selesai saat mode otomatis aktif
                 self.toggle_auto_step()

    # --- FUNGSI POPUP DAN GAMBAR STATIS ---
    def show_final_graph_popup(self):
        """Membuka jendela popup untuk menampilkan Graf Asli dan MST Akhir."""
        if not self.solver.edges:
            messagebox.showwarning("Peringatan", "Tidak ada data graf untuk ditampilkan.")
            return

        popup = tk.Toplevel(self)
        popup.title(f"Hasil Akhir Kruskal | Biaya Total: {self.final_cost}")
        
        # Hitung ukuran popup (2 * CANVAS_WIDTH + padding) Disesuaikan
        popup_width = CANVAS_WIDTH * 2 + 80
        popup_height = CANVAS_HEIGHT + 100
        popup.geometry(f"{popup_width}x{popup_height}")
        popup.configure(bg="#34495e")
        
        # Container frame inside popup
        container = tk.Frame(popup, bg="#34495e", padx=10, pady=10)
        container.pack(fill=tk.BOTH, expand=True)

        # --- Graf Asli Frame ---
        original_frame = tk.Frame(container, bg="#34495e")
        original_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        tk.Label(original_frame, text="GRAF ASLI (Semua Rute)", font=("Helvetica", 12, "bold"), fg="#f39c12", bg="#34495e").pack(pady=(0, 5))
        original_canvas = tk.Canvas(original_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#ecf0f1", bd=0, highlightthickness=0)
        original_canvas.pack()

        # --- MST Akhir Frame ---
        mst_frame = tk.Frame(container, bg="#34495e")
        mst_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)
        tk.Label(mst_frame, text="MST AKHIR (Rute Pilihan)", font=("Helvetica", 12, "bold"), fg="#2ecc71", bg="#34495e").pack(pady=(0, 5))
        mst_canvas = tk.Canvas(mst_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#ecf0f1", bd=0, highlightthickness=0)
        mst_canvas.pack()

        # Draw the graphs using the helper function
        self._draw_static_graph(original_canvas, self.solver.edges, is_mst=False)
        self._draw_static_graph(mst_canvas, self.final_mst, is_mst=True)

    def _draw_static_graph(self, canvas, edges_to_draw, is_mst):
        """Fungsi helper untuk menggambar graf statis. edges_to_draw HARUS (w, u, v)"""
        
        # 1. Gambar Sisi
        for w, u, v in edges_to_draw: # Format: (w, u, v)
            if u not in self.solver.node_positions or v not in self.solver.node_positions:
                continue
                
            x1, y1 = self.solver.node_positions[u]
            x2, y2 = self.solver.node_positions[v]
            
            line_color = "#34495e" if not is_mst else "#2ecc71"
            line_width = 1 if not is_mst else 3
            text_color = "#34495e"
            
            canvas.create_line(x1, y1, x2, y2, fill=line_color, width=line_width)
            
            # Gambar Bobot
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            canvas.create_rectangle(mid_x-10, mid_y-8, mid_x+10, mid_y+8, 
                                         fill="#ecf0f1", outline=line_color)
                                         
            canvas.create_text(mid_x, mid_y, text=str(w), 
                                    fill=text_color, 
                                    font=("Helvetica", 9, "bold"))
                                    
        # 2. Gambar Simpul (Nodes)
        for index, name in enumerate(self.solver.city_names):
            if index not in self.solver.node_positions:
                continue

            x, y = self.solver.node_positions[index]
            
            # Lingkaran Node
            canvas.create_oval(x - NODE_RADIUS, y - NODE_RADIUS, 
                                    x + NODE_RADIUS, y + NODE_RADIUS, 
                                    fill="#3498db", outline="#2980b9", width=2)
            
            # Teks ID Kota
            canvas.create_text(x, y, text=str(index), fill="white", 
                                    font=("Helvetica", 9, "bold"))
            
            # Teks Nama Kota di bawah
            canvas.create_text(x, y + NODE_RADIUS + 8, text=name, fill="#2c3e50", 
                                    font=("Helvetica", 8))
        
    # --- FUNGSI GAMBAR VISUALISASI UTAMA (PROSES) ---
    def draw_graph_step(self):
        """Menggambar graf di kanvas proses berdasarkan langkah visualisasi saat ini."""
        self.process_canvas.delete("all")
        
        current_mst_edges_wuv = []
        current_edge_uvw = None
        current_action = None

        if self.current_step_index >= 0:
            current_step = self.steps[self.current_step_index]
            current_mst_edges_wuv = current_step['mst_so_far']
            current_edge_uvw = current_step['edge']
            current_action = current_step['action']

        # 1. Gambar Sisi
        for w_all, u_all, v_all in self.solver.edges:
            x1, y1 = self.solver.node_positions[u_all]
            x2, y2 = self.solver.node_positions[v_all]
            
            edge_tuple = tuple(sorted((u_all, v_all)))

            is_in_current_mst = any(tuple(sorted((u1, v1))) == edge_tuple for w1, u1, v1 in current_mst_edges_wuv)
            
            is_current_edge = (self.current_step_index >= 0 and 
                               tuple(sorted((current_edge_uvw[0], current_edge_uvw[1]))) == edge_tuple)

            # --- Tentukan Gaya Garis ---
            line_color = "#95a5a6" 
            line_width = 1
            bg_color = "#ecf0f1"
            text_color = "#34495e"
            
            if is_in_current_mst:
                line_color = "#2ecc71"
                line_width = 3
                bg_color = "#ecf0f1" # Warna latar belakang agar label jelas
                text_color = "#2ecc71"
            elif is_current_edge:
                if current_action == 'ADD':
                    line_color = "#3498db" 
                    line_width = 4
                    bg_color = "#ecf0f1"
                    text_color = "#3498db"
                else: # REJECT
                    line_color = "#e74c3c" 
                    line_width = 4
                    bg_color = "#ecf0f1"
                    text_color = "#e74c3c"
            
            # Gambar Garis
            self.process_canvas.create_line(x1, y1, x2, y2, fill=line_color, width=line_width, tags="edge_line")
            
            # Gambar Bobot
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            self.process_canvas.create_rectangle(mid_x-10, mid_y-8, mid_x+10, mid_y+8, 
                                         fill=bg_color, outline=line_color, tags="edge_weight_bg")
                                         
            self.process_canvas.create_text(mid_x, mid_y, text=str(w_all), 
                                    fill=text_color, 
                                    font=("Helvetica", 9, "bold"), tags="edge_weight")

        # 2. Gambar Simpul (Nodes)
        for index, name in enumerate(self.solver.city_names):
            x, y = self.solver.node_positions[index]
            
            # Lingkaran Node
            self.process_canvas.create_oval(x - NODE_RADIUS, y - NODE_RADIUS, 
                                    x + NODE_RADIUS, y + NODE_RADIUS, 
                                    fill="#3498db", outline="#2980b9", width=2, 
                                    tags=(f"node_{index}", "node"))
            
            # Teks ID Kota
            self.process_canvas.create_text(x, y, text=str(index), fill="white", 
                                    font=("Helvetica", 9, "bold"), tags=(f"node_{index}"))
            
            # Teks Nama Kota di bawah
            self.process_canvas.create_text(x, y + NODE_RADIUS + 8, text=name, fill="#2c3e50", 
                                    font=("Helvetica", 8), tags=(f"node_{index}"))

        self.process_canvas.tag_raise("edge_weight_bg")
        self.process_canvas.tag_raise("edge_weight")
        self.process_canvas.tag_raise("node")
        
    # --- FUNGSI DRAG NODE ---
    def start_drag(self, event):
        item = self.process_canvas.find_closest(event.x, event.y)
        tags = self.process_canvas.gettags(item)
        
        if "node" in tags:
            self.dragged_node_id = None
            for tag in tags:
                if tag.startswith("node_"):
                    self.dragged_node_id = int(tag.split("_")[1])
                    break
            
            if self.dragged_node_id is not None:
                self.start_x = event.x
                self.start_y = event.y
                self.process_canvas.config(cursor="hand2")
        else:
            self.dragged_node_id = None
            
    def drag(self, event):
        if self.dragged_node_id is not None:
            dx = event.x - self.start_x
            dy = event.y - self.start_y
            
            old_x, old_y = self.solver.node_positions[self.dragged_node_id]
            
            # Batasi pergerakan agar tidak keluar dari Canvas
            new_x = max(NODE_RADIUS, min(old_x + dx, CANVAS_WIDTH - NODE_RADIUS))
            new_y = max(NODE_RADIUS, min(old_y + dy, CANVAS_HEIGHT - NODE_RADIUS))
            
            # Perbarui posisi di solver
            self.solver.node_positions[self.dragged_node_id] = (new_x, new_y)
            
            self.start_x = event.x
            self.start_y = event.y
            
            # Perbarui tampilan proses saat drag
            self.draw_graph_step()

    def stop_drag(self, event):
        self.dragged_node_id = None
        self.process_canvas.config(cursor="")


if __name__ == "__main__":
    app = KruskalApp()
    app.mainloop()