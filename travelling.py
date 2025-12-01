import math
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class TSPAnnealing:
    def __init__(self, num_cities=6, temp=1000, cooling_rate=0.99):
        self.num_cities = num_cities
        self.temp = temp
        self.initial_temp = temp
        self.cooling_rate = cooling_rate
        
        # Generate random cities with minimum distance constraint
        self.cities = []
        attempts = 0
        while len(self.cities) < num_cities and attempts < 1000:
            candidate = (random.randint(10, 90), random.randint(10, 90))
            if all(math.hypot(candidate[0] - c[0], candidate[1] - c[1]) > 15 for c in self.cities):
                self.cities.append(candidate)
            attempts += 1
            
        while len(self.cities) < num_cities:
             self.cities.append((random.randint(5, 95), random.randint(5, 95)))

        self.labels = [chr(65 + i) for i in range(num_cities)]
        
        # Initial random tour
        self.current_tour = list(range(num_cities))
        random.shuffle(self.current_tour)
        
        # Store Initial State for Comparison
        self.initial_tour = list(self.current_tour)
        self.initial_dist = self.calculate_total_distance(self.initial_tour)
        
        self.best_tour = list(self.current_tour)
        self.current_dist = self.calculate_total_distance(self.current_tour)
        self.best_dist = self.current_dist
        
        self.iteration = 0

    def distance(self, city1, city2):
        return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

    def calculate_total_distance(self, tour):
        dist = 0
        for i in range(len(tour)):
            from_city = self.cities[tour[i]]
            to_city = self.cities[tour[(i + 1) % len(tour)]]
            dist += self.distance(from_city, to_city)
        return dist

    def step(self):
        """Perform one step. Returns False if process is finished (temp too low)."""
        if self.temp <= 0.5:
            return False

        self.temp *= self.cooling_rate
        self.iteration += 1

        new_tour = list(self.current_tour)
        i, j = random.sample(range(self.num_cities), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        
        new_dist = self.calculate_total_distance(new_tour)
        
        accept = False
        if new_dist < self.current_dist:
            accept = True
        else:
            if self.temp > 0.001:
                prob = math.exp((self.current_dist - new_dist) / self.temp)
                accept = random.random() < prob

        if accept:
            self.current_tour = new_tour
            self.current_dist = new_dist
            if self.current_dist < self.best_dist:
                self.best_dist = self.current_dist
                self.best_tour = list(self.current_tour)
        
        return True

def interactive_tsp():
    NUM_CITIES = 6
    
    state = {
        'solver': TSPAnnealing(num_cities=NUM_CITIES),
        'running': False
    }
    
    # Use GridSpec for a nicer layout: 2 columns for maps, 1 row at bottom for stats
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    
    ax_initial = fig.add_subplot(gs[0, 0])
    ax_dynamic = fig.add_subplot(gs[0, 1])
    ax_stats = fig.add_subplot(gs[1, :])
    
    plt.subplots_adjust(bottom=0.15)
    
    # Track both iterations and distances for accurate X-axis plotting
    history = {'iters': [], 'dists': []}

    def draw_graph_background(ax, solver):
        """Helper to draw faint all-to-all connections and labels."""
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.axis('off')
        
        # Draw all possible roads (faint)
        for i in range(solver.num_cities):
            for j in range(i + 1, solver.num_cities):
                p1 = solver.cities[i]
                p2 = solver.cities[j]
                dist = solver.distance(p1, p2)
                # Increased alpha (opacity) to 0.3 and linewidth to 1.0
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.3, linewidth=1.0, zorder=1)
                
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                ax.text(mx, my, f'{dist:.0f}', fontsize=6, color='gray', ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0), zorder=1)
        
        # Draw Cities and Labels
        city_x, city_y = zip(*solver.cities)
        ax.plot(city_x, city_y, 'ro', markersize=8, zorder=5)
        for i, (cx, cy) in enumerate(solver.cities):
            ax.text(cx + 2, cy + 2, solver.labels[i], fontsize=11, fontweight='bold', zorder=10)

    def draw_path(ax, solver, tour, color, style='-', show_costs=False):
        """Helper to draw a specific tour."""
        tour_indices = tour + [tour[0]]
        tour_coords = [solver.cities[i] for i in tour_indices]
        xs, ys = zip(*tour_coords)
        ax.plot(xs, ys, color=color, linestyle=style, alpha=0.8, linewidth=2.0, zorder=2)
        
        if show_costs:
            for k in range(len(tour_indices) - 1):
                p1 = solver.cities[tour_indices[k]]
                p2 = solver.cities[tour_indices[k+1]]
                seg_dist = solver.distance(p1, p2)
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                ax.text(mx, my, f'{seg_dist:.1f}', fontsize=8, color=color, fontweight='bold', ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.2'), zorder=3)
        
        # Highlight Start
        ax.plot(xs[0], ys[0], 'go', markersize=10, zorder=6)

    def draw():
        solver = state['solver']
        
        # --- Left Plot: Graph Topology (Static) ---
        ax_initial.clear()
        ax_initial.set_title(f"Graph Topology\n(All Possible Roads)")
        draw_graph_background(ax_initial, solver)
        # Note: No path drawn here, just the structure
        
        # --- Right Plot: Dynamic Annealing ---
        ax_dynamic.clear()
        status = "FINISHED" if solver.temp <= 0.5 else "RUNNING"
        
        draw_graph_background(ax_dynamic, solver)
        
        if status == "FINISHED":
            # Show ONLY the final best path
            ax_dynamic.set_title(f"Optimization Complete (Total Iterations: {solver.iteration})\nFinal Best Distance: {solver.best_dist:.1f}", color='green')
            draw_path(ax_dynamic, solver, solver.best_tour, 'green', style='-', show_costs=True)
        else:
            # Show the running process
            ax_dynamic.set_title(f"Optimizing... Iteration: {solver.iteration} | Temp: {solver.temp:.1f}", color='black')
            
            # Draw Best So Far (Green Dashed)
            draw_path(ax_dynamic, solver, solver.best_tour, 'green', style='--', show_costs=False)
            
            # Draw Current Candidate (Blue Solid)
            draw_path(ax_dynamic, solver, solver.current_tour, 'blue', show_costs=True)

        # --- Bottom Plot: Stats ---
        ax_stats.clear()
        
        # Record history with explicit iteration number
        history['iters'].append(solver.iteration)
        history['dists'].append(solver.current_dist)
        
        # Plot Iteration vs Distance
        ax_stats.plot(history['iters'], history['dists'], color='purple', label='Current')
        ax_stats.axhline(y=solver.best_dist, color='green', linestyle='--', alpha=0.5, label='Best')
        ax_stats.set_title(f"Distance History (Best: {solver.best_dist:.1f} | Iterations: {solver.iteration})")
        ax_stats.set_ylabel("Distance")
        ax_stats.set_xlabel("Iterations")
        ax_stats.legend(loc='upper right')
        
        plt.draw()

    # --- Button Callbacks ---
    def on_step(event):
        solver = state['solver']
        state['running'] = False
        solver.step()
        draw()

    def on_auto(event):
        if state['running']: return
        
        solver = state['solver']
        state['running'] = True
        
        # Run loop
        while state['running']:
            has_next = solver.step()
            
            # Stop if solver finished
            if not has_next:
                state['running'] = False
                draw() # Final draw to show 'Finished' state
                break
                
            # Update visualization periodically
            if solver.iteration % 3 == 0:
                draw()
                plt.pause(0.001) 
        
        draw()

    def on_stop(event):
        state['running'] = False

    def on_reset(event):
        state['running'] = False
        state['solver'] = TSPAnnealing(num_cities=NUM_CITIES)
        
        # Reset history
        history['iters'] = [0]
        history['dists'] = [state['solver'].current_dist]
        draw()

    # --- Create Buttons ---
    ax_reset_btn = plt.axes([0.05, 0.02, 0.15, 0.06])
    ax_step_btn = plt.axes([0.25, 0.02, 0.15, 0.06])
    ax_auto_btn = plt.axes([0.45, 0.02, 0.15, 0.06])
    ax_stop_btn = plt.axes([0.65, 0.02, 0.15, 0.06])
    
    btn_reset = Button(ax_reset_btn, 'New Graph')
    btn_step = Button(ax_step_btn, 'Next Step')
    btn_auto = Button(ax_auto_btn, 'Auto Run')
    btn_stop = Button(ax_stop_btn, 'Stop')
    
    btn_reset.on_clicked(on_reset)
    btn_step.on_clicked(on_step)
    btn_auto.on_clicked(on_auto)
    btn_stop.on_clicked(on_stop)
    
    # Initialize first draw manually to populate initial history
    history['iters'].append(0)
    history['dists'].append(state['solver'].current_dist)
    draw()
    plt.show()

if __name__ == "__main__":
    print("Starting Interactive TSP...")
    interactive_tsp()