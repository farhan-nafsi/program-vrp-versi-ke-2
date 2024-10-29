import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Data pelanggan
customers = {
    'Depot': {'demand': 0, 'location': (0, 0), 'time_window': (0, 24)},
    'A': {'demand': 50, 'location': (2, 3), 'time_window': (7, 10)},
    'B': {'demand': 30, 'location': (5, 7), 'time_window': (8, 10)},
    'C': {'demand': 70, 'location': (3, 8), 'time_window': (9, 15)},
    'D': {'demand': 60, 'location': (6, 4), 'time_window': (13, 15)},
    'E': {'demand': 40, 'location': (1, 6), 'time_window': (10, 13)},
    'F': {'demand': 20, 'location': (7, 9), 'time_window': (14, 17)},
    'G': {'demand': 30, 'location': (4, 5), 'time_window': (7, 13)},
    'H': {'demand': 90, 'location': (8, 2), 'time_window': (7, 15)}
}

vehicle_count = 3
vehicle_capacity = 100

# Parameter ACO
alpha = 1.0  # importance of pheromone
beta = 2.0   # importance of distance
evaporation_rate = 0.5
num_ants = 10
iterations = 100

# Fungsi jarak Euclidean
def euclidean_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

# Membuat matriks jarak antar pelanggan
distances = {}
for customer1 in customers:
    distances[customer1] = {}
    for customer2 in customers:
        distances[customer1][customer2] = euclidean_distance(customers[customer1]['location'], customers[customer2]['location'])

# Inisialisasi pheromone
pheromones = {customer: {other: 1 for other in customers} for customer in customers}

# Fungsi ACO untuk memilih pelanggan berikutnya berdasarkan probabilitas
def select_next_customer(current_customer, visited, pheromones, distances):
    probabilities = {}
    for customer in customers:
        if customer not in visited and customer != 'Depot':
            probabilities[customer] = (pheromones[current_customer][customer] ** alpha) * ((1.0 / distances[current_customer][customer]) ** beta)
    
    total = sum(probabilities.values())
    if total == 0:
        return 'Depot'  # Balik ke Depot jika tidak ada pilihan lain
    
    probabilities = {k: v / total for k, v in probabilities.items()}
    rand = random.uniform(0, 1)
    cumulative = 0
    for customer, probability in probabilities.items():
        cumulative += probability
        if rand <= cumulative:
            return customer
    return 'Depot'

# Fungsi untuk menghasilkan rute dengan ACO
def aco_vrp():
    best_route = None
    best_distance = float('inf')
    
    for iteration in range(iterations):
        routes = []
        for ant in range(num_ants):
            route = []
            visited = set()
            current_vehicle_load = 0
            current_customer = 'Depot'
            sub_route = [current_customer]
            
            while len(visited) < len(customers) - 1:
                next_customer = select_next_customer(current_customer, visited, pheromones, distances)
                
                if next_customer == 'Depot' or current_vehicle_load + customers[next_customer]['demand'] > vehicle_capacity:
                    # End sub-route and start a new one
                    sub_route.append('Depot')
                    route.append(sub_route)
                    sub_route = ['Depot']
                    current_vehicle_load = 0
                    current_customer = 'Depot'
                else:
                    # Add customer to sub-route
                    sub_route.append(next_customer)
                    visited.add(next_customer)
                    current_vehicle_load += customers[next_customer]['demand']
                    current_customer = next_customer
            
            # Menutup rute terakhir
            if sub_route[-1] != 'Depot':
                sub_route.append('Depot')
            route.append(sub_route)
            routes.append(route)
        
        # Evaluasi rute dan update pheromone
        for route in routes:
            route_distance = 0
            for sub_route in route:
                for i in range(len(sub_route) - 1):
                    route_distance += distances[sub_route[i]][sub_route[i+1]]
            
            if route_distance < best_distance:
                best_distance = route_distance
                best_route = route
            
            # Update pheromone
            for sub_route in route:
                for i in range(len(sub_route) - 1):
                    pheromones[sub_route[i]][sub_route[i+1]] *= (1 - evaporation_rate)
                    pheromones[sub_route[i]][sub_route[i+1]] += (1 / route_distance)
    
    return best_route, best_distance

# Menjalankan ACO VRP dan menampilkan hasil
best_route, best_distance = aco_vrp()
print("Rute Terbaik:", best_route)
print("Jarak Terpendek:", best_distance)

# Visualisasi rute terbaik
def plot_route(route):
    colors = ['r', 'g', 'b', 'c', 'm']  # Warna untuk tiap truk
    plt.figure(figsize=(10, 8))
    
    # Plot depot dan pelanggan
    for customer, data in customers.items():
        x, y = data['location']
        plt.scatter(x, y, c='black' if customer == 'Depot' else 'blue', s=100)
        plt.text(x, y, f'{customer} ({data["demand"]})', fontsize=12, ha='right')
    
    # Plot rute
    for i, sub_route in enumerate(route):
        color = colors[i % len(colors)]
        for j in range(len(sub_route) - 1):
            start, end = sub_route[j], sub_route[j + 1]
            start_loc, end_loc = customers[start]['location'], customers[end]['location']
            plt.plot([start_loc[0], end_loc[0]], [start_loc[1], end_loc[1]], color=color, linestyle='-', linewidth=2, label=f'Truk {i+1}' if j == 0 else "")
    
    plt.xlabel("X Location")
    plt.ylabel("Y Location")
    plt.title("Rute Terbaik dengan Ant Colony Optimization")
    plt.legend()
    plt.grid()
    plt.show()

# Memvisualisasikan rute terbaik
plot_route(best_route)
