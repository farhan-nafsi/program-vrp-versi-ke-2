import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Data pelanggan
customers = {
    'A': {'demand': 50, 'location': (2, 3), 'time_window': (7, 10)},
    'B': {'demand': 30, 'location': (5, 7), 'time_window': (8, 10)},
    'C': {'demand': 70, 'location': (3, 8), 'time_window': (9, 15)},
    'D': {'demand': 60, 'location': (6, 4), 'time_window': (13, 15)},
    'E': {'demand': 40, 'location': (1, 6), 'time_window': (10, 13)},
    'F': {'demand': 20, 'location': (7, 9), 'time_window': (14, 17)},
    'G': {'demand': 30, 'location': (4, 5), 'time_window': (7, 13)},
    'H': {'demand': 90, 'location': (8, 2), 'time_window': (7, 15)}
}

# Kapasitas truk
truck_capacity = 100
num_trucks = 3

# Depot
depot = (0, 0)

# Fungsi menghitung jarak antara dua titik
def distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

# Fungsi membuat rute acak
def create_random_route():
    route = list(customers.keys())
    random.shuffle(route)
    return route

# Fungsi membuat populasi awal
def create_initial_population(pop_size):
    return [create_random_route() for _ in range(pop_size)]

# Fungsi untuk memecah rute menjadi beberapa truk
def split_route_to_trucks(route):
    trucks = []
    truck_load = 0
    current_truck = [depot]  # Mulai dari depot

    for customer in route:
        demand = customers[customer]['demand']
        
        if truck_load + demand > truck_capacity:
            # Jika kapasitas truk sudah penuh, simpan rute saat ini dan mulai rute baru
            current_truck.append(depot)  # Kembali ke depot
            trucks.append(current_truck)  # Simpan rute truk ini
            current_truck = [depot]  # Mulai rute baru dari depot
            truck_load = 0  # Reset muatan truk

        # Tambahkan pelanggan ke truk saat ini
        current_truck.append(customer)
        truck_load += demand
    
    # Tambahkan kembali ke depot untuk truk terakhir
    current_truck.append(depot)
    trucks.append(current_truck)
    
    return trucks

# Fungsi menghitung fitness untuk setiap rute
def fitness(route):
    trucks = split_route_to_trucks(route)
    total_distance = 0

    for truck_route in trucks:
        for i in range(len(truck_route) - 1):
            loc1 = depot if truck_route[i] == depot else customers[truck_route[i]]['location']
            loc2 = depot if truck_route[i + 1] == depot else customers[truck_route[i + 1]]['location']
            total_distance += distance(loc1, loc2)
    
    return total_distance

# Fungsi seleksi untuk memilih parent
def selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected = sorted(selected, key=lambda x: x[1])
    return selected[0][0]

# Fungsi crossover (pertukaran genetik antar parent)
def crossover(parent1, parent2):
    child = []
    child_p1 = []
    child_p2 = []

    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for i in range(start_gene, end_gene):
        child_p1.append(parent1[i])

    child_p2 = [item for item in parent2 if item not in child_p1]

    child = child_p1 + child_p2
    return child

# Fungsi mutasi untuk variasi dalam populasi
def mutate(route, mutation_rate=0.01):
    for swapped in range(len(route)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(route))

            route[swapped], route[swap_with] = route[swap_with], route[swapped]
    return route

# Fungsi utama Genetic Algorithm dengan visualisasi
def genetic_algorithm(pop_size, generations):
    population = create_initial_population(pop_size)
    
    for generation in range(generations):
        fitnesses = [fitness(route) for route in population]
        new_population = []
        
        for _ in range(pop_size):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    # Ambil solusi terbaik
    fitnesses = [fitness(route) for route in population]
    best_route_index = fitnesses.index(min(fitnesses))
    best_route = population[best_route_index]
    best_truck_routes = split_route_to_trucks(best_route)
    
    # Plot rute terbaik untuk setiap truk
    plt.figure(figsize=(10, 5))
    colors = ['b', 'g', 'orange']
    
    for i, truck_route in enumerate(best_truck_routes):
        x = [depot[0]] + [customers[customer]['location'][0] for customer in truck_route if customer != depot] + [depot[0]]
        y = [depot[1]] + [customers[customer]['location'][1] for customer in truck_route if customer != depot] + [depot[1]]
        plt.plot(x, y, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Truck {i + 1}')
    
    plt.scatter(depot[0], depot[1], color='red', label="Depot")
    plt.title('Rute Terbaik Dengan Genetic Algorithm ')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_route, min(fitnesses)

# Jalankan algoritma
best_route, best_distance = genetic_algorithm(pop_size=50, generations=100)
print("Best Route:", best_route)
print("Best Distance:", best_distance)
