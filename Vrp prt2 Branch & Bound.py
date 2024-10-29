import itertools
from queue import PriorityQueue
import math
import matplotlib.pyplot as plt
import numpy as np

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

# Kondisi kendaraan
num_trucks = 3
truck_capacity = 100
depot_location = (0, 0)

# Fungsi untuk menghitung jarak Euclidean antara dua titik
def distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

# Fungsi untuk menghitung rute terbaik untuk setiap truk dengan membatasi kapasitas
def calculate_routes(customers, num_trucks, truck_capacity):
    routes = []
    unvisited_customers = set(customers.keys())
    
    for _ in range(num_trucks):
        route = [depot_location]
        load = 0
        while unvisited_customers:
            next_customer = None
            min_distance = float('inf')
            for customer in unvisited_customers:
                demand = customers[customer]['demand']
                loc = customers[customer]['location']
                if load + demand <= truck_capacity:
                    dist = distance(route[-1], loc)
                    if dist < min_distance:
                        min_distance = dist
                        next_customer = customer
            
            if next_customer is None:
                break

            route.append(customers[next_customer]['location'])
            load += customers[next_customer]['demand']
            unvisited_customers.remove(next_customer)

        route.append(depot_location)
        routes.append(route)
    
    return routes

# Menghitung rute untuk setiap truk
truck_routes = calculate_routes(customers, num_trucks, truck_capacity)

# Visualisasi rute terbaik
def plot_routes(routes, customers, depot_location):
    plt.figure(figsize=(10, 6))
    
    # Plot depot
    plt.plot(depot_location[0], depot_location[1], 'go', markersize=10, label="Depot")
    
    # Plot pelanggan
    for customer, data in customers.items():
        loc = data['location']
        plt.plot(loc[0], loc[1], 'bo', markersize=8)
        plt.text(loc[0] + 0.1, loc[1] + 0.1, f"{customer} ({data['demand']})", fontsize=9)

    # Colors untuk setiap truk
    colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))

    # Plot setiap rute truk dengan warna berbeda
    for i, route in enumerate(routes):
        color = colors[i % len(routes)]
        for j in range(len(route) - 1):
            start = route[j]
            end = route[j + 1]
            plt.plot([start[0], end[0]], [start[1], end[1]], '--', color=color, label=f"Truk {i+1}" if j == 0 else "")

    # Labels dan display
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Rute Terbaik Truk Dengan Branch & Bound")
    plt.legend()
    plt.grid()
    plt.show()

# Plot hasil rute setiap truk
plot_routes(truck_routes, customers, depot_location)
