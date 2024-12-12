import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Data for cities
data = {
    'City': ['Tehran', 'Isfahan', 'Tabriz', 'Shiraz', 'Mashhad',
             'Kermanshah', 'Yazd', 'Karaj', 'Ahvaz', 'Qom',
             'Urmia', 'Arak', 'Kerman', 'Zanjan', 'Sari',
             'Gorgan', 'Bandar Abbas', 'Birjand', 'Sabzevar', 'Bojnurd'],
    'Latitude': [35.6892, 32.6546, 38.0962, 29.5918, 36.2605,
                 34.3293, 31.8974, 35.8325, 31.3193, 34.639,
                 37.5505, 34.0944, 30.2835, 36.6732, 36.6751,
                 36.6975, 27.1884, 32.8716, 33.0565, 37.4875],
    'Longitude': [51.3890, 51.6570, 46.2913, 52.5836, 59.5443,
                  47.1167, 54.3660, 51.9792, 48.6692, 50.8764,
                  45.9773, 49.6957, 57.0786, 48.5014, 53.0204,
                  54.0172, 56.2167, 59.2253, 57.6775, 57.1847]
}

iran_df = pd.DataFrame(data)
points = list(zip(iran_df['Longitude'], iran_df['Latitude']))

# Genetic Algorithm Class
class GeneticAlgorithm:
    def __init__(self, points, population_size=100, generations=500, mutation_rate=0.01,
                 init_method='heuristic', select_method='random', fitness_method='normalized'):
        self.points = points
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Initialize population based on chosen method
        if init_method == 'heuristic':
            self.population = self.initialize_population_heuristic()
        elif init_method == 'cluster':
            self.population = self.initialize_population_cluster()
        
        # Set selection and fitness methods
        self.select_parents = self.select_parents_random if select_method == 'random' else self.select_parents_tournament
        self.fitness_method = self.normalized_fitness if fitness_method == 'normalized' else self.rank_based_fitness

    def initialize_population_heuristic(self):
        return [np.random.permutation(len(self.points)) for _ in range(self.population_size)]

    def initialize_population_cluster(self):
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(self.points)
        cluster_centers = kmeans.cluster_centers_
        
        population = []
        for _ in range(self.population_size):
            order = np.random.permutation(len(cluster_centers))
            individual = [np.random.choice(np.where(kmeans.labels_ == i)[0]) for i in order]
            population.append(individual)
        
        return population

    def calculate_distance(self, tour):
        distance = sum(np.linalg.norm(np.array(self.points[tour[i]]) - np.array(self.points[tour[(i + 1) % len(tour)]])) for i in range(len(tour)))
        return distance

    def select_parents_random(self):
        fitness = [1 / self.calculate_distance(tour) for tour in self.population]
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]
        return np.random.choice(len(self.population), size=2, p=probabilities)

    def select_parents_tournament(self):
        tournament_size = 5
        tournament_indices = np.random.choice(len(self.population), size=tournament_size)
        tournament_fitness = [1 / self.calculate_distance(self.population[i]) for i in tournament_indices]
        winner_indices = tournament_indices[np.argsort(tournament_fitness)[-2:]]
        return winner_indices

    def normalized_fitness(self):
        fitness = [1 / self.calculate_distance(tour) for tour in self.population]
        total_fitness = sum(fitness)
        return [f / total_fitness for f in fitness]

    def rank_based_fitness(self):
        distances = [self.calculate_distance(tour) for tour in self.population]
        sorted_indices = np.argsort(distances)
        ranks = np.arange(len(self.population), dtype=float) + 1
        rank_fitness = ranks[sorted_indices]
        return rank_fitness / rank_fitness.sum()

    def crossover(self, parent1, parent2):
        start, end = sorted(np.random.choice(len(parent1), 2, replace=False))
        child = [None] * len(parent1)
        child[start:end + 1] = parent1[start:end + 1]
        
        current_position = (end + 1) % len(parent1)
        for gene in parent2:
            if gene not in child:
                child[current_position] = gene
                current_position = (current_position + 1) % len(parent1)
        
        return np.array(child)

    def mutate(self, tour):
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(len(tour), 2, replace=False)
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]

    def run(self):
        for generation in range(self.generations):
            new_population = []
            for _ in range(self.population_size):
                parent_indices = self.select_parents()
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            self.population = new_population
        
        best_tour = min(self.population, key=self.calculate_distance)
        best_distance = self.calculate_distance(best_tour)
        
        return best_tour, best_distance

# Running the genetic algorithm with different configurations
results = {}

for init_method in ['heuristic', 'cluster']:
    for select_method in ['random', 'tournament']:
        for fitness_method in ['normalized', 'rank']:
            ga = GeneticAlgorithm(points,
                                  init_method=init_method,
                                  select_method=select_method,
                                  fitness_method=fitness_method)
            best_tour, best_distance = ga.run()
            results[(init_method, select_method, fitness_method)] = (best_tour, best_distance)

# Analyzing and printing results
for key in results:
    init_method, select_method, fitness_method = key
    best_tour, best_distance = results[key]
    print(f"Initialization: {init_method}, Selection: {select_method}, Fitness: {fitness_method} -> Best Distance: {best_distance}")