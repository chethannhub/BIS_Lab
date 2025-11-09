import random
import math

class AntColonyOptimization:
    """
    A class to implement the Ant Colony Optimization algorithm for the Traveling Salesman Problem.
    """
    def __init__(self, distances, n_ants, n_iterations, alpha, beta, evaporation_rate, q=1.0):
        # --- Store Parameters ---
        self.distances = distances  # 2D matrix of distances between cities
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Importance of pheromone
        self.beta = beta    # Importance of heuristic (distance)
        self.evaporation_rate = evaporation_rate
        self.q = q  # Pheromone deposit factor
        
        self.n_cities = len(distances)
        
        # --- Initialization ---
        # As we discussed, we initialize pheromones to a small, uniform value.
        # A 2D matrix is perfect for this, storing pheromone level for path (i, j).
        self.pheromone = [[1 / (self.n_cities * self.n_cities) for _ in range(self.n_cities)] for _ in range(self.n_cities)]
        
        self.best_tour = None
        self.best_tour_length = float('inf')

    def run(self):
        """
        The main loop of the ACO algorithm.
        """
        for iteration in range(self.n_iterations):
            all_tours = []
            for ant_k in range(self.n_ants):
                tour = self._construct_solution(ant_k)
                all_tours.append(tour)
                
                # Check if this ant found a new best tour
                tour_length = self._calculate_tour_length(tour)
                if tour_length < self.best_tour_length:
                    self.best_tour = tour
                    self.best_tour_length = tour_length
            
            self._update_pheromones(all_tours)
            print(f"Iteration {iteration+1}: Best Tour Length = {self.best_tour_length:.2f}")

        return self.best_tour, self.best_tour_length

    def _update_pheromones(self, all_tours):
        """
        Updates the pheromone matrix based on the tours constructed by the ants.
        """
        # Part A: Evaporation - all trails fade slightly
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                self.pheromone[i][j] *= (1 - self.evaporation_rate)

        # Part B: Reinforcement - deposit new pheromone
        for tour in all_tours:
            tour_length = self._calculate_tour_length(tour)
            pheromone_to_add = self.q / tour_length
            for i in range(self.n_cities - 1):
                city_from = tour[i]
                city_to = tour[i + 1]
                self.pheromone[city_from][city_to] += pheromone_to_add
                self.pheromone[city_to][city_from] += pheromone_to_add # For symmetric TSP
            # Add pheromone for the last leg of the tour
            last_leg_from = tour[-1]
            last_leg_to = tour[0]
            self.pheromone[last_leg_from][last_leg_to] += pheromone_to_add
            self.pheromone[last_leg_to][last_leg_from] += pheromone_to_add

    def _construct_solution(self, ant_k):
        """
        Builds a tour for a single ant.
        """
        start_city = random.randint(0, self.n_cities - 1)
        tour = [start_city]
        visited = {start_city}
        
        current_city = start_city
        while len(tour) < self.n_cities:
            next_city = self._select_next_city(current_city, visited)
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
            
        return tour

    def _select_next_city(self, current_city, visited):
        """
        Selects the next city for an ant based on the probabilistic formula.
        """
        probabilities = []
        total_prob = 0.0
        
        # Calculate the denominator of our probability formula
        for city in range(self.n_cities):
            if city not in visited:
                # This is the core formula combining pheromone (tau) and heuristic (eta)
                pheromone_level = self.pheromone[current_city][city] ** self.alpha
                # Heuristic is 1 / distance, as we discussed
                heuristic_level = (1.0 / self.distances[current_city][city]) ** self.beta
                total_prob += pheromone_level * heuristic_level
        
        # Calculate the probability for each possible next city
        for city in range(self.n_cities):
            if city not in visited:
                pheromone_level = self.pheromone[current_city][city] ** self.alpha
                heuristic_level = (1.0 / self.distances[current_city][city]) ** self.beta
                prob = (pheromone_level * heuristic_level) / total_prob
                probabilities.append((city, prob))
        
        # This is the "weighted random" or "roulette wheel" selection
        cities, probs = zip(*probabilities)
        return random.choices(cities, weights=probs, k=1)[0]
    
    def _calculate_tour_length(self, tour):
        """
        Calculates the total length of a given tour.
        """
        length = 0
        for i in range(self.n_cities - 1):
            length += self.distances[tour[i]][tour[i+1]]
        length += self.distances[tour[-1]][tour[0]] # Return to start
        return length

# --- Example Usage ---
if __name__ == '__main__':
    # A simple 5-city problem distance matrix
    distances = [
        [0, 27, 20, 25, 16],
        [27, 0, 16, 17, 29],
        [20, 16, 0, 29, 23],
        [25, 17, 29, 0, 12],
        [16, 29, 23, 12, 0]
    ]
    
    # --- Parameters ---
    N_ANTS = 10
    N_ITERATIONS = 10
    ALPHA = 1.0  # Pheromone importance
    BETA = 5.0   # Heuristic importance
    EVAPORATION_RATE = 0.5
    
    aco = AntColonyOptimization(distances, N_ANTS, N_ITERATIONS, ALPHA, BETA, EVAPORATION_RATE)
    best_tour, best_length = aco.run()
    
    print("\n--- Results ---")
    print(f"Best tour found: {best_tour}")
    print(f"Best tour length: {best_length}")