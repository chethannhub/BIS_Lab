import numpy as np

class GreyWolfOptimizer:
    def __init__(self, func, lb, ub, dim, pop_size=30, max_iter=100):
        """
        :param func: Objective function to optimize
        :param lb: Lower bounds of the search space
        :param ub: Upper bounds of the search space
        :param dim: Number of dimensions (variables) in the search space
        :param pop_size: Number of wolves (population size)
        :param max_iter: Maximum number of iterations
        """
        self.func = func
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter

        self.position = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.func, 1, self.position)

        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float("inf")

        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float("inf")

        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float("inf")

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.pop_size):
               
                fitness_val = self.func(self.position[i])
                if fitness_val < self.alpha_score:
                    self.alpha_score = fitness_val
                    self.alpha_pos = self.position[i]
                elif fitness_val < self.beta_score:
                    self.beta_score = fitness_val
                    self.beta_pos = self.position[i]
                elif fitness_val < self.delta_score:
                    self.delta_score = fitness_val
                    self.delta_pos = self.position[i]

            a = 2 - t * (2 / self.max_iter)
            for i in range(self.pop_size):
                A1 = 2 * a * np.random.random(self.dim) - a
                C1 = 2 * np.random.random(self.dim)
                D_alpha = np.abs(C1 * self.alpha_pos - self.position[i])
                X1 = self.alpha_pos - A1 * D_alpha

                A2 = 2 * a * np.random.random(self.dim) - a
                C2 = 2 * np.random.random(self.dim)
                D_beta = np.abs(C2 * self.beta_pos - self.position[i])
                X2 = self.beta_pos - A2 * D_beta

                A3 = 2 * a * np.random.random(self.dim) - a
                C3 = 2 * np.random.random(self.dim)
                D_delta = np.abs(C3 * self.delta_pos - self.position[i])
                X3 = self.delta_pos - A3 * D_delta

                self.position[i] = (X1 + X2 + X3) / 3

                self.position[i] = np.clip(self.position[i], self.lb, self.ub)

            print(f"Iteration {t + 1}/{self.max_iter}, Best Score: {self.alpha_score}")

        return self.alpha_pos, self.alpha_score

def sphere_function(x):
    return np.sum(x**2)


lower_bound = -5.0
upper_bound = 5.0
dim = 30  
pop_size = 50
max_iter = 10


gwo = GreyWolfOptimizer(func=sphere_function, lb=lower_bound, ub=upper_bound, dim=dim, pop_size=pop_size, max_iter=max_iter)


best_position, best_score = gwo.optimize()


print(f"Best Position: {best_position}")
print(f"Best Score: {best_score}")