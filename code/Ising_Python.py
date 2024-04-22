import numpy as np
from numba import njit

@njit
def initialize_lattice(size):
    return np.random.choice([-1, 1], size=(size, size))

@njit
def energy_diff(lattice, i, j):
    return 2 * lattice[i, j] * (lattice[i-1, j] + lattice[i+1, j] + lattice[i, j-1] + lattice[i, j+1])

@njit
def monte_carlo_step(lattice, beta):
    for i in range(1, lattice.shape[0] - 1):
        for j in range(1, lattice.shape[1] - 1):
            if np.random.rand() < 1 - np.exp(-2 * beta * energy_diff(lattice, i, j)):
                lattice[i, j] *= -1

@njit
def compute_magnetization(lattice):
    return np.sum(lattice)

@njit
def compute_partition_function(lattice, beta, num_steps):
    partition_function = 0
    for _ in range(num_steps):
        monte_carlo_step(lattice, beta)
        partition_function += np.exp(-beta * compute_magnetization(lattice))
    return partition_function / num_steps

@njit
def estimate_errors(lattice, beta, num_steps, num_samples):
    magnetizations = np.zeros(num_samples)
    for i in range(num_samples):
        monte_carlo_step(lattice, beta)
        magnetizations[i] = compute_magnetization(lattice)
    average_magnetization = np.mean(magnetizations)
    variance = np.var(magnetizations)
    error_magnetization = np.sqrt(variance / num_samples)
    return average_magnetization, error_magnetization

def main():
    size = 10 # Size of the lattice
    beta = 1.0 # Inverse temperature
    num_steps = 1000 # Number of Monte Carlo steps per sample
    num_samples = 100 # Number of samples to estimate errors

    lattice = initialize_lattice(size)
    partition_function = compute_partition_function(lattice, beta, num_steps)
    average_magnetization, error_magnetization = estimate_errors(lattice, beta, num_steps, num_samples)

    print(f"Partition function: {partition_function}")
    print(f"Average magnetization: {average_magnetization}")
    print(f"Error in average magnetization: {error_magnetization}")

if __name__ == "__main__":
    main()