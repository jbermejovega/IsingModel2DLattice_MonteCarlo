#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 10 // Size of the lattice
#define NUM_STEPS 1000 // Number of Monte Carlo steps per sample
#define NUM_SAMPLES 100 // Number of samples to estimate errors

void initialize_lattice(int lattice[SIZE][SIZE]) {
    srand(time(NULL));
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            lattice[i][j] = rand() % 2 == 0 ? -1 : 1;
        }
    }
}

int energy_diff(int lattice[SIZE][SIZE], int i, int j) {
    return 2 * lattice[i][j] * (lattice[(i-1+SIZE)%SIZE][j] + lattice[(i+1)%SIZE][j] + lattice[i][(j-1+SIZE)%SIZE] + lattice[i][(j+1)%SIZE]);
}

void monte_carlo_step(int lattice[SIZE][SIZE], double beta) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int delta_energy = energy_diff(lattice, i, j);
            if (rand() / (double)RAND_MAX < exp(-2 * beta * delta_energy)) {
                lattice[i][j] *= -1;
            }
        }
    }
}

int compute_magnetization(int lattice[SIZE][SIZE]) {
    int magnetization = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            magnetization += lattice[i][j];
        }
    }
    return magnetization;
}

double compute_partition_function(int lattice[SIZE][SIZE], double beta, int num_steps) {
    double partition_function = 0;
    for (int step = 0; step < num_steps; step++) {
        monte_carlo_step(lattice, beta);
        partition_function += exp(-beta * compute_magnetization(lattice));
    }
    return partition_function / num_steps;
}

void estimate_errors(int lattice[SIZE][SIZE], double beta, int num_steps, int num_samples) {
    double magnetizations[NUM_SAMPLES];
    for (int i = 0; i < num_samples; i++) {
        monte_carlo_step(lattice, beta);
        magnetizations[i] = compute_magnetization(lattice);
    }
    double average_magnetization = 0;
    for (int i = 0; i < num_samples; i++) {
        average_magnetization += magnetizations[i];
    }
    average_magnetization /= num_samples;

    double variance = 0;
    for (int i = 0; i < num_samples; i++) {
        variance += (magnetizations[i] - average_magnetization) * (magnetizations[i] - average_magnetization);
    }
    variance /= num_samples;

    double error_magnetization = sqrt(variance);

    printf("Partition function: %f\n", compute_partition_function(lattice, beta, num_steps));
    printf("Average magnetization: %f\n", average_magnetization);
    printf("Error in average magnetization: %f\n", error_magnetization);
}

int main() {
    int lattice[SIZE][SIZE];
    double beta = 1.0; // Inverse temperature

    initialize_lattice(lattice);
    estimate_errors(lattice, beta, NUM_STEPS, NUM_SAMPLES);

    return 0;
}