import numpy as np
import matplotlib.pyplot as plt

def eigen_decomposition(Q):
    """ Performs eigenvalue decomposition of the rate matrix Q.
        Returns the eigenvalues and eigenvectors for initialization cost. """
    eigenvalues, eigenvectors = np.linalg.eig(Q)
    return eigenvalues, eigenvectors

def expected_recursion_steps_direct(Q, T, a, b):
    """ Computes the expected number of recursion steps for Direct Sampling. """
    if a == b:
        return -Q[a, a] * T  # Expected state changes when a == b
    else:
        return -Q[a, a] * T  # Similar for transition cases, scaled with time T

def cpu_time_direct(Q, T, a, b, alpha, beta):
    """ Computes the total CPU time for direct sampling. """
    E_L = expected_recursion_steps_direct(Q, T, a, b)
    return alpha + beta * E_L

def plot_direct_sampling(Q, T_range, a, b, alpha, beta):
    """ Plots the direct sampling complexity figures. """
    initialization_times = []
    recursion_steps = []
    sampling_cpu_times = []

    # Compute initialization cost using eigen decomposition
    eigenvalues, eigenvectors = eigen_decomposition(Q)
    initialization_cost = np.abs(eigenvalues).sum()

    for T in T_range:
        E_L = expected_recursion_steps_direct(Q, T, a, b)
        total_time = cpu_time_direct(Q, T, a, b, alpha, beta)

        initialization_times.append(initialization_cost)
        recursion_steps.append(E_L)
        sampling_cpu_times.append(total_time)

    # Plot the figures
    plt.figure(figsize=(14, 8))

    # Initialization CPU Time
    plt.subplot(2, 2, 1)
    plt.plot(T_range, initialization_times, label="Initialization CPU Time", color="orange")
    plt.xlabel("Time (T)")
    plt.ylabel("CPU Time")
    plt.title("Initialization CPU Time")
    plt.grid()
    plt.legend()

    # Expected Number of Recursions
    plt.subplot(2, 2, 2)
    plt.plot(T_range, recursion_steps, label="Expected Recursion Steps", color="green")
    plt.xlabel("Time (T)")
    plt.ylabel("Steps")
    plt.title("Expected Number of Recursions")
    plt.grid()
    plt.legend()

    # Total CPU Time
    plt.subplot(2, 2, 3)
    plt.plot(T_range, sampling_cpu_times, label="Total CPU Time", color="red")
    plt.xlabel("Time (T)")
    plt.ylabel("CPU Time")
    plt.title("Total CPU Time")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Example Q matrix (rate matrix)
    Q = np.array([
        [-1.0,  0.5,  0.5],
        [ 0.3, -0.8,  0.5],
        [ 0.2,  0.6, -0.8]
    ])

    a, b = 0, 0  # Starting and ending states
    alpha = 0.85  # Initialization cost (higher for direct sampling due to eigen decomposition)
    beta = 0.56   # Cost per recursion step
    T_range = np.linspace(0.1, 3.0, 50)  # Time range

    plot_direct_sampling(Q, T_range, a, b, alpha, beta)