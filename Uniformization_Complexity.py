import numpy as np
import matplotlib.pyplot as plt

def uniformization_rate(Q):
    """
    Computes the uniformization rate (maximum absolute value of diagonal entries).
    """
    return np.max(-np.diag(Q))

def auxiliary_transition_matrix(Q, mu):
    """
    Computes the auxiliary transition probability matrix R for uniformization.
    """
    R = np.eye(len(Q)) + Q / mu
    return R

def expected_recursion_steps_uniformization(Q, T, a, b, mu, R):
    """
    Computes the expected number of recursion steps for Uniformization Sampling.
    """
    P_ab = np.exp(Q * T)[a, b]  # Transition probability from a to b in time T
    if P_ab == 0:
        return float('inf')  # Avoid division by zero
    E_L = (mu * T / P_ab) * np.dot(R @ np.exp(Q * T), np.eye(len(Q)))[a, b]
    return E_L

def cpu_time_uniformization(Q, T, a, b, mu, R, alpha, beta):
    """
    Computes the total CPU time for uniformization sampling.
    """
    E_L = expected_recursion_steps_uniformization(Q, T, a, b, mu, R)
    return alpha + beta * E_L

def plot_uniformization_complexity(Q, T_range, a, b, alpha, beta):
    """
    Plots the uniformization complexity figures.
    """
    mu = uniformization_rate(Q)
    R = auxiliary_transition_matrix(Q, mu)
    
    initialization_times = []
    recursion_steps = []
    sampling_cpu_times = []

    for T in T_range:
        E_L = expected_recursion_steps_uniformization(Q, T, a, b, mu, R)
        total_time = cpu_time_uniformization(Q, T, a, b, mu, R, alpha, beta)

        initialization_times.append(alpha)  # Constant initialization cost
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

    a, b = 0, 2  # Starting and ending states
    alpha = 1.05  # Initialization cost (higher for uniformization due to R matrix computation)
    beta = 0.09   # Cost per recursion step
    T_range = np.linspace(0.1, 3.0, 50)  # Time range

    plot_uniformization_complexity(Q, T_range, a, b, alpha, beta)