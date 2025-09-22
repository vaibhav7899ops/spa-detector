import numpy as np
import matplotlib.pyplot as plt

def acceptance_probability(Q, T, a, b):
    """
    Computes the acceptance probability for rejection sampling.
    """
    if a == b:
        return np.exp(Q[a, a] * T)  # For same starting and ending states
    else:
        P_ab = Q[a, b] / abs(Q[a, a]) * (1 - np.exp(Q[a, a] * T))  # Transition probability
        P_aa = np.exp(Q[a, a] * T)
        return P_ab / (1 - P_aa)  # Acceptance probability for a != b

def expected_recursion_steps(Q, T, a, b):
    """
    Computes the expected number of recursion steps.
    """
    if a == b:
        return 1 / acceptance_probability(Q, T, a, b)
    else:
        P_ab = Q[a, b] / abs(Q[a, a]) * (1 - np.exp(Q[a, a] * T))
        P_aa = np.exp(Q[a, a] * T)
        return 1 / (P_ab / (1 - P_aa))

def cpu_time(Q, T, a, b, alpha, beta):
    """
    Computes the total CPU time spent on rejection sampling.
    """
    p_acc = acceptance_probability(Q, T, a, b)
    E_L = expected_recursion_steps(Q, T, a, b)
    return (alpha + beta * E_L) / p_acc

def plot_rejection_sampling(Q, T_range, a, b, alpha, beta):
    """
    Plots the rejection sampling complexity figures.
    """
    acceptance_probs = []
    initialization_times = []
    recursion_steps = []
    sampling_cpu_times = []

    for T in T_range:
        p_acc = acceptance_probability(Q, T, a, b)
        E_L = expected_recursion_steps(Q, T, a, b)
        total_time = cpu_time(Q, T, a, b, alpha, beta)
        
        acceptance_probs.append(p_acc)
        initialization_times.append(alpha)
        recursion_steps.append(E_L)
        sampling_cpu_times.append(total_time)

    # Plot the figures
    plt.figure(figsize=(14, 8))

    # Acceptance Probability
    plt.subplot(2, 2, 1)
    plt.plot(T_range, acceptance_probs, label="Acceptance Probability")
    plt.xlabel("Time (T)")
    plt.ylabel("Probability")
    plt.title("Acceptance Probability")
    plt.grid()
    plt.legend()

    # Initialization CPU Time
    plt.subplot(2, 2, 2)
    plt.plot(T_range, initialization_times, label="Initialization CPU Time", color="orange")
    plt.xlabel("Time (T)")
    plt.ylabel("CPU Time")
    plt.title("Initialization CPU Time")
    plt.grid()
    plt.legend()

    # Expected Number of Recursions
    plt.subplot(2, 2, 3)
    plt.plot(T_range, recursion_steps, label="Expected Recursion Steps", color="green")
    plt.xlabel("Time (T)")
    plt.ylabel("Steps")
    plt.title("Expected Number of Recursions")
    plt.grid()
    plt.legend()

    # Total CPU Time
    plt.subplot(2, 2, 4)
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
    alpha = 0.05  # Initialization cost
    beta = 0.04   # Cost per recursion step
    T_range = np.linspace(0.1, 3.0, 50)  # Time range

    plot_rejection_sampling(Q, T_range, a, b, alpha, beta)