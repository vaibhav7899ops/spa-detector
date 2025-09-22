import numpy as np
import matplotlib.pyplot as plt

def calculate_efficiency(method, T, acceptance_probability=None):
    """
    Simulate computational efficiency based on the sampling method and time.

    Parameters:
        method (str): Sampling method ('rejection', 'direct', 'uniformization').
        T (float): Evolutionary distance (time).
        acceptance_probability (float): Probability of acceptance (for rejection sampling).

    Returns:
        float: Simulated efficiency (CPU time).
    """
    if method == "rejection":
        if acceptance_probability is None:
            raise ValueError("Acceptance probability must be provided for rejection sampling.")
        return 1 / acceptance_probability if acceptance_probability > 0 else float('inf')

    elif method == "direct":
        return 1 + 0.5 * T  # Scales linearly with time

    elif method == "uniformization":
        return 1 + 0.8 * T  # Overhead for virtual substitutions

    else:
        raise ValueError("Invalid method specified.")

def simulate_example(example, T_values):
    """
    Simulate computational efficiency for different methods for a given example.

    Parameters:
        example (int): Example number (1, 2, or 3).
        T_values (list): List of evolutionary distances.

    Returns:
        dict: Efficiency data for each sampling method.
    """
    data = {"T": T_values, "rejection": [], "direct": [], "uniformization": []}
    for T in T_values:
        if example == 1:  # Example 1
            acceptance_probability = max(0.1, 1 - T / 10)
        elif example == 2:  # Example 2
            acceptance_probability = max(0.05, 1 - T / 20)
        elif example == 3:  # Example 3
            acceptance_probability = 0.9 if T < 0.3 else (0.5 if T < 0.9 else 0.2)
        else:
            raise ValueError("Invalid example specified.")

        # Calculate efficiencies
        data["rejection"].append(calculate_efficiency("rejection", T, acceptance_probability))
        data["direct"].append(calculate_efficiency("direct", T))
        data["uniformization"].append(calculate_efficiency("uniformization", T))

    return data

def plot_all_examples(T_values):
    """
    Plot computational efficiency for Examples 1, 2, and 3 in a single window.

    Parameters:
        T_values (list): List of evolutionary distances.
    """
    examples = [1, 2, 3]
    titles = [
        "Example 1: Nucleotide-Level Evolution",
        "Example 2: Codon-Level Evolution",
        "Example 3: Sequence-Level Evolution"
    ]

    plt.figure(figsize=(18, 6))

    for i, example in enumerate(examples, 1):
        data = simulate_example(example, T_values)
        plt.subplot(1, 3, i)
        plt.plot(data["T"], data["rejection"], label="Rejection Sampling", marker='o')
        plt.plot(data["T"], data["direct"], label="Direct Sampling", marker='s')
        plt.plot(data["T"], data["uniformization"], label="Uniformization", marker='^')
        plt.title(titles[i - 1])
        plt.xlabel("Evolutionary Distance (T)")
        plt.ylabel("CPU Time (Arbitrary Units)")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    T_values = np.linspace(0.1, 2.0, 20)
    plot_all_examples(T_values)