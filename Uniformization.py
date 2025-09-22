import numpy as np
import matplotlib.pyplot as plt

def validate_transition_matrix(Q):
    """Validates the input transition matrix Q."""
    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Transition matrix must be square.")
    N = Q.shape[0]
    if np.any(Q - np.diag(np.diag(Q)) < 0):
        raise ValueError("Off-diagonal elements must be non-negative.")
    for i in range(N):
        if not np.isclose(Q[i, i], -np.sum(Q[i, :]) + Q[i, i]):
            raise ValueError(f"Row {i} does not sum to zero.")

def uniformization_sampling(Q, a, b, T, max_iterations=1000):
    """
    Perform Uniformization Sampling for a CTMC.

    Parameters:
        Q: np.ndarray
            Transition rate matrix (NxN).
        a: int
            Initial state (0-indexed).
        b: int
            Final state (0-indexed).
        T: float
            Total time duration.
        max_iterations: int
            Maximum number of iterations to avoid infinite loops.

    Returns:
        list of tuples
            A list of state transitions [(time, state), ...].
    """
    N = Q.shape[0]
    mu = np.max(-np.diag(Q))  # Uniformization rate
    R = np.eye(N) + Q / mu  # Transition probability matrix

    for _ in range(max_iterations):
        path = [(0, a)]
        current_time = 0
        current_state = a

        # Sample the number of transitions from a Poisson distribution
        num_transitions = np.random.poisson(mu * T)

        if num_transitions == 0:
            path.append((T, current_state))
            if current_state == b:
                return path
            continue

        # Uniformly distribute transition times
        transition_times = np.sort(np.random.uniform(0, T, num_transitions))

        for t in transition_times:
            # Sample the next state from the transition matrix R
            next_state_probs = R[current_state, :]
            next_state = np.random.choice(N, p=next_state_probs)

            path.append((t, next_state))
            current_state = next_state

        # Ensure the last state is at time T
        path.append((T, current_state))

        # Check if the final state matches the desired state
        if path[-1][1] == b:
            return path

    raise RuntimeError(f"Failed to generate a valid path after {max_iterations} iterations.")

def plot_uniformization_sampling(Q, a, b, T, num_paths):
    """Generate and plot paths for Uniformization Sampling."""
    plt.figure(figsize=(8, 6))
    for i in range(num_paths):
        try:
            path = uniformization_sampling(Q, a, b, T)
            times, states = zip(*path)
            plt.step(times, states, where='post', label=f"Path {i+1}")
        except RuntimeError:
            print(f"Path {i+1} failed to generate.")

    plt.title("Uniformization Sampling Paths")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Q = np.array([
        [-0.5,  0.3,  0.2],
        [ 0.1, -0.4,  0.3],
        [ 0.2,  0.1, -0.3]
    ])
    print("Uniformization Sampling - Example Transition Matrix:")
    print(Q)

    a = 0  # Initial state
    b = 2  # Final state
    T = 5.0  # Total time
    num_paths = 5 # No of paths to be printed

    plot_uniformization_sampling(Q, a, b, T, num_paths)