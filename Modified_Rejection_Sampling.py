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

def sample_waiting_time(rate):
    """Sample waiting time from an exponential distribution."""
    if rate <= 0:
        return float('inf')
    return np.random.exponential(1 / rate)

def sample_next_state(current_state, Q, Q_a):
    """Sample the next state based on transition probabilities."""
    transition_probs = Q[current_state, :] / Q_a[current_state]
    transition_probs[current_state] = 0  # Exclude self-transitions
    transition_probs /= transition_probs.sum()  # Normalize
    return np.random.choice(len(Q), p=transition_probs)

def modified_rejection_sampling(Q, a, b, T, max_iterations=1000):
    """Perform Modified Rejection Sampling for a CTMC."""
    N = Q.shape[0]
    Q_a = -np.diag(Q)  # Total rates for each state

    for _ in range(max_iterations):
        path = [(0, a)]
        current_time = 0
        current_state = a

        while current_time < T:
            waiting_time = sample_waiting_time(Q_a[current_state])
            next_time = current_time + waiting_time

            if next_time >= T:
                path.append((T, current_state))
                break

            next_state = sample_next_state(current_state, Q, Q_a)
            path.append((next_time, next_state))
            current_state = next_state
            current_time = next_time

        if path[-1][1] == b:
            return path

    raise RuntimeError(f"Failed to generate a valid path after {max_iterations} iterations.")

def plot_modified_rejection_sampling(Q, a, b, T, num_paths):
    """Generate and plot paths for Modified Rejection Sampling."""
    plt.figure(figsize=(8, 6))
    for i in range(num_paths):
        try:
            path = modified_rejection_sampling(Q, a, b, T)
            times, states = zip(*path)
            plt.step(times, states, where='post', label=f"Path {i+1}")
        except RuntimeError:
            print(f"Path {i+1} failed to generate.")

    plt.title("Modified Rejection Sampling Paths")
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
    print("Modified Rejection Sampling - Example Transition Matrix:")
    print(Q)

    a = 0  # Initial state
    b = 2  # Final state
    T = 5.0  # Total time
    num_paths = 5 # No of paths to be printed

    plot_modified_rejection_sampling(Q, a, b, T, num_paths)
