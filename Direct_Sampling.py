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

def eigenvalue_decomposition(Q):
    """Perform eigenvalue decomposition of the rate matrix Q."""
    eigenvalues, eigenvectors = np.linalg.eig(Q)
    U = eigenvectors
    D = np.diag(eigenvalues)
    U_inv = np.linalg.inv(U)
    return U, D, U_inv

def sample_first_state(Q, a, b, T, U, D, U_inv):
    """Sample the first state transition time and next state."""
    Q_a = -Q[a, a]  # Total rate of leaving state `a`
    p_a = np.exp(-Q_a * T) / np.sum(U[a, :] @ np.diag(np.exp(D.diagonal() * T)) @ U_inv[:, b])

    if np.random.rand() < p_a:
        return a, T

    next_state_probs = Q[a, :] / Q_a
    next_state_probs[a] = 0  # Exclude self-transition
    next_state_probs /= next_state_probs.sum()  # Normalize
    next_state = np.random.choice(len(Q), p=next_state_probs)

    waiting_time = np.random.exponential(1 / Q_a)
    return next_state, waiting_time

def direct_sampling(Q, a, b, T):
    """Perform Direct Sampling for a CTMC."""
    U, D, U_inv = eigenvalue_decomposition(Q)
    path = [(0, a)]
    current_time = 0
    current_state = a

    while current_time < T:
        next_state, waiting_time = sample_first_state(Q, current_state, b, T - current_time, U, D, U_inv)
        next_time = current_time + waiting_time

        if next_time >= T:
            path.append((T, current_state))
            break

        path.append((next_time, next_state))
        current_time = next_time
        current_state = next_state

    return path

def plot_direct_sampling(Q, a, b, T, num_paths):
    """Generate and plot paths for Direct Sampling."""
    plt.figure(figsize=(8, 6))
    for i in range(num_paths):
        path = direct_sampling(Q, a, b, T)
        times, states = zip(*path)
        plt.step(times, states, where='post', label=f"Path {i+1}")

    plt.title("Direct Sampling Paths")
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
    print("Direct Sampling - Example Transition Matrix:")
    print(Q)

    a = 0  # Initial state
    b = 2  # Final state
    T = 10.0  # Total time
    num_paths = 5 # No of paths to be printed

    plot_direct_sampling(Q, a, b, T, num_paths)