# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np
from scipy import signal
from collections import deque


class KalmanFilter:
    # Changed to a 2D Constant Velocity Kalman Filter
    def __init__(self, process_variance, measurement_variance, dt=1.0, initial_value=0.0, initial_velocity=0.0):
        # State vector: [position, velocity]
        self.x_hat = np.array([initial_value, initial_velocity]) # x_hat: estimated state [position, velocity]

        # Process noise covariance (Q) - adapted for constant velocity model
        # Q = [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] * sigma_a^2 (where sigma_a is acceleration noise std dev)
        # For simplicity, we'll use a simplified Q based on provided process_variance.
        # A more rigorous approach would derive Q from acceleration noise.
        # Let's assume process_variance is related to the overall uncertainty in the state transition.
        # We'll use a diagonal Q for simplicity, but a full matrix is better.
        # For a constant velocity model, Q is typically a 2x2 matrix.
        # Let's approximate it as:
        # Q_pos = process_variance * dt**2 / 2
        # Q_vel = process_variance * dt**2 / 2
        # For now, let's simplify and assume process_variance is a scalar that influences Q.
        # A common approach for CV model Q:
        sigma_a_sq = process_variance # Assuming process_variance is related to acceleration noise variance
        self.Q = np.array([
            [dt**4/4, dt**3/2],
            [dt**3/2, dt**2]
        ]) * sigma_a_sq

        # Measurement noise covariance (R) - assuming we only measure position
        self.R = np.array([[measurement_variance]]) # R is 1x1 since we measure only position

        # Error covariance (P)
        self.P = np.eye(2) * 1.0 # Initialize P as identity matrix for a 2D state

        self.dt = dt # Time step

    # Modified update method to accept prediction_hint and use 2D state
    def update(self, measurement, prediction_hint=None):
        # Prediction step
        # State transition matrix (F) for constant velocity model
        # F = [[1, dt], [0, 1]]
        F = np.array([[1, self.dt], [0, 1]])

        # Predict state
        x_hat_minus = F @ self.x_hat
        if prediction_hint is not None:
            # Incorporate prediction hint from polynomial fitting
            # Directly use the hint to influence the predicted position
            x_hat_minus[0] = prediction_hint
            # Adjust velocity based on the difference from previous estimate
            # This is a heuristic to integrate the prediction without fully overriding the Kalman model
            if self.x_hat[0] != 0: # Avoid division by zero
                x_hat_minus[1] = (prediction_hint - self.x_hat[0]) / self.dt

        # Predict error covariance
        P_minus = F @ self.P @ F.T + self.Q

        # Update step
        # Measurement matrix (H) - we measure position only
        # H = [[1, 0]]
        H = np.array([[1, 0]])

        # Kalman gain (K)
        K = P_minus @ H.T @ np.linalg.inv(H @ P_minus @ H.T + self.R)

        # Update state estimate
        self.x_hat = x_hat_minus + K @ (measurement - H @ x_hat_minus)

        # Update error covariance
        self.P = (np.eye(2) - K @ H) @ P_minus

        return self.x_hat[0] # Return the estimated position (first element of state)

# Modified adaptive_filter to use the 2D Kalman Filter
def adaptive_filter(x, window_size=20, process_variance=0.01, measurement_variance=0.1):
    """
    Adaptive signal processing algorithm using Kalman Filter.
    This version now uses the Constant Velocity Kalman Filter.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples) - used for initialization, not direct windowing
        process_variance: Process noise variance for Kalman filter
        measurement_variance: Measurement noise variance for Kalman filter

    Returns:
        y: Filtered output signal with length = len(x)
    """
    if len(x) == 0:
        return np.array([])

    y = np.zeros(len(x))
    # Initialize Kalman filter with first measurement, assuming initial velocity is 0
    kf = KalmanFilter(process_variance, measurement_variance, initial_value=x[0], initial_velocity=0.0)

    for i in range(len(x)):
        y[i] = kf.update(x[i])

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20, process_variance=0.01, measurement_variance=0.1):
    """
    Enhanced version with trend preservation using an adaptive Kalman filter.
    The window_size is used to estimate initial noise characteristics.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples) - used for initial noise estimation
        process_variance: Base process noise variance for Kalman filter
        measurement_variance: Base measurement noise variance for Kalman filter

    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        # Fallback to simple Kalman if not enough data for initial variance estimation
        return adaptive_filter(x, window_size, process_variance, measurement_variance)

    y = np.zeros(len(x))

    # Estimate initial measurement variance from the first window
    initial_measurement_variance = np.var(x[:window_size]) if window_size > 1 else measurement_variance
    if initial_measurement_variance == 0: # Avoid division by zero if initial window is flat
        initial_measurement_variance = measurement_variance

    # Initialize Kalman filter with initial estimates for Q and R
    # Use initial_velocity=0.0 for the CV Kalman Filter
    kf = KalmanFilter(process_variance, initial_measurement_variance, initial_value=x[0], initial_velocity=0.0)

    # Keep track of recent measurements for adaptive parameter estimation and polynomial fitting
    recent_measurements = deque(x[:window_size], maxlen=window_size)
    # Store filtered history for polynomial fitting to get a smoother trend estimate
    filtered_history = deque([x[0]] * window_size, maxlen=window_size)

    for i in range(len(x)):
        prediction_hint = None
        if i >= window_size:
            recent_measurements.append(x[i])
            current_raw_window = np.array(recent_measurements)
            current_filtered_window = np.array(filtered_history)

            # Adapt measurement_variance (R) based on local signal volatility
            local_measurement_variance = np.var(current_raw_window)
            # Ensure R doesn't become too small, and is at least a fraction of the base variance
            kf.R[0,0] = max(measurement_variance * 0.1, local_measurement_variance) if local_measurement_variance > 0 else measurement_variance

            # Adapt process_variance (Q) based on local rate of change volatility
            if len(current_raw_window) > 1:
                local_diff_variance = np.var(np.diff(current_raw_window))
                # Scale local_diff_variance to influence process_variance (Q)
                # Ensure Q doesn't drop too low or go excessively high
                # Heuristic scaling factors: 0.1 for min, 10 for max, 0.5 for scaling diff variance
                sigma_a_sq_adapted = max(process_variance * 0.1, min(process_variance * 10, local_diff_variance * 0.5))
                kf.Q = np.array([
                    [kf.dt**4/4, kf.dt**3/2],
                    [kf.dt**3/2, kf.dt**2]
                ]) * sigma_a_sq_adapted
            else:
                # Revert to base Q if not enough data for difference variance calculation
                kf.Q = np.array([
                    [kf.dt**4/4, kf.dt**3/2],
                    [kf.dt**3/2, kf.dt**2]
                ]) * process_variance

            # Predictive enhancement using polynomial fitting on filtered history
            if len(current_filtered_window) >= 2: # Need at least 2 points for linear fit
                # Time points for fitting (relative to current point)
                t_fit = np.arange(-window_size + 1, 1) # e.g., [-19, ..., 0] for current point
                # Fit a linear polynomial (degree 1) to the recent filtered data
                coeffs = np.polyfit(t_fit, current_filtered_window, 1)
                # Extrapolate one step into the future (t=1)
                prediction_hint = np.polyval(coeffs, 1)

        # Pass the prediction_hint to the KF update method
        y[i] = kf.update(x[i], prediction_hint)
        # Update filtered history for the next iteration's prediction
        filtered_history.append(y[i])

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies the selected algorithm.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    else:
        return adaptive_filter(input_signal, window_size)


# EVOLVE-BLOCK-END


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    """
    Generate synthetic test signal with known characteristics.

    Args:
        length: Length of the signal
        noise_level: Standard deviation of noise to add
        seed: Random seed for reproducibility

    Returns:
        Tuple of (noisy_signal, clean_signal)
    """
    np.random.seed(seed)
    t = np.linspace(0, 10, length)

    # Create a complex signal with multiple components
    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)  # Low frequency component
        + 1.5 * np.sin(2 * np.pi * 2 * t)  # Medium frequency component
        + 0.5 * np.sin(2 * np.pi * 5 * t)  # Higher frequency component
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)  # Decaying oscillation
    )

    # Add non-stationary behavior
    trend = 0.1 * t * np.sin(0.2 * t)  # Slowly varying trend
    clean_signal += trend

    # Add random walk component for non-stationarity
    random_walk = np.cumsum(np.random.randn(length) * 0.05)
    clean_signal += random_walk

    # Add noise
    noise = np.random.normal(0, noise_level, length)
    noisy_signal = clean_signal + noise

    return noisy_signal, clean_signal


def run_signal_processing(signal_length=1000, noise_level=0.3, window_size=20):
    """
    Run the signal processing algorithm on a test signal.

    Returns:
        Dictionary containing results and metrics
    """
    # Generate test signal
    noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)

    # Process the signal
    filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    # Calculate basic metrics
    if len(filtered_signal) > 0:
        # Align signals for comparison (account for processing delay)
        delay = window_size - 1
        aligned_clean = clean_signal[delay:]
        aligned_noisy = noisy_signal[delay:]

        # Ensure same length
        min_length = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:min_length]
        aligned_clean = aligned_clean[:min_length]
        aligned_noisy = aligned_noisy[:min_length]

        # Calculate correlation with clean signal
        correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1] if min_length > 1 else 0

        # Calculate noise reduction
        noise_before = np.var(aligned_noisy - aligned_clean)
        noise_after = np.var(filtered_signal - aligned_clean)
        noise_reduction = (noise_before - noise_after) / noise_before if noise_before > 0 else 0

        return {
            "filtered_signal": filtered_signal,
            "clean_signal": aligned_clean,
            "noisy_signal": aligned_noisy,
            "correlation": correlation,
            "noise_reduction": noise_reduction,
            "signal_length": min_length,
        }
    else:
        return {
            "filtered_signal": [],
            "clean_signal": [],
            "noisy_signal": [],
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": 0,
        }


if __name__ == "__main__":
    # Test the algorithm
    results = run_signal_processing()
    print(f"Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")
