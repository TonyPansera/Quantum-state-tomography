#!/usr/bin/env python3
"""
Dataset generator for 1-qubit quantum state tomography (regression).
- Uses QuTiP to construct qubit states
- Generates random pure states uniformly on the Bloch sphere
- Simulates shot measurements in X, Y, Z
- Outputs a CSV file with:
    features:  X_mean, Y_mean, Z_mean
    targets:   theta, phi, cos_phi, sin_phi
    (plus ideal expectation values for reference)
"""

import numpy as np
import pandas as pd
from qutip import basis, sigmax, sigmay, sigmaz, expect, Qobj

# ----------------------------
# Global parameters
# ----------------------------

N_STATES = 5000        # number of different states on the Bloch sphere
N_SHOTS = 100          # number of measurement shots per observable (X,Y,Z)
RANDOM_SEED = 42       # for reproducibility
OUTPUT_CSV = "../data/qst_mle_dataset_mixed_states.csv"


# ----------------------------
# State generation
# ----------------------------

def sample_bloch_angles():
    """
    Sample (theta, phi) uniformly on the Bloch sphere.

    Sampling rule:
        u ~ Uniform[-1,1]
        theta = arccos(u)
        phi ~ Uniform[0, 2π)
    """
    # We want a direction uniformly distributed on the sphere.
    # The surface element is: dA = sin(theta) * d(theta) * d(phi)
    # Therefore theta does NOT have a uniform density: p(theta) proportional to sin(theta)
    # To obtain the correct law, set u = cos(theta)
    # Then: du = -sin(theta) * d(theta)  ->  uniform density in u
    # So sampling u ~ Uniform[-1, 1] gives the correct p(theta)
    u = np.random.uniform(-1.0, 1.0)
    theta = np.arccos(u)
    
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    return theta, phi

def sample_bloch_vector():
    """
    Sample (theta, phi, r) uniformly to get a qubit state (pure or mixed).

    Sampling rule:
        u ~ Uniform[-1,1]
        theta = arccos(u)
        phi ~ Uniform[0, 2π)
        r ~ Uniform[0,1]
    """
    # We want a direction uniformly distributed on the sphere.
    # The surface element is: dA = sin(theta) * d(theta) * d(phi)
    # Therefore theta does NOT have a uniform density: p(theta) proportional to sin(theta)
    # To obtain the correct law, set u = cos(theta)
    # Then: du = -sin(theta) * d(theta)  ->  uniform density in u
    # So sampling u ~ Uniform[-1, 1] gives the correct p(theta)
    u = np.random.uniform(-1.0, 1.0)
    theta = np.arccos(u)
    r = np.random.uniform(0, 1)
    
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    return theta, phi, r

def sample_bloch_angles_near_poles(epsilon=0.2):
    """
    Sample (theta, phi) with theta around 0 or pi and phi ~ Uniform[0, 2π)
    """
    if np.random.rand() < 0.5:
        # Near the north pole (theta ≈ 0)
        theta = np.random.uniform(0, epsilon)
    else:
        # Near the south pole (theta ≈ π)
        theta = np.random.uniform(np.pi - epsilon, np.pi)
    
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    return theta, phi

def sample_bloch_angles_outside_poles(epsilon=0.2):
    """
    Sample (theta, phi) with theta around 0 or pi and phi ~ Uniform[0, 2π)
    """

    theta = np.random.uniform(np.pi/2 - epsilon, np.pi/2 + epsilon)
    
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    return theta, phi

def ket_from_angles(theta, phi):
    """
    Construct a 1-qubit pure state |psi(theta, phi)> in computational basis:

        |psi> = cos(theta/2) |0> + e^{i phi} sin(theta/2) |1>
    """
    c0 = np.cos(theta / 2.0)
    c1 = np.sin(theta / 2.0) * np.exp(1j * phi)
    # |0> and |1> in computational basis
    zero = basis(2, 0)
    one = basis(2, 1)
    ket = c0 * zero + c1 * one
    return ket.unit()


# ----------------------------
# Measurement simulation
# ----------------------------

# Pauli operators
SIGMA_X = sigmax()
SIGMA_Y = sigmay()
SIGMA_Z = sigmaz()


def simulate_pauli_measurements(ket, n_shots):
    """
    Simulate finite-shot projective measurements in the Pauli bases X, Y, Z.

    For each observable {X, Y, Z}:
        - Compute ideal expectation value <O> = Tr(rho(O))
        - Use Bernoulli sampling with:
              P(+1) = (1 + <O>) / 2
              P(-1) = (1 - <O>) / 2
        - Estimate empirical mean over n_shots.

    Returns: X_mean, Y_mean, Z_mean, X_ideal, Y_ideal, Z_ideal
    """
    # Density matrix
    rho = ket * ket.dag()  

    # ! IDEAL ! expectation values (from a random Bloch vector)
    ex_x = float(expect(SIGMA_X, rho))
    ex_y = float(expect(SIGMA_Y, rho))
    ex_z = float(expect(SIGMA_Z, rho))

    def sample_mean_from_expectation(exp_val, shots):
        p_plus = (1.0 + exp_val) / 2.0
        
        # Draw shots in {+1, -1}
        outcomes = []

        for _ in range(shots):
            if np.random.rand() < p_plus:
                outcomes.append(1.0)
            else:
                outcomes.append(-1.0)
        return np.mean(outcomes)

    # Empirical means from shots
    x_mean = sample_mean_from_expectation(ex_x, n_shots)
    y_mean = sample_mean_from_expectation(ex_y, n_shots)
    z_mean = sample_mean_from_expectation(ex_z, n_shots)

    return x_mean, y_mean, z_mean, ex_x, ex_y, ex_z


def bloch_vector_from_thetaphi_r(theta, phi, r):
    # theta, phi en radians
    rx = r * np.sin(theta) * np.cos(phi)
    ry = r * np.sin(theta) * np.sin(phi)
    rz = r * np.cos(theta)
    return np.array([rx, ry, rz])

def simulate_pauli_measurements_mixed_states(theta, phi, r, n_shots):
    # Density matrix
    rho00 = 0.5 * (1 + r * np.cos(theta))
    rho11 = 0.5 * (1 - r * np.cos(theta))
    rho01 = 0.5 * (r * np.sin(theta) * np.exp(-1j * phi))
    rho10 = np.conjugate(rho01)
    
    rho_mat = np.array([
        [rho00, rho01],
        [rho10, rho11]
    ], dtype=complex)
    
    rho = Qobj(rho_mat)

    # ! IDEAL ! expectation values (from a random Bloch vector)
    ex_x = float(expect(SIGMA_X, rho))
    ex_y = float(expect(SIGMA_Y, rho))
    ex_z = float(expect(SIGMA_Z, rho))

    def sample_mean_from_expectation(exp_val, shots):
        p_plus = (1.0 + exp_val) / 2.0
        
        # Draw shots in {+1, -1}
        outcomes = []

        for _ in range(shots):
            if np.random.rand() < p_plus:
                outcomes.append(1.0)
            else:
                outcomes.append(-1.0)
        return np.mean(outcomes)

    # Empirical means from shots
    x_mean = sample_mean_from_expectation(ex_x, n_shots)
    y_mean = sample_mean_from_expectation(ex_y, n_shots)
    z_mean = sample_mean_from_expectation(ex_z, n_shots)

    return x_mean, y_mean, z_mean, ex_x, ex_y, ex_z


# ----------------------------
# Dataset generation
# ----------------------------

def generate_dataset(n_states, n_shots):
    """
    Generate a dataset for regression quantum state tomography of 1 qubit.

    For each state k:
        - Sample (theta_k, phi_k) uniformly on the Bloch sphere.
        - Build |psi_k>.
        - Simulate measurements in X, Y, Z with n_shots.
        - Store:
            - X_mean, Y_mean, Z_mean
            - theta, phi, cos_phi, sin_phi
            - X_ideal, Y_ideal, Z_ideal
    """
    rows = []

    for k in range(n_states):
        theta, phi, r = sample_bloch_vector()

        (x_mean, y_mean, z_mean, x_ideal, y_ideal, z_ideal) = simulate_pauli_measurements_mixed_states(theta, phi, r, n_shots)
        
        row = {
            "X_mean": x_mean,
            "Y_mean": y_mean,
            "Z_mean": z_mean,
            "theta_ideal": theta,
            "phi_ideal": phi,
            "r_ideal": r,
            "cos_phi_ideal": np.cos(phi),
            "sin_phi_ideal": np.sin(phi),
            "cos_theta_ideal": np.cos(theta),
            "sin_theta_ideal": np.sin(theta),
            "X_ideal": x_ideal,
            "Y_ideal": y_ideal,
            "Z_ideal": z_ideal,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ----------------------------
# Main entry point
# ----------------------------
np.random.seed(RANDOM_SEED)

print("Generating dataset for 1-qubit QST:")
print(f"  - Number of states : {N_STATES}")
print(f"  - Shots per Pauli  : {N_SHOTS}")

df = generate_dataset(N_STATES, N_SHOTS)

print("\nSample of generated data:")
print(df.head())

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDataset saved to: {OUTPUT_CSV}")

