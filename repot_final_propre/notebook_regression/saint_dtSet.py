import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time


# 1) Qubit Tomography Dataset Generation


def generate_qubit_tomography_dataset_base(
    n_states: int,
    n_shots: int,
    mode: str = "finite_shots",      # "finite_shots": binomial sampling; "ideal": no statistical noise
    include_ideal: bool = True,      # store theta_ideal, phi_ideal, X_ideal, ...
    include_csv: bool = False,
    csv_path: str | None = None,
    include_decoherence: bool = False,
    decoherence_level: float = 0.0,  # maximum shrink intensity
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Generates a simulated qubit tomography dataset,
    WITHOUT calculating the MLE columns.

    PHYSICAL PIPELINE:
        ideal state (θ,φ) -> pure vector (X_ideal,Y_ideal,Z_ideal)
        -> (optional noise channel) -> REAL state (X_real,Y_real,Z_real)
        -> (measurements at n_shots) -> features (X_mean,Y_mean,Z_mean)

    - FEATURES to use for ML: X_mean, Y_mean, Z_mean.
    - Continuous physical "labels" (for regression): X_real, Y_real, Z_real.
    - Ideal values (theta_ideal, phi_ideal, X_ideal, Y_ideal, Z_ideal)
      are used solely for analysis/visualization.

    This function DOES NOT calculate MLE columns: this will be done
    by a separate function to allow for post-hoc ML vs MLE comparison.

    Parameters
    ----------
    n_states : int
        Number of states (rows in the dataset).
    n_shots : int
        Number of measurements per observable (X, Y, Z).
    mode : {"finite_shots", "ideal"}
        - "finite_shots": simulates statistical noise (±1 draws).
        - "ideal"       : no statistical noise, X_mean = X_real, etc.
    include_ideal : bool
        If True, stores ideal values (theta_ideal, phi_ideal, X_ideal, ...).
    include_csv : bool
        If True, also saves the DataFrame as a CSV.
    csv_path : str | None
        Path to the CSV. If None and include_csv=True, a default name is used.
    include_decoherence : bool
        If True, applies a decoherence channel (anisotropic shrink).
    decoherence_level : float in [0,1]
        Maximum decoherence level. The higher it is, the more the Bloch vector is contracted.
    random_state : int | None
        Seed for the pseudo-random generator (reproducibility).

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing:
            - X_mean, Y_mean, Z_mean (features)
            - X_real, Y_real, Z_real (continuous labels, real state)
            - (optional) theta_ideal, phi_ideal, X_ideal, Y_ideal, Z_ideal
        -> No MLE columns here.

    !!! Saves n_shots in the DataFrame metadata, as it is used by the MLE !!!
    """

    rng = np.random.default_rng(random_state)

    if mode not in ("finite_shots", "ideal"):
        raise ValueError("mode must be 'finite_shots' or 'ideal'")

    if include_decoherence and not (0.0 <= decoherence_level <= 1.0):
        raise ValueError("decoherence_level must be in [0,1]")

    records = []

    # Internal function to simulate n_shots based on an expectation value <A>
    def simulate_mean(exp_val: float) -> float:
        """
        Simulates n_shots ±1 measurements for a Pauli observable,
        given that E[A] = exp_val.

        p(+) = (1 + exp_val) / 2
        p(-) = 1 - p(+)
        """
        p_plus = 0.5 * (1.0 + exp_val)

        # Numerical safety: clip to [0,1]
        p_plus = float(np.clip(p_plus, 0.0, 1.0))

        outcomes = rng.choice(
            [+1.0, -1.0],
            size=n_shots,
            p=[p_plus, 1.0 - p_plus],
        )
        return outcomes.mean()

    # Loop over states
    for _ in range(n_states):

      
        # 1) Uniform sampling on the Bloch sphere -> ideal state (pure)
      
        # Uniform sampling of u = cos(theta) in [-1,1]
        u = rng.uniform(-1.0, 1.0)
        theta_ideal = np.arccos(u)              # theta in [0, pi]
        phi_ideal = rng.uniform(0.0, 2.0 * np.pi)

        # Cartesian coordinates on the Bloch sphere
        X_ideal = np.sin(theta_ideal) * np.cos(phi_ideal)
        Y_ideal = np.sin(theta_ideal) * np.sin(phi_ideal)
        Z_ideal = np.cos(theta_ideal)

    
        # 2) Apply decoherence channel (optional)
        #    -> get REAL state (potentially mixed)
       
        if include_decoherence and decoherence_level > 0.0:

            # With a certain probability (proportion of noisy states)
            if rng.uniform() < decoherence_level:

                # Choose a random shrink "strength"
                strength = rng.uniform(0.0, decoherence_level)
                base_factor = 1.0 - strength

                # Random anisotropy axis by axis (to break symmetry)
                anisotropy = rng.uniform(0.5, 1.5, size=3)

                # Final factors on X, Y, Z, bounded in [0,1]
                factors = np.clip(base_factor * anisotropy, 0.0, 1.0)

                X_real = factors[0] * X_ideal
                Y_real = factors[1] * Y_ideal
                Z_real = factors[2] * Z_ideal

            else:
                # No decoherence for this draw
                X_real, Y_real, Z_real = X_ideal, Y_ideal, Z_ideal

        else:
            # No decoherence at all: real state = ideal state (pure)
            X_real, Y_real, Z_real = X_ideal, Y_ideal, Z_ideal

     
        # 3) Measurement simulation (statistical noise)

        if mode == "ideal":
            # No statistical noise: features are exactly the real state
            X_mean, Y_mean, Z_mean = X_real, Y_real, Z_real

        else:
            # "finite_shots": simulate n_shots draws for X, Y, Z
            X_mean = simulate_mean(X_real)
            Y_mean = simulate_mean(Y_real)
            Z_mean = simulate_mean(Z_real)

    
        # 4) Dictionary construction (one DataFrame row)
    
        record = {
            # FEATURES (always present)
            "X_mean": X_mean,
            "Y_mean": Y_mean,
            "Z_mean": Z_mean,

            # Continuous physical LABELS (real state)
            "X_real": X_real,
            "Y_real": Y_real,
            "Z_real": Z_real,
            # Add n_shots to each row for safety,
            # or could rely on attributes, but row-wise is safer for concat
            "n_shots_sim": n_shots
        }

        # 5) Option: also store ideal values
        if include_ideal:
            record.update({
                "theta_ideal": theta_ideal,
                "phi_ideal": phi_ideal,
                "X_ideal": X_ideal,
                "Y_ideal": Y_ideal,
                "Z_ideal": Z_ideal,
            })

        records.append(record)


    # 5) Creation of the final DataFrame

    df = pd.DataFrame.from_records(records)

    # Extra safety: attach n_shots to dataframe attributes
    df.attrs['n_shots'] = n_shots
    # Option: save as CSV
    if include_csv:
        if csv_path is None:
            csv_path = f"tomography_dataset_{n_states}_shots{n_shots}.csv"
        df.to_csv(csv_path, index=False)

    return df


def perform_mle_tomography(df_input: pd.DataFrame, n_shots: int = None) -> tuple[pd.DataFrame, float]:
    """
    Applies MLE to reconstruct the density matrix rho.
    Supports mixed states (r <= 1).
    """
    
    # 1. Automatic handling of n_shots
    if n_shots is None:
        if 'n_shots' in df_input.attrs:
            n_shots = df_input.attrs['n_shots']
        elif 'n_shots_sim' in df_input.columns:
            n_shots = int(df_input['n_shots_sim'].iloc[0])
        else:
            raise ValueError("n_shots not found.")

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Measurement projectors (+1) for X, Y, Z
    # P_i+ = (I + sigma_i) / 2
    I = np.eye(2, dtype=complex)
    Proj_X = (I + sigma_x) / 2.0
    Proj_Y = (I + sigma_y) / 2.0
    Proj_Z = (I + sigma_z) / 2.0

    #  Cost function (NLL) 
    def nll_rho(t_params, n_x, n_y, n_z, N):
        # 1. Reconstruct T from 4 real parameters (Cholesky)
        # T = [[t0, 0], [t1 + i*t2, t3]]
        t0, t1, t2, t3 = t_params
        T = np.array([[t0, 0], [t1 + 1j*t2, t3]], dtype=complex)
        
        # 2. rho = T† T / Tr(T† T)
        rho_un = T.conj().T @ T
        norm = np.trace(rho_un).real
        rho = rho_un / norm
        
        # 3. Theoretical probabilities p = Tr(rho * Proj)
        px = np.trace(rho @ Proj_X).real
        py = np.trace(rho @ Proj_Y).real
        pz = np.trace(rho @ Proj_Z).real
        
        # 4. Safety clipping
        eps = 1e-9
        px = np.clip(px, eps, 1-eps)
        py = np.clip(py, eps, 1-eps)
        pz = np.clip(pz, eps, 1-eps)
        
        # 5. Log-likelihood
        ll = (n_x * np.log(px) + (N - n_x) * np.log(1 - px) +
              n_y * np.log(py) + (N - n_y) * np.log(1 - py) +
              n_z * np.log(pz) + (N - n_z) * np.log(1 - pz))
        
        return -ll

    #  Execution 
    df_result = df_input.copy()
    
    # Recalculate counts
    if "numberX" not in df_result.columns:
        df_result["numberX"] = ((1 + df_result["X_mean"]) / 2.0 * n_shots).round().astype(int)
        df_result["numberY"] = ((1 + df_result["Y_mean"]) / 2.0 * n_shots).round().astype(int)
        df_result["numberZ"] = ((1 + df_result["Z_mean"]) / 2.0 * n_shots).round().astype(int)

    # Lists to store results
    x_mle, y_mle, z_mle = [], [], []
    
    start_time = time.time()

    for _, row in df_result.iterrows():
        # Naive initialization: T = Identity (rho = I/2 = maximally mixed state)
        # This is a safe starting point for the optimizer.
        t_init = [1.0, 0.0, 0.0, 1.0]
        
        # Optimization
        res = minimize(
            nll_rho,
            x0=t_init,
            args=(row["numberX"], row["numberY"], row["numberZ"], n_shots),
            method="Nelder-Mead" # Nelder-Mead is often more robust for this non-convex landscape
        )
        
        # Reconstruction of the final rho matrix
        t0, t1, t2, t3 = res.x
        T = np.array([[t0, 0], [t1 + 1j*t2, t3]], dtype=complex)
        rho_final = (T.conj().T @ T) / np.trace(T.conj().T @ T)
        
        # Extraction of Bloch coordinates (x, y, z) from the density matrix
        # r_i = Tr(rho * sigma_i)
        rx_val = np.trace(rho_final @ sigma_x).real
        ry_val = np.trace(rho_final @ sigma_y).real
        rz_val = np.trace(rho_final @ sigma_z).real
        
        x_mle.append(rx_val)
        y_mle.append(ry_val)
        z_mle.append(rz_val)

    end_time = time.time()
    
    # Save results
    df_result["X_mle"] = x_mle
    df_result["Y_mle"] = y_mle
    df_result["Z_mle"] = z_mle
    
    # Calculate MLE angles a posteriori (optional, for compatibility)
    # r = sqrt(x^2 + y^2 + z^2), theta, phi
    r_mle = np.sqrt(np.array(x_mle)**2 + np.array(y_mle)**2 + np.array(z_mle)**2)
    # Clip to avoid rounding errors on arccos
    z_mle_norm = np.array(z_mle) / np.clip(r_mle, 1e-9, None) 
    
    df_result["theta_mle"] = np.arccos(np.clip(z_mle_norm, -1.0, 1.0))
    df_result["phi_mle"] = np.arctan2(y_mle, x_mle)
    df_result["phi_mle"] = df_result["phi_mle"].apply(lambda p: p + 2*np.pi if p < 0 else p)
    df_result["r_mle"] = r_mle

    return df_result, (end_time - start_time)

def build_purity_classification_dataset(
    n_states_total: int,
    mixed_proportion: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Constructs a dataset for binary classification ("pure state vs. mixed state").

    This function encapsulates all physics/simulation aspects with FIXED choices
    and only allows variation of parameters relevant to ML:
        - the total dataset size (n_states_total),
        - the number of shots per measurement (n_shots),
        - the target proportion of mixed states (mixed_proportion).

    IMPOSED PHYSICAL CHOICES (not configurable here):
        - n_shots             : 1000  (number of measurements per observable),
        - measurement mode    : "ideal" (no statistical noise),
        - include_ideal       : True  (keep ideal values for analysis),
        - include_decoherence : True  (decoherence enabled),
        - decoherence_level   : 0.6   (typical noise channel strength),
        - eps_purity          : 1e-2  (tolerance around radius 1),
        - label_col           : "label_purity" (1 = pure, 0 = mixed).

    PIPELINE:
        1) Generates a base dataset using `generate_qubit_tomography_dataset_base`,
           applying a decoherence channel (decoherence_level = 0.6).
        2) Calculates the real Bloch radius:
               r_real = sqrt(X_real^2 + Y_real^2 + Z_real^2).
        3) Defines a "pure vs. mixed" label based on this radius:
               pure state  : r_real >= 1 - eps_purity
               mixed state : r_real <  1 - eps_purity
        4) Constructs a classification dataset while respecting the target
           mixed_proportion (0 < mixed_proportion < 1) AS BEST AS POSSIBLE.
           The effective size may be slightly smaller than n_states_total
           if the simulation does not produce enough pure or mixed states
           to exactly match the proportion.

    Parameters
    ----------
    n_states_total : int
        Total number of states to generate BEFORE class balancing.
        The final dataset size will be less than or equal to this value.
    mixed_proportion : float in (0,1)
        TARGET proportion of mixed states in the final dataset.
        Example:
            - 0.5  -> 50% mixed / 50% pure (balanced dataset),
            - 0.7  -> 70% mixed / 30% pure.

    Returns
    -------
    df_clf : pd.DataFrame
        Complete DataFrame containing:
            - X_real, Y_real, Z_real   : real state components (features),
            - bloch_radius_real        : norm of the real Bloch vector (feature),
            - is_pure                  : boolean (True = pure, False = mixed) (feature),
            - theta_ideal, phi_ideal, X_ideal, Y_ideal, Z_ideal : ideal values (analysis),
            - label_purity             : int (1 = pure, 0 = mixed) (target label).
    X : pd.DataFrame
        Subset DataFrame of features for classification:
            columns ["X_real", "Y_real", "Z_real", "bloch_radius_real", "is_pure"].
    y : pd.Series
        Series of labels (0/1) corresponding to "label_purity".

    Raises
    ------
    ValueError
        - if mixed_proportion is not in the interval (0,1),
        - if the simulation does not produce AT LEAST one pure state and one mixed state.
    """

    # 0) Input parameter verification

    if not (0.0 < mixed_proportion < 1.0):
        raise ValueError("mixed_proportion must be strictly between 0 and 1.")

    # FIXED internal parameters (physical choices)
    N_SHOTS = 1000
    MODE = "ideal"
    INCLUDE_IDEAL = True
    INCLUDE_DECOHERENCE = True
    DECOHERENCE_LEVEL = 0.6
    EPS_PURITY = 1e-2
    LABEL_COL = "label_purity"
    FEATURE_COLS = ["X_real", "Y_real", "Z_real", "bloch_radius_real", "is_pure"]

   
    # 1) Generation of a base dataset with decoherence
  
    df_base = generate_qubit_tomography_dataset_base(
        n_states=n_states_total,
        n_shots=N_SHOTS,
        mode=MODE,
        include_ideal=INCLUDE_IDEAL,
        include_csv=False,
        csv_path=None,
        include_decoherence=INCLUDE_DECOHERENCE,
        decoherence_level=DECOHERENCE_LEVEL,
        random_state=None,  # random
    )


    # 2) Calculation of the real Bloch radius and definition of the pure/mixed label

    r_real = np.sqrt(
        df_base["X_real"]**2
        + df_base["Y_real"]**2
        + df_base["Z_real"]**2
    )
    df_base["bloch_radius_real"] = r_real

    # A state is considered pure if it is sufficiently close to the sphere surface
    df_base["is_pure"] = df_base["bloch_radius_real"] >= (1.0 - EPS_PURITY)

    # Numerical label:
    #   1 = pure state
    #   0 = mixed state
    df_base[LABEL_COL] = df_base["is_pure"].astype(int)


    # 3) Construction of a dataset with the target mixed_proportion

    df_pure  = df_base[df_base[LABEL_COL] == 1]
    df_mixte = df_base[df_base[LABEL_COL] == 0]

    n_pure_available  = len(df_pure)
    n_mixed_available = len(df_mixte)

    if n_pure_available == 0 or n_mixed_available == 0:
        raise ValueError(
            "The simulation did not produce enough states of both types "
            f"(pure: {n_pure_available}, mixed: {n_mixed_available}). "
            "Increase n_states_total or adjust internal parameters."
        )

    # TARGET number of mixed and pure states, before adjustment
    target_mixed = mixed_proportion * n_states_total
    target_pure  = (1.0 - mixed_proportion) * n_states_total

    # Calculate a scaling factor to ensure we don't request more
    # states than the simulation produced.
    scale = min(
        n_mixed_available / target_mixed,
        n_pure_available  / target_pure,
        1.0
    )

    # EFFECTIVE numbers to sample from each class
    n_mixed = int(target_mixed * scale)
    n_pure  = int(target_pure  * scale)

    # Minimal safety check
    if n_mixed == 0 or n_pure == 0:
        raise ValueError(
            "Impossible to construct a dataset respecting the desired proportion "
            "with at least one example of each class. "
            f"(after adjustment: n_mixed={n_mixed}, n_pure={n_pure})"
        )

    # Random sampling from each pool
    df_mixte_sample = df_mixte.sample(n=n_mixed, random_state=None)
    df_pure_sample  = df_pure.sample(n=n_pure,  random_state=None)

    # Concatenation
    df_clf = pd.concat([df_pure_sample, df_mixte_sample], ignore_index=True)

    # Final shuffle to break any structural order
    df_clf = df_clf.sample(frac=1.0, random_state=None).reset_index(drop=True)


    # 4) Separation of features / labels for ML

    X = df_clf[FEATURE_COLS].copy()
    y = df_clf[LABEL_COL].copy()

    return df_clf, X, y
