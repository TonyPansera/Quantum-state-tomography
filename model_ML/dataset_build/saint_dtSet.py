import numpy as np
import pandas as pd

def generate_qubit_tomography_dataset(
    n_states: int,
    n_shots: int,
    mode: str = "finite_shots",      # "finite_shots" : tirages binomiaux ; "ideal" : pas de bruit statistique
    include_ideal: bool = True,      # stocker theta_ideal, phi_ideal, X_ideal, ...
    include_mle: bool = True,        # calculer un estimateur MLE optionnel
    include_csv: bool = False,
    csv_path: str | None = None,
    include_decoherence: bool = False,
    decoherence_level: float = 0.0,  # intensité maximale du shrink
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Génère un dataset simulé de tomographie quantique d'un qubit.

    PIPELINE PHYSIQUE :
        état idéal (θ,φ) -> vecteur pur (X_ideal,Y_ideal,Z_ideal)
        -> (canal de bruit optionnel) -> état RÉEL (X_real,Y_real,Z_real)
        -> (mesures à n_shots) -> features (X_mean,Y_mean,Z_mean)
        -> (MLE optionnel)

    ARGUMENTS :
        mode = "ideal" :
            - pas de bruit statistique
            - X_mean = X_real (aucun tirage binomial)
        mode = "finite_shots" :
            - tirages ±1 avec probabilité correcte
            - bruit statistique réel

    >>> IMPORTANT POUR LE MACHINE LEARNING <<<
        - Les FEATURES à utiliser = X_mean, Y_mean, Z_mean.
        - Les LABELS physiquement corrects = X_real, Y_real, Z_real.
        - X_ideal,Y_ideal... servent uniquement à l'analyse (pas au ML).
        - Le MLE N'EST PAS un label et ne doit jamais être utilisé comme tel.

    SORTIE :
        DataFrame contenant :
            - Toujours : X_mean, Y_mean, Z_mean (FEATURES)
            - Toujours : X_real, Y_real, Z_real (LABELS physiques, pour entrainer et à prédire)
            - Optionnel : valeurs IDEALES (theta_ideal, ...) (si include_ideal, sert pour des graphes)
            - Optionnel : valeurs MLE (theta_mle, ...) (si include_mle, mais notre but est de comparer ML vs MLE, donc toujours normalement)
    """

    rng = np.random.default_rng(random_state)

    if mode not in ("finite_shots", "ideal"):
        raise ValueError("mode doit être 'finite_shots' ou 'ideal'")

    if include_decoherence and not (0.0 <= decoherence_level <= 1.0):
        raise ValueError("decoherence_level doit être dans [0,1]")

    records = []

    # Fonction interne pour simuler les n_shots
    def simulate_mean(exp_val: float) -> float:
        p_plus = 0.5 * (1.0 + exp_val)
        p_plus = float(np.clip(p_plus, 0.0, 1.0))
        outcomes = rng.choice([+1.0, -1.0], size=n_shots,
                              p=[p_plus, 1.0 - p_plus])
        return outcomes.mean()

    # Boucle sur les états
    for _ in range(n_states):

        # 1) Tirage uniforme sur la sphère (état idéal)
        u = rng.uniform(-1.0, 1.0)               # u = cos(theta)
        theta_ideal = np.arccos(u)
        phi_ideal = rng.uniform(0.0, 2.0*np.pi)

        X_ideal = np.sin(theta_ideal) * np.cos(phi_ideal)
        Y_ideal = np.sin(theta_ideal) * np.sin(phi_ideal)
        Z_ideal = np.cos(theta_ideal)

        # 2) Application du canal de bruit :
        #    - shrink anisotrope aléatoire sur (X,Y,Z)
        #    - garantit que l'état reste physique (||r||<=1)
        if include_decoherence and decoherence_level > 0.0:

            # probabilité qu'un état soit bruité
            if rng.uniform() < decoherence_level:

                # shrink moyen
                strength = rng.uniform(0.0, decoherence_level)
                base_factor = 1.0 - strength

                # anisotropie aléatoire axe par axe
                anisotropy = rng.uniform(0.5, 1.5, size=3)
                factors = np.clip(base_factor * anisotropy, 0.0, 1.0)

                X_real = factors[0] * X_ideal
                Y_real = factors[1] * Y_ideal
                Z_real = factors[2] * Z_ideal

            else:
                X_real, Y_real, Z_real = X_ideal, Y_ideal, Z_ideal

        else:
            X_real, Y_real, Z_real = X_ideal, Y_ideal, Z_ideal

        # 3) Simulation des mesures
        if mode == "ideal":
            # Pas de fluctuation : features = état réel
            X_mean, Y_mean, Z_mean = X_real, Y_real, Z_real

        else:
            # bruit statistique (tirages binomiaux)
            X_mean = simulate_mean(X_real)
            Y_mean = simulate_mean(Y_real)
            Z_mean = simulate_mean(Z_real)

        # 4) Construction du record
        record = {
            "X_mean": X_mean,     # FEATURES
            "Y_mean": Y_mean,
            "Z_mean": Z_mean,

            "X_real": X_real,     # LABELS (état réel)
            "Y_real": Y_real,
            "Z_real": Z_real,
        }

        # 5) Ajouter les valeurs idéales
        if include_ideal:
            record.update({
                "theta_ideal": theta_ideal,
                "phi_ideal": phi_ideal,
                "X_ideal": X_ideal,
                "Y_ideal": Y_ideal,
                "Z_ideal": Z_ideal,
            })

        # 6) MLE optionnel
        if include_mle:
            r_vec = np.array([X_mean, Y_mean, Z_mean], dtype=float)

            # Cas avec NaN (si tu actives une future perte de mesure)
            if np.isnan(r_vec).any():
                record.update({
                    "X_mle": np.nan,
                    "Y_mle": np.nan,
                    "Z_mle": np.nan,
                    "theta_mle": np.nan,
                    "phi_mle": np.nan,
                })
            else:
                # Projection si r_mean dépasse la boule (bruit statistique)
                r_norm = np.linalg.norm(r_vec)
                if r_norm > 1.0:
                    r_vec /= r_norm

                X_mle, Y_mle, Z_mle = r_vec
                theta_mle = np.arccos(Z_mle)
                phi_mle = np.arctan2(Y_mle, X_mle)

                record.update({
                    "X_mle": X_mle,
                    "Y_mle": Y_mle,
                    "Z_mle": Z_mle,
                    "theta_mle": theta_mle,
                    "phi_mle": phi_mle,
                })

        records.append(record)

    df = pd.DataFrame.from_records(records)

    if include_csv:
        if csv_path is None:
            csv_path = f"tomography_dataset_{n_states}_shots{n_shots}.csv"
        df.to_csv(csv_path, index=False)

    return df
