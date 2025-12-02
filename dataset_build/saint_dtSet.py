import numpy as np
import pandas as pd


def generate_qubit_tomography_dataset(
    n_states: int,
    n_shots: int,
    mode: str = "finite_shots",      # "finite_shots" ou "ideal"
    include_ideal: bool = True,      # stocker θ, φ et <X,Y,Z> idéaux
    include_mle: bool = True,        # calculer θ_mle, φ_mle à partir de X_mean,Y_mean,Z_mean
    include_csv: bool = False,       # éventuellement sauver le DataFrame en CSV
    csv_path: str | None = None,
    include_decoherence: bool = False,
    decoherence_level: float = 0.0,  # probabilité de dépolarisant p \in [0,1]
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Génère un dataset de tomographie quantique pour un qubit.

    Pour chaque état k = 1..n_states :
      - Tirage uniforme sur la sphère de Bloch :
            u = cos(theta) ~ U[-1,1]
            theta = arccos(u)
            phi   ~ U[0, 2π]
      - Etat pur : |ψ> = cos(theta/2)|0> + e^{iφ} sin(theta/2)|1>
      - Valeurs moyennes idéales (Bloch) :
            <X> = sin(theta) cos(phi)
            <Y> = sin(theta) sin(phi)
            <Z> = cos(theta)
      - Optionnel : canal dépolarisant de paramètre p = decoherence_level
            r -> (1-p) r
      - Mode "ideal" :
            X_mean, Y_mean, Z_mean = <X>, <Y>, <Z> (après décohérence éventuelle)
        Mode "finite_shots" :
            tirage de n_shots valeurs ±1 pour chaque observable,
            X_mean, Y_mean, Z_mean = moyennes empiriques.

      - MLE (si include_mle=True) :
            r = (X_mean, Y_mean, Z_mean)
            si ||r|| > 1 : r <- r / ||r||        # projection sur la sphère
            theta_mle = arccos(Z_mle)
            phi_mle   = atan2(Y_mle, X_mle)

    Colonnes du DataFrame retourné :
      - toujours :  X_mean, Y_mean, Z_mean
      - si include_ideal=True :
            theta_true, phi_true, X_true, Y_true, Z_true
      - si include_mle=True :
            X_mle, Y_mle, Z_mle, theta_mle, phi_mle
    """

    rng = np.random.default_rng(random_state)

    if mode not in ("finite_shots", "ideal"):
        raise ValueError("mode doit être 'finite_shots' ou 'ideal'")

    if include_decoherence and not (0.0 <= decoherence_level <= 1.0):
        raise ValueError("decoherence_level doit être dans [0,1]")

    records = []

    for _ in range(n_states):
        # 1) Tirage uniforme sur la sphère de Bloch
        u = rng.uniform(-1.0, 1.0)      # u = cos(theta)
        theta = np.arccos(u)
        phi = rng.uniform(0.0, 2.0 * np.pi)

        # 2) Valeurs moyennes idéales <X>, <Y>, <Z> pour l'état pur
        #    (formules analytiques pour un qubit sur la sphère de Bloch)
        X_true = np.sin(theta) * np.cos(phi)
        Y_true = np.sin(theta) * np.sin(phi)
        Z_true = np.cos(theta)

        # 3) Eventuelle décohérence (canal dépolarisant simple)
        #    ρ -> (1-p) ρ + p I/2  <=>  r -> (1-p) r
        if include_decoherence and decoherence_level > 0.0:
            factor = 1.0 - decoherence_level
            X_eff = factor * X_true
            Y_eff = factor * Y_true
            Z_eff = factor * Z_true
        else:
            X_eff, Y_eff, Z_eff = X_true, Y_true, Z_true

        # 4) Simulation des mesures
        if mode == "ideal":
            # Pas de bruit statistique, uniquement décohérence éventuelle
            X_mean, Y_mean, Z_mean = X_eff, Y_eff, Z_eff

        else:  # mode == "finite_shots"
            def simulate_mean(exp_val: float) -> float:
                # Probabilité P(+1) = (1 + <O>)/2
                p_plus = 0.5 * (1.0 + exp_val)
                # Sécurité numérique
                p_plus = float(np.clip(p_plus, 0.0, 1.0))
                outcomes = rng.choice([+1.0, -1.0], size=n_shots,
                                      p=[p_plus, 1.0 - p_plus])
                return outcomes.mean()

            X_mean = simulate_mean(X_eff)
            Y_mean = simulate_mean(Y_eff)
            Z_mean = simulate_mean(Z_eff)

        # 5) Enregistrement de la ligne
        record = {
            "X_mean": X_mean,
            "Y_mean": Y_mean,
            "Z_mean": Z_mean,
        }

        if include_ideal:
            record.update({
                "theta_true": theta,
                "phi_true": phi,
                "X_true": X_true,
                "Y_true": Y_true,
                "Z_true": Z_true,
            })

        if include_mle:
            r_vec = np.array([X_mean, Y_mean, Z_mean], dtype=float)
            r_norm = np.linalg.norm(r_vec)

            if r_norm > 1.0:
                r_vec = r_vec / r_norm

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

    # 6) Sauvegarde éventuelle en CSV (optionnelle)
    if include_csv:
        if csv_path is None:
            csv_path = f"tomography_dataset_n{n_states}_shots{n_shots}.csv"
        df.to_csv(csv_path, index=False)

    return df
