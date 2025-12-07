import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1) Génération du dataset de tomographie (SANS MLE)
# ---------------------------------------------------------------------------

def generate_qubit_tomography_dataset_base(
    n_states: int,
    n_shots: int,
    mode: str = "finite_shots",      # "finite_shots" : tirages binomiaux ; "ideal" : pas de bruit statistique
    include_ideal: bool = True,      # stocker theta_ideal, phi_ideal, X_ideal, ...
    include_csv: bool = False,
    csv_path: str | None = None,
    include_decoherence: bool = False,
    decoherence_level: float = 0.0,  # intensité maximale du shrink
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Génère un dataset simulé de tomographie quantique d'un qubit,
    SANS calculer les colonnes MLE.

    PIPELINE PHYSIQUE :
        état idéal (θ,φ) -> vecteur pur (X_ideal,Y_ideal,Z_ideal)
        -> (canal de bruit optionnel) -> état RÉEL (X_real,Y_real,Z_real)
        -> (mesures à n_shots) -> features (X_mean,Y_mean,Z_mean)

    - Les FEATURES à utiliser pour le ML : X_mean, Y_mean, Z_mean.
    - Les "labels" physiques continus (pour la régression) : X_real, Y_real, Z_real.
    - Les valeurs idéales (theta_ideal, phi_ideal, X_ideal, Y_ideal, Z_ideal)
      servent uniquement à l'analyse / visualisation.

    Cette fonction NE calcule PAS les colonnes MLE : cela sera fait
    par une fonction séparée, pour pouvoir comparer ML vs MLE a posteriori.

    Paramètres
    ----------
    n_states : int
        Nombre d'états (lignes du dataset).
    n_shots : int
        Nombre de mesures par observable (X, Y, Z).
    mode : {"finite_shots", "ideal"}
        - "finite_shots" : on simule le bruit statistique (tirages ±1).
        - "ideal"        : pas de bruit statistique, X_mean = X_real, etc.
    include_ideal : bool
        Si True, on stocke les valeurs idéales (theta_ideal, phi_ideal, X_ideal, ...).
    include_csv : bool
        Si True, on sauvegarde aussi le DataFrame en CSV.
    csv_path : str | None
        Chemin du CSV. Si None et include_csv=True, un nom par défaut est utilisé.
    include_decoherence : bool
        Si True, on applique un canal de décohérence (shrink anisotrope).
    decoherence_level : float in [0,1]
        Niveau maximal de décohérence. Plus c'est grand, plus on contracte le vecteur de Bloch.
    random_state : int | None
        Graine du générateur pseudo-aléatoire (reproducibilité).

    Retour
    ------
    df : pd.DataFrame
        DataFrame contenant :
            - X_mean, Y_mean, Z_mean (features)
            - X_real, Y_real, Z_real (labels continus, état réel)
            - (optionnel) theta_ideal, phi_ideal, X_ideal, Y_ideal, Z_ideal
        -> Pas de colonnes MLE ici.
    """

    rng = np.random.default_rng(random_state)

    if mode not in ("finite_shots", "ideal"):
        raise ValueError("mode doit être 'finite_shots' ou 'ideal'")

    if include_decoherence and not (0.0 <= decoherence_level <= 1.0):
        raise ValueError("decoherence_level doit être dans [0,1]")

    records = []

    # Fonction interne pour simuler les n_shots à partir d'une valeur d'espérance <A>
    def simulate_mean(exp_val: float) -> float:
        """
        Simule n_shots mesures ±1 pour un observable de Pauli,
        sachant que E[A] = exp_val.

        p(+) = (1 + exp_val) / 2
        p(-) = 1 - p(+)
        """
        p_plus = 0.5 * (1.0 + exp_val)

        # Sécurité numérique : on clippe dans [0,1]
        p_plus = float(np.clip(p_plus, 0.0, 1.0))

        outcomes = rng.choice(
            [+1.0, -1.0],
            size=n_shots,
            p=[p_plus, 1.0 - p_plus],
        )
        return outcomes.mean()

    # Boucle sur les états
    for _ in range(n_states):

        # ------------------------------------------------------------------
        # 1) Tirage uniforme sur la sphère de Bloch -> état idéal (pur)
        # ------------------------------------------------------------------
        # Tirage uniforme de u = cos(theta) dans [-1,1]
        u = rng.uniform(-1.0, 1.0)
        theta_ideal = np.arccos(u)              # theta dans [0, pi]
        phi_ideal = rng.uniform(0.0, 2.0 * np.pi)

        # Coordonnées cartésiennes sur la sphère de Bloch
        X_ideal = np.sin(theta_ideal) * np.cos(phi_ideal)
        Y_ideal = np.sin(theta_ideal) * np.sin(phi_ideal)
        Z_ideal = np.cos(theta_ideal)

        # ------------------------------------------------------------------
        # 2) Application du canal de décohérence (optionnel)
        #    -> on obtient l'état RÉEL (potentiellement mixte)
        # ------------------------------------------------------------------
        if include_decoherence and decoherence_level > 0.0:

            # Avec une certaine probabilité (proportion d'états bruités)
            if rng.uniform() < decoherence_level:

                # On choisit une "force" de shrink aléatoire
                strength = rng.uniform(0.0, decoherence_level)
                base_factor = 1.0 - strength

                # Anisotropie aléatoire axe par axe (pour briser la symétrie)
                anisotropy = rng.uniform(0.5, 1.5, size=3)

                # Facteurs finaux sur X, Y, Z, bornés dans [0,1]
                factors = np.clip(base_factor * anisotropy, 0.0, 1.0)

                X_real = factors[0] * X_ideal
                Y_real = factors[1] * Y_ideal
                Z_real = factors[2] * Z_ideal

            else:
                # Pas de décohérence pour ce tirage
                X_real, Y_real, Z_real = X_ideal, Y_ideal, Z_ideal

        else:
            # Pas de décohérence du tout : état réel = état idéal (pur)
            X_real, Y_real, Z_real = X_ideal, Y_ideal, Z_ideal

        # ------------------------------------------------------------------
        # 3) Simulation des mesures (bruit statistique)
        # ------------------------------------------------------------------
        if mode == "ideal":
            # Pas de bruit statistique : les features sont exactement l'état réel
            X_mean, Y_mean, Z_mean = X_real, Y_real, Z_real

        else:
            # "finite_shots" : on simule n_shots tirages pour X, Y, Z
            X_mean = simulate_mean(X_real)
            Y_mean = simulate_mean(Y_real)
            Z_mean = simulate_mean(Z_real)

        # ------------------------------------------------------------------
        # 4) Construction du dictionnaire (une ligne du DataFrame)
        # ------------------------------------------------------------------
        record = {
            # FEATURES (toujours présents)
            "X_mean": X_mean,
            "Y_mean": Y_mean,
            "Z_mean": Z_mean,

            # LABELS continus physiques (état réel)
            "X_real": X_real,
            "Y_real": Y_real,
            "Z_real": Z_real,
        }

        # 5) Option : on stocke aussi les valeurs idéales
        if include_ideal:
            record.update({
                "theta_ideal": theta_ideal,
                "phi_ideal": phi_ideal,
                "X_ideal": X_ideal,
                "Y_ideal": Y_ideal,
                "Z_ideal": Z_ideal,
            })

        records.append(record)

    # ----------------------------------------------------------------------
    # 5) Création du DataFrame final
    # ----------------------------------------------------------------------
    df = pd.DataFrame.from_records(records)

    # Option : sauvegarde en CSV
    if include_csv:
        if csv_path is None:
            csv_path = f"tomography_dataset_{n_states}_shots{n_shots}.csv"
        df.to_csv(csv_path, index=False)

    return df


# ---------------------------------------------------------------------------
# 2) Fonction séparée : calcul des colonnes MLE à partir de X_mean, Y_mean, Z_mean
# ---------------------------------------------------------------------------

def add_mle_from_means(
    df: pd.DataFrame,
    projection_on_bloch_ball: bool = True,
) -> pd.DataFrame:
    """
    Calcule et ajoute les colonnes MLE (X_mle, Y_mle, Z_mle, theta_mle, phi_mle)
    à un DataFrame issu de `generate_qubit_tomography_dataset_base`.

    Hypothèse :
        - df contient les colonnes "X_mean", "Y_mean", "Z_mean".
        - Ces colonnes sont les moyennes expérimentales des observables de Pauli.

    Idée :
        - Pour un qubit, l'estimateur MLE simple consiste à prendre le vecteur
          des moyennes (X_mean, Y_mean, Z_mean), puis éventuellement à le
          projeter dans la boule de Bloch (norme <= 1).
        - On convertit ensuite ce vecteur (X_mle, Y_mle, Z_mle) en
          coordonnées sphériques (theta_mle, phi_mle).

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entrée. Il n'est pas modifié sur place (on travaille sur une copie).
    projection_on_bloch_ball : bool
        Si True, on projette les vecteurs qui dépassent la norme 1
        (résultat du bruit statistique) sur la sphère de rayon 1.

    Retour
    ------
    df_out : pd.DataFrame
        Copie de df, avec colonnes supplémentaires :
            - X_mle, Y_mle, Z_mle
            - theta_mle, phi_mle

        Pour les lignes où (X_mean, Y_mean, Z_mean) contiennent un NaN,
        toutes les colonnes MLE sont mises à NaN.
    """

    # On travaille sur une copie pour éviter de modifier df in-place
    df_out = df.copy()

    # Récupération des vecteurs de moyennes sous forme de tableau NumPy
    r_means = df_out[["X_mean", "Y_mean", "Z_mean"]].to_numpy(dtype=float)

    # Masque des lignes contenant au moins un NaN dans X_mean, Y_mean ou Z_mean
    mask_nan = np.isnan(r_means).any(axis=1)

    # Pour éviter des problèmes dans le calcul des normes, on remplace
    # temporairement les NaN par 0. Ils seront remis à NaN plus tard.
    r_means_safe = r_means.copy()
    r_means_safe[mask_nan] = 0.0

    # Norme de chaque vecteur
    norms = np.linalg.norm(r_means_safe, axis=1)

    # Option : projection dans la boule de Bloch si la norme dépasse 1
    if projection_on_bloch_ball:
        mask_outside = norms > 1.0
        # On normalise uniquement les vecteurs qui dépassent la sphère
        r_means_safe[mask_outside] = (
            r_means_safe[mask_outside] / norms[mask_outside, None]
        )
        # Après projection, la norme est 1 pour ces vecteurs
        norms[mask_outside] = 1.0

    # Les composantes MLE sont simplement les composantes de r_means_safe
    X_mle = r_means_safe[:, 0]
    Y_mle = r_means_safe[:, 1]
    Z_mle = r_means_safe[:, 2]

    # Conversion en coordonnées sphériques :
    # theta = arccos(z), phi = atan2(y, x)
    # On clippe Z_mle dans [-1,1] pour éviter les erreurs numériques.
    Z_clipped = np.clip(Z_mle, -1.0, 1.0)
    theta_mle = np.arccos(Z_clipped)
    phi_mle = np.arctan2(Y_mle, X_mle)

    # On enregistre d'abord les valeurs calculées
    df_out["X_mle"] = X_mle
    df_out["Y_mle"] = Y_mle
    df_out["Z_mle"] = Z_mle
    df_out["theta_mle"] = theta_mle
    df_out["phi_mle"] = phi_mle

    # Puis on remet à NaN les lignes qui avaient des NaN en entrée
    df_out.loc[mask_nan, ["X_mle", "Y_mle", "Z_mle", "theta_mle", "phi_mle"]] = np.nan

    return df_out

import numpy as np
import pandas as pd


def build_purity_classification_dataset(
    n_states_total: int,
    mixed_proportion: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Construit un dataset pour la classification binaire "état pur vs état mixte".

    Cette fonction encapsule toute la physique / simulation avec des choix FIXES
    et ne laisse varier que ce qui est intéressant pour le ML :
        - la taille totale du dataset (n_states_total),
        - le nombre de shots par mesure (n_shots),
        - la proportion cible d'états mixtes (mixed_proportion).

    CHOIX PHYSIQUES IMPOSES (non paramétrables ici) :
        - n_shots                 : 1000  (nombre de mesures par observable),
        - mode de mesure          : "ideal" (pas de bruit statistique),
        - include_ideal           : True  (on garde les valeurs idéales pour analyse),
        - include_decoherence     : True  (on active la décohérence),
        - decoherence_level       : 0.6   (force typique du canal de bruit),
        - eps_purity              : 1e-2  (tolérance autour du rayon 1),
        - label_col               : "label_purity" (1 = pur, 0 = mixte).

    PIPELINE :
        1) Génère un dataset de base avec `generate_qubit_tomography_dataset_base`,
           en appliquant un canal de décohérence (decoherence_level = 0.6).
        2) Calcule le rayon de Bloch réel :
               r_real = sqrt(X_real^2 + Y_real^2 + Z_real^2).
        3) Définit un label "pur vs mixte" à partir de ce rayon :
               état pur   : r_real >= 1 - eps_purity
               état mixte : r_real <  1 - eps_purity
        4) Construit un dataset de classification en respectant AU MIEUX
           la proportion cible mixed_proportion (0 < mixed_proportion < 1).
           La taille effective peut être légèrement inférieure à n_states_total
           si la simulation ne produit pas assez d'états purs ou mixtes pour
           respecter exactement la proportion.

    Paramètres
    ----------
    n_states_total : int
        Nombre total d'états à générer AVANT équilibrage des classes.
        La taille finale du dataset sera inférieure ou égale à cette valeur.
    mixed_proportion : float in (0,1)
        Proportion CIBLE d'états mixtes dans le dataset final.
        Exemple :
            - 0.5  -> 50% mixtes / 50% purs (dataset équilibré),
            - 0.7  -> 70% mixtes / 30% purs.

    Retour
    ------
    df_clf : pd.DataFrame
        DataFrame complet contenant :
            - X_real, Y_real, Z_real   : composantes de l'état réel (features),
            - bloch_radius_real        : norme du vecteur de Bloch réel (feature),
            - is_pure                  : booléen (True = pur, False = mixte) (feature),
            - theta_ideal, phi_ideal, X_ideal, Y_ideal, Z_ideal : valeurs idéales (analyse),
            - label_purity             : int (1 = pur, 0 = mixte) (label à prédire).
    X : pd.DataFrame
        Sous-DataFrame des features pour la classification :
            colonnes ["X_real", "Y_real", "Z_real", "bloch_radius_real", "is_pure"].
    y : pd.Series
        Série des labels (0/1) correspondants à "label_purity".

    Exceptions
    ----------
    ValueError
        - si mixed_proportion n'est pas dans l'intervalle (0,1),
        - si la simulation ne produit pas AU MOINS un état pur et un état mixte.
    """

    # ----------------------------------------------------------------------
    # 0) Vérifications des paramètres d'entrée
    # ----------------------------------------------------------------------
    if not (0.0 < mixed_proportion < 1.0):
        raise ValueError("mixed_proportion doit être strictement entre 0 et 1.")

    # Paramètres internes FIXES (choix physiques)
    N_SHOTS = 1000
    MODE = "ideal"
    INCLUDE_IDEAL = True
    INCLUDE_DECOHERENCE = True
    DECOHERENCE_LEVEL = 0.6
    EPS_PURITY = 1e-2
    LABEL_COL = "label_purity"
    FEATURE_COLS = ["X_real", "Y_real", "Z_real", "bloch_radius_real", "is_pure"]

    # ----------------------------------------------------------------------
    # 1) Génération d'un dataset de base avec décohérence
    # ----------------------------------------------------------------------
    df_base = generate_qubit_tomography_dataset_base(
        n_states=n_states_total,
        n_shots=N_SHOTS,
        mode=MODE,
        include_ideal=INCLUDE_IDEAL,
        include_csv=False,
        csv_path=None,
        include_decoherence=INCLUDE_DECOHERENCE,
        decoherence_level=DECOHERENCE_LEVEL,
        random_state=None,  # aléatoire
    )

    # ----------------------------------------------------------------------
    # 2) Calcul du rayon de Bloch réel et définition du label pur / mixte
    # ----------------------------------------------------------------------
    r_real = np.sqrt(
        df_base["X_real"]**2
        + df_base["Y_real"]**2
        + df_base["Z_real"]**2
    )
    df_base["bloch_radius_real"] = r_real

    # Un état est considéré comme pur s'il est suffisamment proche de la sphère
    df_base["is_pure"] = df_base["bloch_radius_real"] >= (1.0 - EPS_PURITY)

    # Label numérique :
    #   1 = état pur
    #   0 = état mixte
    df_base[LABEL_COL] = df_base["is_pure"].astype(int)

    # ----------------------------------------------------------------------
    # 3) Construction d'un dataset avec la proportion cible mixed_proportion
    # ----------------------------------------------------------------------
    df_pure  = df_base[df_base[LABEL_COL] == 1]
    df_mixte = df_base[df_base[LABEL_COL] == 0]

    n_pure_available  = len(df_pure)
    n_mixed_available = len(df_mixte)

    if n_pure_available == 0 or n_mixed_available == 0:
        raise ValueError(
            "La simulation n'a pas produit suffisamment d'états des deux types "
            f"(purs: {n_pure_available}, mixtes: {n_mixed_available}). "
            "Augmente n_states_total ou ajuste les paramètres internes."
        )

    # Nombre CIBLE d'états mixtes et purs, avant ajustement
    target_mixed = mixed_proportion * n_states_total
    target_pure  = (1.0 - mixed_proportion) * n_states_total

    # On calcule un facteur d'échelle pour ne pas demander plus
    # d'états que ce que la simulation a produit.
    scale = min(
        n_mixed_available / target_mixed,
        n_pure_available  / target_pure,
        1.0
    )

    # Nombres EFFECTIFS à prélever dans chaque classe
    n_mixed = int(target_mixed * scale)
    n_pure  = int(target_pure  * scale)

    # Sécurité minimale
    if n_mixed == 0 or n_pure == 0:
        raise ValueError(
            "Impossible de construire un dataset respectant la proportion voulue "
            "avec au moins un exemple de chaque classe. "
            f"(après ajustement : n_mixed={n_mixed}, n_pure={n_pure})"
        )

    # Échantillonnage aléatoire dans chaque pool
    df_mixte_sample = df_mixte.sample(n=n_mixed, random_state=None)
    df_pure_sample  = df_pure.sample(n=n_pure,  random_state=None)

    # Concaténation
    df_clf = pd.concat([df_pure_sample, df_mixte_sample], ignore_index=True)

    # Mélange final pour casser tout ordre structurel
    df_clf = df_clf.sample(frac=1.0, random_state=None).reset_index(drop=True)

    # ----------------------------------------------------------------------
    # 4) Séparation features / labels pour le ML
    # ----------------------------------------------------------------------
    X = df_clf[FEATURE_COLS].copy()
    y = df_clf[LABEL_COL].copy()

    return df_clf, X, y
