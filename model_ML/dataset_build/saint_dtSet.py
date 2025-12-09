import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

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

    !!! Sauvegarde n_shots dans les métadonnées du DataFrame. car c'est utilisé par le MLE !!!
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
            # On ajoute n_shots dans chaque ligne pour la sécurité, 
            # ou on peut l'avoir juste en attribut, mais par ligne c'est plus sûr pour le concat
            "n_shots_sim": n_shots
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

    # Sécurité supplémentaire : on attache n_shots aux attributs du dataframe
    df.attrs['n_shots'] = n_shots
    # Option : sauvegarde en CSV
    if include_csv:
        if csv_path is None:
            csv_path = f"tomography_dataset_{n_states}_shots{n_shots}.csv"
        df.to_csv(csv_path, index=False)

    return df


# ---------------------------------------------------------------------------
# 2) Fonction MLE Optimisée (Maximum Likelihood Estimation)
# ---------------------------------------------------------------------------

def perform_mle_tomography(df_input: pd.DataFrame, n_shots: int = None) -> tuple[pd.DataFrame, float]:
    """
    Applique le MLE sur le dataset.
    Détecte automatiquement n_shots si présent dans le dataframe.
    """
    # 1. Gestion automatique de n_shots
    if n_shots is None:
        if 'n_shots' in df_input.attrs:
            n_shots = df_input.attrs['n_shots']
        elif 'n_shots_sim' in df_input.columns:
            # On prend la valeur de la première ligne (supposée constante)
            n_shots = int(df_input['n_shots_sim'].iloc[0])
        else:
            raise ValueError("n_shots non fourni et introuvable dans le DataFrame. Veuillez spécifier l'argument n_shots.")

    # --- Fonctions internes MLE ---
    def bloch_from_angles(theta, phi):
        return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)

    def angles_from_bloch(nx, ny, nz):
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        if norm == 0: return 0.0, 0.0
        nx, ny, nz = nx/norm, ny/norm, nz/norm
        theta = np.arccos(np.clip(nz, -1.0, 1.0))
        phi = np.arctan2(ny, nx)
        return theta, (phi + 2*np.pi) % (2*np.pi)

    def neg_log_likelihood(params, n_x, n_y, n_z, N):
        theta, phi = params
        nx, ny, nz = bloch_from_angles(theta, phi)
        # Proba théorique
        px, py, pz = (1+nx)/2, (1+ny)/2, (1+nz)/2
        # Clipping de sécurité
        eps = 1e-12
        px, py, pz = np.clip([px, py, pz], eps, 1-eps)
        # NLL
        ll = (n_x*np.log(px) + (N-n_x)*np.log(1-px) +
              n_y*np.log(py) + (N-n_y)*np.log(1-py) +
              n_z*np.log(pz) + (N-n_z)*np.log(1-pz))
        return -ll

    # --- Exécution ---
    df_result = df_input.copy()
    start_time = time.time()
    
    # Recalcul des comptes (counts)
    if "numberX" not in df_result.columns:
        df_result["numberX"] = ((1 + df_result["X_mean"]) / 2.0 * n_shots).round().astype(int)
        df_result["numberY"] = ((1 + df_result["Y_mean"]) / 2.0 * n_shots).round().astype(int)
        df_result["numberZ"] = ((1 + df_result["Z_mean"]) / 2.0 * n_shots).round().astype(int)

    theta_mle_list, phi_mle_list = [], []
    bounds = [(0.0, np.pi), (0.0, 2.0 * np.pi)]

    for _, row in df_result.iterrows():
        # Point de départ : inversion linéaire
        t0, p0 = angles_from_bloch(row["X_mean"], row["Y_mean"], row["Z_mean"])
        
        # Optimisation
        res = minimize(
            neg_log_likelihood, 
            x0=[t0, p0], 
            args=(row["numberX"], row["numberY"], row["numberZ"], n_shots),
            bounds=bounds, 
            method="L-BFGS-B"
        )
        theta_mle_list.append(res.x[0])
        phi_mle_list.append(res.x[1] % (2*np.pi))

    end_time = time.time()
    
    df_result["theta_mle"] = theta_mle_list
    df_result["phi_mle"] = phi_mle_list
    
    # Conversion en cartésien pour comparaison facile
    nx, ny, nz = bloch_from_angles(np.array(theta_mle_list), np.array(phi_mle_list))
    df_result["X_mle"], df_result["Y_mle"], df_result["Z_mle"] = nx, ny, nz
    
    return df_result, (end_time - start_time)

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
