# -*- coding: utf-8 -*-
# Objectif:
# - Définir 6 états purs à 1 qubit : |0>, |1>, |+>, |->, |i>, |-i>
# - Pour chaque état, simuler plusieurs mesures projectives dans les bases de Pauli X, Y, Z
# - Stocker les résultats dans un DataFrame pandas

import numpy as np
import pandas as pd
from qutip import basis, sigmax, sigmay, sigmaz, expect


# Définition des états purs
# |0> et |1> dans la base computationnelle
ket0 = basis(2, 0)                 # |0>
ket1 = basis(2, 1)                 # |1>

# |+> = (|0> + |1>)/sqrt(2)
ket_plus = (ket0 + ket1).unit()    # unit() normalise le vecteur
ket_moins = (ket0 - ket1).unit()

# |i> = (|0> + i|1>)/sqrt(2)
ket_i = (ket0 + 1j*ket1).unit()
ket_moins_i = (ket0 - 1j*ket1).unit()

states = {
    "0": ket0,
    "1": ket1,
    "+": ket_plus,
    "i": ket_i,
    "-": ket_moins,
    "-i": ket_moins_i
}


# Opérateurs de Pauli et mesures associées
paulis = {
    "X": sigmax(),   # σx
    "Y": sigmay(),   # σy
    "Z": sigmaz()    # σz
}

def sample_pauli_measurements(ket, pauli_op, n_shots, p_err=0.0, rng=None):
    """
    Simule n_shots mesures projectives de l'observable pauli_op sur l'état |psi>.
    Retourne une liste d'issues {+1, -1}.
    
    Principe:
    - On diagonalise pauli_op => eigenvalues (±1) et eigenkets
    - Probabilité de l'issue k: p_k = <psi|P_k|psi> = |<e_k|psi>|^2
    - On tire n_shots fois selon ces probabilités.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Diagonalisation (spectre et états propres)
    # eigvals sont les valeurs propres
    # eigkets sont les vecteurs propres
    eigvals, eigkets = pauli_op.eigenstates()

    # Probabilités Born p_k = <psi| (|e_k><e_k|) |psi>
    # Calcule les probabilités de mesure de ket (ket) 
    # dans la base des eigenkets fournis (ek * ek.dag() = A)
    probs = np.array(
        [expect(ek * ek.dag(), ket) for ek in eigkets],  # projecteur |e_k><e_k|
        dtype=float
    )
    probs /= probs.sum()

    # Tirage aléatoire des outcomes selon probs
    ideal_outcomes = rng.choice(eigvals, size=n_shots, p=probs)
    
    # --- AJOUT DU BRUIT DE MESURE ---
    if p_err > 0:
        flip_mask = rng.random(n_shots) < p_err   # booléen : True = flip
        noisy_outcomes = np.where(flip_mask, -ideal_outcomes, ideal_outcomes)
    else:
        noisy_outcomes = ideal_outcomes
    # --------------------------------

    # Conversion en float natif (eigvals Qutip -> numpy scalars)
    return [float(o) for o in noisy_outcomes]


def dataset_creation(n_shots, p_err=0.0):
    rows = []  # liste de lignes pour le DataFrame final

    for label_state, ket in states.items():
        for basis_name, pauli_op in paulis.items():
            outcomes = sample_pauli_measurements(ket, pauli_op, n_shots, p_err, rng=None)
            # On ajoute une ligne par "shot" (mesure individuelle)
            for shot_id, outcome in enumerate(outcomes, start=1):
                rows.append({
                    "state": label_state,  # étiquette de l'état préparé
                    "basis": basis_name,   # base de mesure (X,Y,Z)
                    "shot": shot_id,       # numéro de mesure répétée
                    "outcome": outcome     # résultat ±1
                })

    df = pd.DataFrame(rows)
    
    return df


# Parametres des datasets qu'on veut créer
nombre_shots = [100, 500, 1000]
error = [0.0, 0.02, 0.1]

# ! Création des CSVs !

for shots in nombre_shots:
    for err in error:
        dataframe = dataset_creation(shots, err)
        filename = f"../data/mesures_pauli_err{err:.3f}_shots{shots}.csv"
        dataframe.to_csv(filename, index=False)

print("CSV générés")