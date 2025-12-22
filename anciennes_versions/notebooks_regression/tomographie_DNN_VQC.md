# Notebook 2: Advanced Quantum Tomography
## Deep Learning (DNN) & Variational Quantum Circuits (VQC) vs. Classical SVR

## 1. Scientific Context & Exploration Goals
In the previous notebook, we established a robust baseline using **Support Vector Regression (SVR)**, a classical kernel-based method. While effective, SVR relies on fixed kernels (RBF, Polynomial) which may not perfectly capture the complex geometry of quantum states under decoherence.

In this second exploratory phase, we move towards **Advanced Architectures**. Our goal is to determine if models with higher representational capacity (Deep Learning) or native quantum priors (VQC) can surpass the classical baseline, particularly in the regime of **impure states (decoherence)** and **limited data**.

We investigate two challengers:
1.  **Deep Neural Networks (DNN):** Using the universal approximation theorem to model the mapping from measurements to density matrices with high non-linearity.
2.  **Variational Quantum Circuits (VQC):** A "Quantum Machine Learning" approach. We hypothesize that a quantum circuit possesses a natural *inductive bias* for quantum data, potentially requiring fewer parameters to represent the Hilbert space than a classical network.

## 2. A Paradigm Shift: "Physics-Informed" Training
Unlike standard regression which minimizes the Euclidean distance (MSE), we introduce a **Custom Loss Function** grounded in Quantum Information Theory.

### The "Fidelity-Based" Backpropagation
Standard ML optimizes geometry. We want to optimize **physics**.
Instead of blindingly minimizing $MSE = ||\vec{r}_{pred} - \vec{r}_{real}||^2$, we configure our Neural Networks (both Classical and Quantum) to directly maximize the **Quantum Fidelity** ($F$).

During the **Backpropagation** pass, the gradient of the Fidelity is computed with respect to the model weights. This forces the optimizer to prioritize directions that increase the physical overlap between the predicted and true states.

**The Mathematical Loss Function:**
For a single qubit state defined by a Bloch vector $\vec{r}$, the loss $\mathcal{L}$ to minimize is:

$$\mathcal{L} = 1 - F(\rho_{pred}, \rho_{real})$$

Where the Fidelity $F$ for single-qubit Bloch vectors is given analytically by:
$$F(\vec{r}_{p}, \vec{r}_{t}) = \frac{1}{2} \left( 1 + \vec{r}_{p} \cdot \vec{r}_{t} + \sqrt{(1 - ||\vec{r}_{p}||^2)(1 - ||\vec{r}_{t}||^2)} \right)$$

* **Interpretation:** The term $\vec{r}_{p} \cdot \vec{r}_{t}$ aligns the vectors directionally. The term under the square root penalizes errors in **purity** (vector length). This allows the DNN to specifically "learn" decoherence.

## 3. Architecture Overview

### A. Deep Neural Network (DNN - PyTorch)
* **Structure:** A Multi-Layer Perceptron (MLP) with fully connected layers and non-linear activation functions (ReLU).
* **Why:** To test if a "Universal Approximator" can learn the noise models better than fixed kernels.

### B. Variational Quantum Circuit (VQC - PennyLane)
* **Concept:** We use a parameterized quantum circuit as the model.
* **Mechanism:**
    1.  **Encoding:** Classical inputs ($X, Y, Z$) are embedded into a quantum state via rotation gates.
    2.  **Processing:** A sequence of trainable gates (Ansatz) manipulates the state.
    3.  **Measurement:** We measure the expectation values of Pauli operators to obtain the output vector.
* **Hypothesis:** "Quantum for Quantum". A quantum circuit naturally evolves on the Bloch sphere (or inside it for mixed states via subsystems), which might offer better generalization with fewer parameters.

## 4. Implementation: High-Performance Computing (GPU)
With SVC, we had quite some long training time. So in order to handle the computational load of training deep networks and simulating quantum circuits, we leverage **GPU Acceleration**:
* **PyTorch (CUDA/MPS):** For tensor operations and automatic differentiation of the DNN.
* **PennyLane Lightning GPU:** Using high-performance state-vector simulators (like `lightning.gpu` or `lightning.qubit`) to accelerate the VQC simulation and gradient calculation (adjoint differentiation).

We also do this as a way to learn modern high-performance ML pipelines.

## 5. Input/Output Interfaces
To ensure a rigorous comparison with the SVR baseline from Notebook 1, the I/O structure remains identical:

* **Input $\mathbf{X}$:** Noisy measurement expectations $[ \langle X \rangle_{noise}, \langle Y \rangle_{noise}, \langle Z \rangle_{noise} ]$.
* **Output $\mathbf{y}$:** Predicted Bloch vector components $[\hat{x}, \hat{y}, \hat{z}]$.

*Note: The predicted vector is implicitly constrained to valid physical states (norm $\le$ 1) either via activation functions (Tanh) or penalty terms in the loss.*


```python
import torch
import torch.nn as nn
import pennylane as qml

# ==========================================
# 1. SETUP GPU & DEVICE AGNOSTIC CODE
# ==========================================
def get_device():
    """Détecte automatiquement le meilleur accélérateur disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Pour les Mac M1/M2/M3 (Metal Performance Shaders)
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"✅ Computation Device: {DEVICE}")

# ==========================================
# 2. CUSTOM LOSS : QUANTUM FIDELITY
# ==========================================
class QuantumFidelityLoss(nn.Module):
    """
    Fonction de perte inversée : Loss = 1 - Fidélité.
    Pousse le réseau à maximiser la superposition physique avec l'état cible.
    """
    def __init__(self):
        super(QuantumFidelityLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # y_pred, y_true shape: (batch_size, 3) correspondant à [X, Y, Z]
        
        # 1. Calcul des normes carrées (r^2)
        # On clippe à 1.0 - epsilon pour éviter les racines de nombres négatifs
        # si le réseau prédit temporairement un vecteur > 1.
        r2_pred = torch.sum(y_pred**2, dim=1).clamp(max=1.0 - 1e-6)
        r2_true = torch.sum(y_true**2, dim=1).clamp(max=1.0 - 1e-6)
        
        # 2. Produit scalaire (Orientation)
        dot_prod = torch.sum(y_pred * y_true, dim=1)
        
        # 3. Terme de pureté (Grandeur)
        # Formule : sqrt((1 - r_pred^2)(1 - r_true^2))
        purity_term = torch.sqrt((1.0 - r2_pred) * (1.0 - r2_true))
        
        # 4. Fidélité
        fidelity = 0.5 * (1.0 + dot_prod + purity_term)
        
        # 5. On retourne la perte moyenne (on veut minimiser 1 - F)
        return (1.0 - fidelity).mean()

# ==========================================
# 3. MODEL 1: DEEP NEURAL NETWORK (DNN)
# ==========================================
class TomographyDNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super(TomographyDNN, self).__init__()
        
        self.net = nn.Sequential(
            # Entrée : X_mean, Y_mean, Z_mean
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            # Sortie : X_pred, Y_pred, Z_pred
            nn.Linear(hidden_dim, output_dim),
            
            # Tanh force la sortie entre [-1, 1], ce qui aide physiquement
            # car une coordonnée de Bloch ne peut pas dépasser 1.
            nn.Tanh() 
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. MODEL 2: VARIATIONAL QUANTUM CIRCUIT (VQC)
# ==========================================
# Configuration du simulateur quantique
# 'lightning.qubit' est un simulateur C++ rapide pour CPU.
# Pour le GPU 'vrai', il faut 'lightning.gpu' (voir explications plus bas).
n_qubits = 4
dev = qml.device("lightning.qubit", wires=n_qubits) 

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs, weights):
    """
    Circuit Variationnel.
    inputs: Données classiques (Batch)
    weights: Paramètres apprenables du circuit
    """
    # 1. Encodage des données (Angle Embedding)
    # On encode les features X, Y, Z dans les rotations des qubits
    # On répète les inputs pour remplir les 4 qubits si nécessaire
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    
    # 2. Couches Variationnelles (Ansatz)
    # StrongEntanglingLayers est très expressif pour apprendre des états complexes
    qml.StrongEntanglingLayers(weights, wires=range(n_qubits))
    
    # 3. Mesures : On veut 3 sorties (X, Y, Z)
    # On mesure les espérances de Pauli sur les 3 premiers qubits pour récupérer 3 valeurs.
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]

class TomographyVQC(nn.Module):
    def __init__(self, n_layers=3):
        super(TomographyVQC, self).__init__()
        
        # Définition de la forme des poids pour StrongEntanglingLayers
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # qml.qnn.TorchLayer transforme le QNode PennyLane en une couche PyTorch standard !
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Optionnel : Une petite couche classique pour "calibrer" la sortie quantique
        # Cela aide le VQC à mapper exactement vers l'espace de Bloch cible.
        self.post_processing = nn.Linear(3, 3) 

    def forward(self, x):
        # On passe dans le quantique
        x_q = self.q_layer(x)
        # On ajuste légèrement l'échelle
        return torch.tanh(self.post_processing(x_q))
```


```python

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 14
          1 {
          2   "cells": [
          3     {
          4       "cell_type": "markdown",
          5       "metadata": {},
          6       "source": [
          7         "# Notebook 2 \u2014 DNN vs VQC vs baseline\n",
          8         "Boucle d'exploration pour comparer le DNN, le VQC et une baseline rapide (SVR/MLE approxim\u00e9e) en fonction du niveau de d\u00e9coh\u00e9rence. Les classes `TomographyDNN`, `TomographyVQC` et `QuantumFidelityLoss` sont suppos\u00e9es d\u00e9j\u00e0 d\u00e9finies dans la cellule pr\u00e9c\u00e9dente.\n"
          9       ]
         10     },
         11     {
         12       "cell_type": "code",
         13       "metadata": {},
    ---> 14       "execution_count": null,
         15       "outputs": [],
         16       "source": [
         17         "import time\n",
         18         "import numpy as np\n",
         19         "import pandas as pd\n",
         20         "import torch\n",
         21         "from torch.utils.data import DataLoader, TensorDataset\n",
         22         "from sklearn.model_selection import train_test_split\n",
         23         "import matplotlib.pyplot as plt\n",
         24         "\n",
         25         "# Import du g\u00e9n\u00e9rateur de dataset\n",
         26         "try:\n",
         27         "    from saint_dtSet import generate_qubit_tomography_dataset_base\n",
         28         "except ImportError:\n",
         29         "    from dataset_build.saint_dtSet import generate_qubit_tomography_dataset_base\n",
         30         "\n",
         31         "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
         32         "torch.manual_seed(0)\n",
         33         "np.random.seed(0)\n",
         34         "print(f\"Device: {DEVICE}\")\n"
         35       ]
         36     },
         37     {
         38       "cell_type": "code",
         39       "metadata": {},
         40       "execution_count": null,
         41       "outputs": [],
         42       "source": [
         43         "def fidelity_from_bloch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:\n",
         44         "    \"\"\"\n",
         45         "    Calcul analytique de la fid\u00e9lit\u00e9 entre deux vecteurs de Bloch (qubit).\n",
         46         "    pred et target: tensors (..., 3).\n",
         47         "    \"\"\"\n",
         48         "    dot = (pred * target).sum(dim=-1)\n",
         49         "    norm_pred = (pred ** 2).sum(dim=-1)\n",
         50         "    norm_target = (target ** 2).sum(dim=-1)\n",
         51         "    under_sqrt = torch.clamp(1.0 - norm_pred, min=0.0) * torch.clamp(1.0 - norm_target, min=0.0)\n",
         52         "    fidelity = 0.5 * (1.0 + dot + torch.sqrt(under_sqrt))\n",
         53         "    return torch.clamp(fidelity, 0.0, 1.0)\n",
         54         "\n",
         55         "\n",
         56         "def build_dataloaders(df: pd.DataFrame, batch_size: int = 32):\n",
         57         "    features = df[['X_mean', 'Y_mean', 'Z_mean']].to_numpy(dtype=np.float32)\n",
         58         "    targets = df[['X_real', 'Y_real', 'Z_real']].to_numpy(dtype=np.float32)\n",
         59         "    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42, shuffle=True)\n",
         60         "\n",
         61         "    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))\n",
         62         "    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))\n",
         63         "\n",
         64         "    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)\n",
         65         "    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)\n",
         66         "    return train_loader, test_loader, (X_test, y_test)\n",
         67         "\n",
         68         "\n",
         69         "def evaluate_model(model, data_loader, device=DEVICE):\n",
         70         "    model.eval()\n",
         71         "    preds, targets = [], []\n",
         72         "    with torch.no_grad():\n",
         73         "        for xb, yb in data_loader:\n",
         74         "            xb = xb.to(device)\n",
         75         "            yb = yb.to(device)\n",
         76         "            pred = model(xb)\n",
         77         "            preds.append(pred.detach().cpu())\n",
         78         "            targets.append(yb.detach().cpu())\n",
         79         "    preds_t = torch.cat(preds, dim=0)\n",
         80         "    targets_t = torch.cat(targets, dim=0)\n",
         81         "    fidelities = fidelity_from_bloch(preds_t, targets_t)\n",
         82         "    return fidelities.mean().item()\n",
         83         "\n",
         84         "\n",
         85         "def compute_baseline_fidelity(x_test: np.ndarray, y_test: np.ndarray) -> float:\n",
         86         "    # Baseline rapide ~SVR/MLE : pr\u00e9dicteur trivial X_mean->X_real (sans contrainte de sph\u00e8re).\n",
         87         "    preds = torch.from_numpy(x_test)\n",
         88         "    targets = torch.from_numpy(y_test)\n",
         89         "    return fidelity_from_bloch(preds, targets).mean().item()\n"
         90       ]
         91     },
         92     {
         93       "cell_type": "code",
         94       "metadata": {},
         95       "execution_count": null,
         96       "outputs": [],
         97       "source": [
         98         "def train_model(model, train_loader, optimizer, loss_fn, epochs: int, device=DEVICE):\n",
         99         "    \"\"\"\n",
        100         "    Boucle g\u00e9n\u00e9rique d'entra\u00eenement.\n",
        101         "    D\u00e9place inputs/targets sur DEVICE avant forward comme demand\u00e9.\n",
        102         "    \"\"\"\n",
        103         "    model.to(device)\n",
        104         "    history = []\n",
        105         "    for epoch in range(1, epochs + 1):\n",
        106         "        model.train()\n",
        107         "        running_loss = 0.0\n",
        108         "        for xb, yb in train_loader:\n",
        109         "            xb = xb.to(device)\n",
        110         "            yb = yb.to(device)\n",
        111         "            optimizer.zero_grad()\n",
        112         "            preds = model(xb)\n",
        113         "            loss = loss_fn(preds, yb)\n",
        114         "            loss.backward()\n",
        115         "            optimizer.step()\n",
        116         "            running_loss += loss.item() * xb.size(0)\n",
        117         "\n",
        118         "        epoch_loss = running_loss / len(train_loader.dataset)\n",
        119         "        history.append(epoch_loss)\n",
        120         "        if epoch == 1 or epoch % max(1, epochs // 5) == 0:\n",
        121         "            print(f\"Epoch {epoch:>3}/{epochs} - loss: {epoch_loss:.4f}\")\n",
        122         "    return history\n"
        123       ]
        124     },
        125     {
        126       "cell_type": "code",
        127       "metadata": {},
        128       "execution_count": null,
        129       "outputs": [],
        130       "source": [
        131         "# ---------------------------------------------------------------------------\n",
        132         "# Boucle d'exp\u00e9rimentation principale\n",
        133         "# ---------------------------------------------------------------------------\n",
        134         "N_STATES = 2000\n",
        135         "N_SHOTS = 500\n",
        136         "DECOHERENCE_LEVELS = [0.0, 0.2, 0.5, 0.8]\n",
        137         "BATCH_SIZE = 32\n",
        138         "EPOCHS_DNN = 25\n",
        139         "EPOCHS_VQC = 12  # VQC plus lent -> moins d'epochs\n",
        140         "LR_DNN = 0.01\n",
        141         "LR_VQC = 0.05\n",
        142         "\n",
        143         "results = {\"dnn\": [], \"vqc\": [], \"baseline\": []}\n",
        144         "\n",
        145         "for level in DECOHERENCE_LEVELS:\n",
        146         "    print(f\"\n=== D\u00e9coh\u00e9rence {level} ===\")\n",
        147         "    df = generate_qubit_tomography_dataset_base(\n",
        148         "        n_states=N_STATES,\n",
        149         "        n_shots=N_SHOTS,\n",
        150         "        include_decoherence=True,\n",
        151         "        decoherence_level=level,\n",
        152         "        mode=\"finite_shots\",\n",
        153         "        random_state=1234\n",
        154         "    )\n",
        155         "\n",
        156         "    train_loader, test_loader, (x_test, y_test) = build_dataloaders(df, batch_size=BATCH_SIZE)\n",
        157         "\n",
        158         "    # ---------------------- DNN ----------------------\n",
        159         "    dnn = TomographyDNN().to(DEVICE)\n",
        160         "    dnn_optimizer = torch.optim.Adam(dnn.parameters(), lr=LR_DNN)\n",
        161         "    dnn_loss = QuantumFidelityLoss()\n",
        162         "    _ = train_model(dnn, train_loader, dnn_optimizer, dnn_loss, epochs=EPOCHS_DNN, device=DEVICE)\n",
        163         "    dnn_fid = evaluate_model(dnn, test_loader, device=DEVICE)\n",
        164         "\n",
        165         "    # ---------------------- VQC ----------------------\n",
        166         "    vqc = TomographyVQC().to(DEVICE)\n",
        167         "    vqc_optimizer = torch.optim.Adam(vqc.parameters(), lr=LR_VQC)\n",
        168         "    vqc_loss = QuantumFidelityLoss()\n",
        169         "    _ = train_model(vqc, train_loader, vqc_optimizer, vqc_loss, epochs=EPOCHS_VQC, device=DEVICE)\n",
        170         "    vqc_fid = evaluate_model(vqc, test_loader, device=DEVICE)\n",
        171         "\n",
        172         "    # -------------------- Baseline --------------------\n",
        173         "    baseline_fid = compute_baseline_fidelity(x_test, y_test)\n",
        174         "\n",
        175         "    results[\"dnn\"].append(dnn_fid)\n",
        176         "    results[\"vqc\"].append(vqc_fid)\n",
        177         "    results[\"baseline\"].append(baseline_fid)\n",
        178         "\n",
        179         "    print(f\"DNN fidelity (test): {dnn_fid:.4f}\")\n",
        180         "    print(f\"VQC fidelity (test): {vqc_fid:.4f}\")\n",
        181         "    print(f\"Baseline fidelity    : {baseline_fid:.4f}\")\n",
        182         "\n",
        183         "print('Boucle termin\u00e9e.')\n",
        184         "results\n"
        185       ]
        186     },
        187     {
        188       "cell_type": "code",
        189       "metadata": {},
        190       "execution_count": null,
        191       "outputs": [],
        192       "source": [
        193         "# Visualisation des fid\u00e9lit\u00e9s moyennes\n",
        194         "plt.figure(figsize=(8, 5))\n",
        195         "plt.plot(DECOHERENCE_LEVELS, results['dnn'], '-o', label='DNN')\n",
        196         "plt.plot(DECOHERENCE_LEVELS, results['vqc'], '-o', label='VQC')\n",
        197         "plt.plot(DECOHERENCE_LEVELS, results['baseline'], '-o', label='Baseline (SVR/MLE rapide)')\n",
        198         "plt.xlabel('Niveau de d\u00e9coh\u00e9rence')\n",
        199         "plt.ylabel('Fid\u00e9lit\u00e9 moyenne (test)')\n",
        200         "plt.ylim(0.0, 1.05)\n",
        201         "plt.grid(True, alpha=0.3)\n",
        202         "plt.legend()\n",
        203         "plt.show()\n"
        204       ]
        205     }
        206   ],
        207   "metadata": {
        208     "kernelspec": {
        209       "display_name": "Python 3",
        210       "language": "python",
        211       "name": "python3"
        212     },
        213     "language_info": {
        214       "codemirror_mode": {
        215         "name": "ipython",
        216         "version": 3
        217       },
        218       "file_extension": ".py",
        219       "mimetype": "text/x-python",
        220       "name": "python",
        221       "nbconvert_exporter": "python",
        222       "pygments_lexer": "ipython3",
        223       "version": "3.11.9"
        224     }
        225   },
        226   "nbformat": 4,
        227   "nbformat_minor": 5
        228 }
    

    NameError: name 'null' is not defined

