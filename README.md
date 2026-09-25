# 🌱 SOPHEA : Système d'Optimisation de Portefeuille à Haute Efficacité Algorithmique

SOPHEA est un système d'information "Basse Consommation" appliqué à l'ingénierie financière. Développé dans le cadre de la SAE sur les systèmes d'information frugaux, ce projet concilie les exigences de précision de la finance quantitative avec les impératifs de la sobriété numérique (*Green Coding*).

## 📊 Piliers de l'Approche Frugale

1. **Stratégie de Données (*Small Data*) :** 
   - Privilégie la qualité et la pertinence des séries temporelles épurées (KPIs) plutôt que le traitement lourd de pétaoctets de données non structurées (*Big Data*).

2. **Optimisation Stochastique (*Quasi-Monte Carlo* & *Juste-Assez*) :**
   - Substitution du hasard pur par des suites à faible discrépance de Sobol pour accélérer la convergence mathématique.
   - Implémentation d'un modèle d'Intelligence Artificielle (régression linéaire supervisée) prédisant dynamiquement le nombre optimal de simulations requis en fonction de la volatilité du marché.

3. **Efficacité Algorithmique & Vectorisation :**
   - Résolution de l'optimisation moyenne-variance de Markowitz de manière vectorisée via `NumPy` pour minimiser l'empreinte computationnelle.

## English Below 

# 🌱 SOPHEA: High Algorithmic Efficiency Portfolio Optimization System

SOPHEA is a "Low-Energy" information system applied to financial engineering. Developed as part of the frugal information systems academic project, this repository bridges the precision demands of quantitative finance with the imperatives of digital sobriety (*Green Coding*).

## 📊 Pillars of the Frugal Approach

1. **Data Strategy (*Small Data*):** 
   - Prioritizes the quality and relevance of streamlined time-series datasets (KPIs) over the heavy processing of petabytes of unstructured data (*Big Data*).

2. **Stochastic Optimization (*Quasi-Monte Carlo* & *Just-Enough*):**
   - Substitution of pure randomness with low-discrepancy Sobol sequences to accelerate mathematical convergence.
   - Implementation of an Artificial Intelligence model (supervised linear regression) that dynamically predicts the optimal number of simulations required based on market volatility.

3. **Algorithmic Efficiency & Vectorization:**
   - Solving Markowitz mean-variance optimization in a vectorized manner using `NumPy` to minimize computational footprint.
---

## 🛠️ Installation et Lancement

1. **Cloner le dépôt / Clone the reposit :**
   ```bash
   git clone [https://github.com/votre-nom-d-utilisateur/sophea-frugal-portfolio-optimization.git](https://github.com/votre-nom-d-utilisateur/sophea-frugal-portfolio-optimization.git)
   cd sophea-frugal-portfolio-optimization
  2. **Installer les dépendances et lancer le frontend / Install requirements and run frontend :**
   ```bash
   pip install - r requirements.txt
   streamlit run dashboard.py
