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

---

## 🛠️ Installation et Lancement

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/votre-nom-d-utilisateur/sophea-frugal-portfolio-optimization.git](https://github.com/votre-nom-d-utilisateur/sophea-frugal-portfolio-optimization.git)
   cd sophea-frugal-portfolio-optimization
  2. **Installer les dépendances et lancer le frontend:**
   ```bash
   pip install - r requirements.txt
   streamlit run dashboard.py
