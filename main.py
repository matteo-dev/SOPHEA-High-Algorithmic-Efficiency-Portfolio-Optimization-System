# Projet : Optimisation de Portefeuille ESG avec Markowitz et QMC
# Partie Backend - main.py

# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from scipy.stats import qmc # Générateur de Sobol pour les simulations Quasi-Monte Carlo
import time
from functools import wraps # Pour mesurer le temps d'exécution de manière élégante
import yfinance as yf # Pour collecter des données financières réelles (Yahoo Finance)
from sklearn.linear_model import LinearRegression # Pour entraîner notre modèle de Machine Learning supervisé
from sklearn.model_selection import train_test_split # Pour séparer les données d'entraînement et de test

# Fonction pour générer le nombre de simulation requis en fonction de la volatilité et du bruit
def generer_donnees_apprentissage_ia(n_samples=1000):

    np.random.seed(42)

    # Volatilité historique allant de 5% à 40% (fictif pour l'exemple)
    volatilite_historique = np.random.uniform(0.05, 0.40, n_samples)
    
    # Le nombre de simulations idéal avec du bruit 
    simulations_requises = 20000 * volatilite_historique + np.random.normal(0, 500, n_samples)
    
    # On borne entre un minimum et un maximum
    simulations_requises = np.clip(simulations_requises, 128, 65536).astype(int)
    
    X = volatilite_historique.reshape(-1, 1)
    y = simulations_requises
    return X, y

# Fonction pour l'entraînement d'un modèle de Machine Learning supervisé 
def entrainer_modele_ia():
    X, y = generer_donnees_apprentissage_ia()
    
    # Séparation Train (80%) / Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Choix d'un modèle simple de régression linéaire 
    modele = LinearRegression()
    modele.fit(X_train, y_train)
    
    # Évaluation de la précision de l'IA
    score_test = modele.score(X_test, y_test)
    
    return modele, score_test

# Décorateur pour mesurer le temps d'exécution et l'efficacité énergétique de nos fonctions d'optimisation
def mesure_frugalite(func):
    @wraps(func)
    # Fonction wrapper pour mesurer le temps d'exécution et afficher un message de performance
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"[Eco-Monitor] Temps d'exécution de '{func.__name__}' : {execution_time:.6f} secondes")
        return result, execution_time
    return wrapper

# Fonction pour collecter des données fictives sur les fonds ESG, avec des rendements attendus et une matrice de covariance simplifiée
def collecter_small_data_esg():
    actifs = ["Fonds_ESG_Eau", "Fonds_ESG_Solaire", "Fonds_ESG_Social", "Fonds_ESG_Gouvernance"]
    
    # Rendements annuels espérés (fictifs pour l'exemple)
    rendements_attendus = np.array([0.08, 0.12, 0.06, 0.07])
    
    # Matrice de covariance (risque et corrélations entre les fonds)
    matrice_cov = np.array([
        [0.04, 0.01, 0.005, 0.002],
        [0.01, 0.06, 0.004, 0.001],
        [0.005, 0.004, 0.03, 0.008],
        [0.002, 0.001, 0.008, 0.02]
    ])
    
    return actifs, rendements_attendus, matrice_cov

# Fonction pour générer des poids de portefeuille à l'aide de la méthode Quasi-Monte Carlo (Sobol), 
# avec une transformation pour simuler une distribution exponentielle et normaliser les poids
def generer_poids_qmc(n_simulations, n_actifs):

    # Initialisation du générateur de Sobol
    sampler = qmc.Sobol(d=n_actifs, scramble=True)
    
    # Sobol est optimal avec des puissances de 2. 
    # On trouve la puissance de 2 la plus proche pour m.
    m = int(max(1, np.round(np.log2(n_simulations)))) # Arrondi de sécurité ajouté pour l'IA
    echantillons_qmc = sampler.random_base2(m=m)
    
    # Transformation mathématique : on utilise un logarithme négatif pour 
    # simuler une distribution exponentielle, puis on normalise pour que 
    # la somme des poids de chaque portefeuille soit égale à 1 (100%).
    echantillons_exp = -np.log(echantillons_qmc + 1e-9) # 1e-9 pour éviter log(0)
    poids_portefeuilles = echantillons_exp / np.sum(echantillons_exp, axis=1, keepdims=True)
    
    return poids_portefeuilles

@mesure_frugalite
# Fonction d'optimisation de portefeuille utilisant Markowitz et Quasi-Monte Carlo, avec une approche "Juste Assez" 
# pour limiter les simulations en fonction du niveau de précision souhaité
def optimiser_markowitz_frugal(rendements_attendus, matrice_cov, niveau_precision):
    n_actifs = len(rendements_attendus)
    
    # Application stricte du concept "Juste Assez"
    # Est-il nécessaire de faire 1 000 000 de simulations pour une précision à 10^-6 ?
    if isinstance(niveau_precision, int): 
        # L'IA a prédit un nombre exact de simulations !
        n_simulations = niveau_precision
    elif niveau_precision == "Faible (10^-2)":
        n_simulations = 2**7   # 128 simulations (Très rapide, très green)
    elif niveau_precision == "Moyenne (10^-4)":
        n_simulations = 2**11  # 2048 simulations (Équilibre)
    elif niveau_precision == "Haute (10^-6)":
        n_simulations = 2**16  # 65536 simulations (Plus coûteux en énergie)
    else:
        n_simulations = 2**9   # 512 par défaut
        
    # 1. Génération vectorisée des poids (pas de boucle 'for', c'est du Green Coding Python)
    poids = generer_poids_qmc(n_simulations, n_actifs)
    
    # Mise à jour du nombre réel de simulations générées par Sobol (puissance de 2)
    simulations_reelles = len(poids)
    
    # 2. Calcul vectorisé des rendements pour chaque portefeuille simulé
    rendements_portefeuilles = np.dot(poids, rendements_attendus)
    
    # 3. Calcul vectorisé des risques (volatilité/variance)
    # L'utilisation de 'einsum' ou d'opérations matricielles est vitale pour la performance
    variance_portefeuilles = np.sum(poids * (poids @ matrice_cov), axis=1)
    risques_portefeuilles = np.sqrt(variance_portefeuilles)
    
    # 4. Évaluation de la performance (Ratio de Sharpe, avec taux sans risque = 0 pour simplifier)
    ratios_sharpe = rendements_portefeuilles / risques_portefeuilles
    
    # 5. Identification du portefeuille optimal (celui qui maximise le ratio de Sharpe)
    index_optimal = np.argmax(ratios_sharpe)
    poids_optimaux = poids[index_optimal]
    
    # Compilation des résultats dans un dictionnaire pour une utilisation facile dans le dashboard
    resultats = {
        "poids_optimaux": poids_optimaux,
        "rendement_attendu": rendements_portefeuilles[index_optimal],
        "risque_estime": risques_portefeuilles[index_optimal],
        "ratio_sharpe": ratios_sharpe[index_optimal],
        "simulations_realisees": simulations_reelles
    }
    
    return resultats

# Fonction pour collecter des données financières réelles à partir de Yahoo Finance, 
# avec une approche frugale pour limiter l'empreinte réseau et gérer les erreurs de manière robuste
def collecter_donnees_reelles(tickers, periode="2y", intervalle="1mo"):
    if not tickers:
        return [], np.array([]), np.array([])
        
    # Téléchargement frugal
    data = yf.download(tickers, period=periode, interval=intervalle, progress=False)
    
    # Sécurité 1 : Si Yahoo Finance ne trouve rien ou s'il y a une erreur réseau
    if data.empty:
        return [], np.array([]), np.array([])
        
    # Sécurité 2 : Gestion robuste du nom de la colonne (Adj Close vs Close)
    if 'Adj Close' in data.columns or (isinstance(data.columns, pd.MultiIndex) and 'Adj Close' in data.columns.levels[0]):
        prix = data['Adj Close']
    elif 'Close' in data.columns or (isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.levels[0]):
        prix = data['Close']
    else:
        return [], np.array([]), np.array([])
        
    # Sécurité 3 : Forcer en DataFrame si un seul ticker
    if isinstance(prix, pd.Series):
        prix = prix.to_frame()
        prix.columns = tickers
        
    # Calcul des rendements mensuels
    rendements = prix.pct_change().dropna()
    
    # Annualisation simple (12 mois)
    rendements_attendus = rendements.mean() * 12
    matrice_cov = rendements.cov() * 12
    
    return list(rendements_attendus.index), rendements_attendus.values, matrice_cov.values