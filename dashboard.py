# Projet : Optimisation de Portefeuille ESG avec Markowitz et QMC
# Partie Frontend - dashboard.py

# Importation des bibliothèques nécessaires pour le dashboard
import streamlit as st
import pandas as pd
import numpy as np

# Importation des fonctions depuis notre backend (main.py)
from main import collecter_small_data_esg, optimiser_markowitz_frugal, collecter_donnees_reelles, entrainer_modele_ia

st.set_page_config(page_title="SOPHEA - Finance Verte", page_icon="🌱", layout="wide")

st.title("🌱 SOPHEA : Dashboard d'Optimisation Frugale")
st.markdown("""
**Système d'Optimisation de Portefeuille à Haute Efficacité Algorithmique.**
Ce tableau de bord démontre notre approche d'optimisation sous contrainte de sobriété numérique.
""")

st.sidebar.header("⚙️ Paramètres de Frugalité")

st.sidebar.markdown("""
**Le concept du "Juste Assez" :**
Est-il nécessaire d'exiger une précision absolue si une décision d'investissement requiert moins de finesse ?
""")

# Ajout d'un toggle pour activer ou non l'IA
ia_active = st.sidebar.toggle("🤖 Activer l'IA (Méta-apprentissage)")

if ia_active:
    st.sidebar.success("Mode IA activé. Le système prévoira le nombre de simulations optimales en fonction de la volatilité.")
    # On entraîne le modèle à la volée (ou on le récupère)
    modele_ia, score_test = entrainer_modele_ia()
    st.sidebar.info(f"Modèle supervisé entraîné avec succès (Précision R²: {score_test:.2f}).")
    niveau_precision = None 
else:
    # L'utilisateur choisit la complexité manuellement si l'IA est désactivée
    niveau_precision = st.sidebar.select_slider(
        "Sélectionnez le niveau de précision requis :",
        options=["Faible (10^-2)", "Moyenne (10^-4)", "Haute (10^-6)"],
        value="Faible (10^-2)"
    )
    st.sidebar.info("💡 Remarque : Augmenter la précision augmente de manière exponentielle le nombre de simulations Quasi-Monte Carlo, et donc l'empreinte carbone du calcul.")

tab_esg, tab_perso = st.tabs(["🌱 Demo ESG (Données Statiques)", "📈 Portefeuille Personnalisé (Données Réelles)"])

# Démonstration avec un univers de fonds ESG qualitatifs, en utilisant des données statiques pour limiter l'empreinte réseau
with tab_esg:
    actifs_esg, rendements_esg, cov_esg = collecter_small_data_esg()

    col1, col2 = st.columns([1, 2])

    # Affichage de l'univers ESG et des rendements attendus
    with col1:
        st.subheader("📊 Approche Small Data")
        st.write("Univers restreint aux fonds ESG qualitatifs :")
        df_actifs = pd.DataFrame({"Fonds ESG": actifs_esg, "Rendement Espéré": rendements_esg})
        st.dataframe(df_actifs.style.format({"Rendement Espéré": "{:.1%}"}))

    # Optimisation du portefeuille ESG avec une approche frugale, en limitant les simulations selon le niveau de précision choisi
    with col2:
        st.subheader("🚀 Résultats de l'Optimisation")
        if st.button("Lancer l'optimisation (ESG)", type="primary"):
            with st.spinner("Calcul QMC en cours..."):
                
                # Boucle avec l'IA active
                if ia_active:
                    volatilite_moyenne = np.mean(np.sqrt(np.diag(cov_esg)))
                    prediction_ia = modele_ia.predict([[volatilite_moyenne]])[0]
                    niveau_precision = int(prediction_ia)
                    st.write(f"🧠 *L'IA a détecté une volatilité de {volatilite_moyenne:.2%} et a estimé le besoin de juste-assez à {niveau_precision} simulations cibles.*")
                
                resultats, temps_exec = optimiser_markowitz_frugal(rendements_esg, cov_esg, niveau_precision)
                
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Temps d'exécution", f"{temps_exec:.5f} s", delta="Green")
                kpi2.metric("Simulations générées", f"{resultats['simulations_realisees']}")
                kpi3.metric("Ratio de Sharpe", f"{resultats['ratio_sharpe']:.2f}")
                
                df_poids = pd.DataFrame({"Fonds": actifs_esg, "Allocation (%)": resultats["poids_optimaux"] * 100})
                st.bar_chart(df_poids.set_index("Fonds"))
                st.success("Optimisation terminée avec succès ! La démarche analytique a permis d'éviter les calculs stochastiques lourds classiques.")

# Deuxième onglet pour permettre à l'utilisateur de créer son propre univers d'investissement en entrant des tickers réels, 
# avec une récupération de données frugale (Small Data) et une optimisation basée sur ces données réelles
with tab_perso:
    st.subheader("🔎 Créez votre univers d'investissement frugal")
    st.write("Sélectionnez de vraies actions. Les données récupérées sont mensuelles sur 2 ans (Small Data) pour limiter l'empreinte réseau.")
    
    # L'utilisateur tape ses tickers
    tickers_input = st.text_input("Entrez les symboles Yahoo Finance (séparés par des espaces, ex: AAPL MSFT LVMUY GOOG)", "AAPL MSFT GOOG")
    tickers_list = [t.upper() for t in tickers_input.split() if t]

    # Validation pour s'assurer que l'utilisateur a entré au moins 2 tickers, sinon on affiche un message d'avertissement
    if len(tickers_list) < 2:
        st.warning("Veuillez entrer au moins 2 tickers pour optimiser un portefeuille.")
    else:
        col3, col4 = st.columns([1, 2])
        
        # Affichage des tickers sélectionnés et récupération des données réelles de manière frugale, 
        # avec une optimisation basée sur ces données réelles
        with col3:
            st.info(f"Actifs sélectionnés : {', '.join(tickers_list)}")
            # On récupère les données réelles
            actifs_reels, rendements_reels, cov_reels = collecter_donnees_reelles(tickers_list)
            
            if len(actifs_reels) > 0:
                df_reels = pd.DataFrame({"Actif": actifs_reels, "Rendement Historique (Ann.)": rendements_reels})
                st.dataframe(df_reels.style.format({"Rendement Historique (Ann.)": "{:.1%}"}))

        # Optimisation du portefeuille basé sur les données réelles, 
        # avec une approche frugale pour limiter les simulations selon le niveau de précision choisi
        with col4:
            if st.button("Optimiser mon portefeuille", type="primary"):
                with st.spinner("Analyse et optimisation en cours..."):
                    if len(actifs_reels) > 0:
                        
                        # Boucle avec l'IA active
                        if ia_active:
                            volatilite_moyenne_reelle = np.mean(np.sqrt(np.diag(cov_reels)))
                            prediction_ia_reelle = modele_ia.predict([[volatilite_moyenne_reelle]])[0]
                            niveau_precision = int(prediction_ia_reelle)
                            st.write(f"🧠 *L'IA a détecté une volatilité marché de {volatilite_moyenne_reelle:.2%} et a estimé le besoin de Juste-Assez à {niveau_precision} simulations cibles.*")
                        
                        resultats_reels, temps_reel = optimiser_markowitz_frugal(rendements_reels, cov_reels, niveau_precision)
                        
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Temps d'exécution", f"{temps_reel:.5f} s")
                        k2.metric("Simulations", f"{resultats_reels['simulations_realisees']}")
                        k3.metric("Ratio de Sharpe", f"{resultats_reels['ratio_sharpe']:.2f}")
                        
                        df_poids_reels = pd.DataFrame({"Actif": actifs_reels, "Allocation (%)": resultats_reels["poids_optimaux"] * 100})
                        st.bar_chart(df_poids_reels.set_index("Actif"))
                        st.success("Allocation calculée sur données réelles avec succès !")
                    else:
                        st.error("Erreur lors de la récupération des données. Vérifiez les tickers.")