# Projet : Optimisation de Portefeuille ESG avec Markowitz et QMC
# Partie Frontend - dashboard.py

# Importation des bibliothèques nécessaires pour le dashboard
import streamlit as st
import pandas as pd
import numpy as np

# Importation des fonctions depuis notre backend (main.py)
from main import collecter_small_data_esg, optimiser_markowitz_frugal, collecter_donnees_reelles, entrainer_modele_ia

st.set_page_config(page_title="SOPHEA - Green Finance", page_icon="🌱", layout="wide")

st.title("🌱 SOPHEA: Frugal Optimization Dashboard")
st.markdown("""
**High-Efficiency Algorithmic Portfolio Optimization System.**
This dashboard demonstrates our optimization approach under numerical sobriety constraints.
""")

st.sidebar.header("⚙️ Frugality Settings")

st.sidebar.markdown("""
**The "Just Enough" Concept:**
Is it necessary to demand absolute precision if an investment decision requires less finesse?
""")

# Ajout d'un toggle pour activer ou non l'IA
ia_active = st.sidebar.toggle("🤖 Enable AI (Meta-learning)")

if ia_active:
    st.sidebar.success("AI mode enabled. The system will predict the optimal number of simulations based on volatility.")
    # On entraîne le modèle à la volée (ou on le récupère)
    modele_ia, score_test = entrainer_modele_ia()
    st.sidebar.info(f"Supervised model trained successfully (R² Accuracy: {score_test:.2f}).")
    niveau_precision = None 
else:
    # L'utilisateur choisit la complexité manuellement si l'IA est désactivée
    niveau_precision = st.sidebar.select_slider(
        "Select the required precision level:",
        options=["Low (10^-2)", "Medium (10^-4)", "High (10^-6)"],
        value="Low (10^-2)"
    )
    st.sidebar.info("💡 Note: Increasing precision exponentially increases the number of Quasi-Monte Carlo simulations, and therefore the carbon footprint of the calculation.")

tab_esg, tab_perso = st.tabs(["🌱 ESG Demo (Static Data)", "📈 Custom Portfolio (Real Data)"])

# Démonstration avec un univers de fonds ESG qualitatifs, en utilisant des données statiques pour limiter l'empreinte réseau
with tab_esg:
    actifs_esg, rendements_esg, cov_esg = collecter_small_data_esg()

    col1, col2 = st.columns([1, 2])

    # Affichage de l'univers ESG et des rendements attendus
    with col1:
        st.subheader("📊 Small Data Approach")
        st.write("Universe restricted to qualitative ESG funds:")
        df_actifs = pd.DataFrame({"ESG Fund": actifs_esg, "Expected Return": rendements_esg})
        st.dataframe(df_actifs.style.format({"Expected Return": "{:.1%}"}))

    # Optimisation du portefeuille ESG avec une approche frugale, en limitant les simulations selon le niveau de précision choisi
    with col2:
        st.subheader("🚀 Optimization Results")
        if st.button("Run optimization (ESG)", type="primary"):
            with st.spinner("QMC calculation in progress..."):
                
                # Boucle avec l'IA active
                if ia_active:
                    volatilite_moyenne = np.mean(np.sqrt(np.diag(cov_esg)))
                    prediction_ia = modele_ia.predict([[volatilite_moyenne]])[0]
                    niveau_precision = int(prediction_ia)
                    st.write(f"🧠 *The AI detected a volatility of {volatilite_moyenne:.2%} and estimated the just-enough need at {niveau_precision} target simulations.*")
                
                resultats, temps_exec = optimiser_markowitz_frugal(rendements_esg, cov_esg, niveau_precision)
                
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Execution time", f"{temps_exec:.5f} s", delta="Green")
                kpi2.metric("Simulations generated", f"{resultats['simulations_realisees']}")
                kpi3.metric("Sharpe Ratio", f"{resultats['ratio_sharpe']:.2f}")
                
                df_poids = pd.DataFrame({"Fund": actifs_esg, "Allocation (%)": resultats["poids_optimaux"] * 100})
                st.bar_chart(df_poids.set_index("Fund"))
                st.success("Optimization successfully completed! The analytical approach avoided the usual heavy stochastic calculations.")

# Deuxième onglet pour permettre à l'utilisateur de créer son propre univers d'investissement en entrant des tickers réels, 
# avec une récupération de données frugale (Small Data) et une optimisation basée sur ces données réelles
with tab_perso:
    st.subheader("🔎 Create your frugal investment universe")
    st.write("Select real stocks. The retrieved data is monthly over 2 years (Small Data) to limit the network footprint.")
    
    # L'utilisateur tape ses tickers
    tickers_input = st.text_input("Enter Yahoo Finance symbols (separated by spaces, e.g.: AAPL MSFT LVMUY GOOG)", "AAPL MSFT GOOG")
    tickers_list = [t.upper() for t in tickers_input.split() if t]

    # Validation pour s'assurer que l'utilisateur a entré au moins 2 tickers, sinon on affiche un message d'avertissement
    if len(tickers_list) < 2:
        st.warning("Please enter at least 2 tickers to optimize a portfolio.")
    else:
        col3, col4 = st.columns([1, 2])
        
        # Affichage des tickers sélectionnés et récupération des données réelles de manière frugale, 
        # avec une optimisation basée sur ces données réelles
        with col3:
            st.info(f"Selected assets: {', '.join(tickers_list)}")
            # On récupère les données réelles
            actifs_reels, rendements_reels, cov_reels = collecter_donnees_reelles(tickers_list)
            
            if len(actifs_reels) > 0:
                df_reels = pd.DataFrame({"Asset": actifs_reels, "Historical Return (Ann.)": rendements_reels})
                st.dataframe(df_reels.style.format({"Historical Return (Ann.)": "{:.1%}"}))

        # Optimisation du portefeuille basé sur les données réelles, 
        # avec une approche frugale pour limiter les simulations selon le niveau de précision choisi
        with col4:
            if st.button("Optimize my portfolio", type="primary"):
                with st.spinner("Analysis and optimization in progress..."):
                    if len(actifs_reels) > 0:
                        
                        # Boucle avec l'IA active
                        if ia_active:
                            volatilite_moyenne_reelle = np.mean(np.sqrt(np.diag(cov_reels)))
                            prediction_ia_reelle = modele_ia.predict([[volatilite_moyenne_reelle]])[0]
                            niveau_precision = int(prediction_ia_reelle)
                            st.write(f"🧠 *The AI detected a market volatility of {volatilite_moyenne_reelle:.2%} and estimated the Just-Enough need at {niveau_precision} target simulations.*")
                        
                        resultats_reels, temps_reel = optimiser_markowitz_frugal(rendements_reels, cov_reels, niveau_precision)
                        
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Execution time", f"{temps_reel:.5f} s")
                        k2.metric("Simulations", f"{resultats_reels['simulations_realisees']}")
                        k3.metric("Sharpe Ratio", f"{resultats_reels['ratio_sharpe']:.2f}")
                        
                        df_poids_reels = pd.DataFrame({"Asset": actifs_reels, "Allocation (%)": resultats_reels["poids_optimaux"] * 100})
                        st.bar_chart(df_poids_reels.set_index("Asset"))
                        st.success("Allocation calculated on real data successfully!")
                    else:
                        st.error("Error retrieving data. Check the tickers.")
