import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pickle
import io
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier,
                              BaggingClassifier)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             accuracy_score)
from sklearn.dummy import DummyClassifier, DummyRegressor

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TP2 – ML Ensemble Models",
    page_icon="DT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root variables ── */
:root {
  --orange:    #F97316;
  --orange-lt: #FED7AA;
  --orange-dk: #C2410C;
  --blue:      #1D4ED8;
  --blue-lt:   #DBEAFE;
  --blue-dk:   #1E3A8A;
  --white:     #FFFFFF;
  --gray-50:   #F9FAFB;
  --gray-100:  #F3F4F6;
  --gray-200:  #E5E7EB;
  --gray-600:  #4B5563;
  --gray-800:  #1F2937;
}

/* ── Global font ── */
html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif;
  color: var(--gray-800);
}
code, pre, .stCode {
  font-family: 'JetBrains Mono', monospace !important;
}

/* ── Remove default Streamlit padding on top ── */
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--blue-dk) 0%, var(--blue) 60%, #2563EB 100%);
  color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label {
  background: rgba(255,255,255,0.08);
  border-radius: 10px;
  padding: 10px 14px;
  margin: 4px 0;
  display: block;
  cursor: pointer;
  transition: background 0.2s;
  font-weight: 500;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(255,255,255,0.18);
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
  color: white !important;
}

/* ── Hero banner ── */
.hero-banner {
  background: linear-gradient(135deg, var(--blue-dk) 0%, var(--blue) 50%, var(--orange) 100%);
  border-radius: 16px;
  padding: 2rem 2.5rem;
  color: white;
  margin-bottom: 1.5rem;
}
.hero-banner h1 { font-size: 2rem; font-weight: 800; margin: 0 0 .4rem; }
.hero-banner p  { font-size: 1rem; opacity: .85; margin: 0; }

/* ── Section title ── */
.section-title {
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--blue-dk);
  border-left: 5px solid var(--orange);
  padding-left: 12px;
  margin: 1.5rem 0 1rem;
}
.sub-title {
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--blue);
  margin: 1rem 0 .5rem;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
  flex: 1; min-width: 130px;
  background: var(--white);
  border: 1.5px solid var(--gray-200);
  border-top: 4px solid var(--orange);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
.metric-card .val { font-size: 1.5rem; font-weight: 700; color: var(--blue-dk); }
.metric-card .lbl { font-size: .78rem; color: var(--gray-600); margin-top: 2px; }

/* ── Info / Warning boxes ── */
.info-box {
  background: var(--blue-lt);
  border-left: 4px solid var(--blue);
  border-radius: 8px;
  padding: .8rem 1rem;
  margin: .8rem 0;
  font-size: .92rem;
}
.warn-box {
  background: var(--orange-lt);
  border-left: 4px solid var(--orange);
  border-radius: 8px;
  padding: .8rem 1rem;
  margin: .8rem 0;
  font-size: .92rem;
}

/* ── Table styling ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
thead tr { background: var(--blue-dk) !important; color: white !important; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(90deg, var(--blue) 0%, var(--blue-dk) 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  padding: .5rem 1.4rem !important;
  transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { gap: .5rem; }
[data-baseweb="tab"] {
  border-radius: 8px 8px 0 0 !important;
  font-weight: 600 !important;
}
[aria-selected="true"] {
  background: var(--orange) !important;
  color: white !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
  border: 1.5px solid var(--gray-200) !important;
  border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────────
def metric_cards(items):
    """items: list of (label, value) tuples"""
    cols_html = "".join(
        f'<div class="metric-card"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
        for l, v in items
    )
    st.markdown(f'<div class="metric-row">{cols_html}</div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def sub(title):
    st.markdown(f'<div class="sub-title">{title}</div>', unsafe_allow_html=True)

def info(msg):
    st.markdown(f'<div class="info-box">ℹ️ {msg}</div>', unsafe_allow_html=True)

def warn(msg):
    st.markdown(f'<div class="warn-box">⚠️ {msg}</div>', unsafe_allow_html=True)

def styled_fig():
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F9FAFB')
    ax.tick_params(colors='#4B5563')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E7EB')
    ax.spines['bottom'].set_color('#E5E7EB')
    return fig, ax

ORANGE  = "#F97316"
BLUE    = "#1D4ED8"
BLUE_DK = "#1E3A8A"

# ─── Load Auto-MPG ──────────────────────────────────────────────────────────────
@st.cache_data
def load_auto_mpg():
    df = pd.read_csv("auto-mpg.csv")  # Changé de chemin absolu
    df = df.drop(columns=["car name"], errors="ignore")
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ─── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## DT TP2 – ML Models")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "📊 Partie 1 – Classification", "📈 Partie 2 – Régression", "🔬 Partie 3 – Dataset Libre"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Données utilisées**")
    st.markdown("• `auto-mpg.csv` (Parties 1 & 2)")
    st.markdown("• Dataset libre (Partie 3)")
    st.markdown("---")
    st.markdown("*Licence MTQ S6 – 2025-2026*")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE : ACCUEIL
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.markdown("""
    <div class="hero-banner">
      <h1>INFO4212: TP2 – Implémentation & Déploiement des Modèles ML</h1>
      <p> Par TCHUENTEU GUETCHUENG DAVID - 20U2891 </p>
      <p>Sous la supervision de Stéphane C. K. TÉKOUABOU (PhD & Ing.)</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card" style="border-top-color:#1D4ED8">
          <div class="val">📊</div>
          <div class="lbl" style="font-size:.95rem;font-weight:600;color:#1E3A8A">Partie 1 – Classification</div>
          <p style="font-size:.82rem;margin-top:.5rem">Méthodes d'ensemble (Bagging, Random Forest, Boosting) sur auto-mpg</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card" style="border-top-color:#F97316">
          <div class="val">📈</div>
          <div class="lbl" style="font-size:.95rem;font-weight:600;color:#C2410C">Partie 2 – Régression</div>
          <p style="font-size:.82rem;margin-top:.5rem">KNN Regressor, arbres, Random Forest sur auto-mpg (prédiction mpg)</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card" style="border-top-color:#0EA5E9">
          <div class="val">🔬</div>
          <div class="lbl" style="font-size:.95rem;font-weight:600;color:#0369A1">Partie 3 – Dataset Libre</div>
          <p style="font-size:.82rem;margin-top:.5rem">Expérimentation sur un nouveau jeu de données (Census / UCI)</p>
        </div>""", unsafe_allow_html=True)

    section("Objectifs du TP")
    st.markdown("""
    Ce TP couvre l'ensemble du pipeline d'un projet ML : depuis l'exploration des données jusqu'au déploiement.

    **Compétences développées :**
    - Consolidation des algorithmes ML (KNN, Arbres, Forêts Aléatoires, Boosting)
    - Méthodes d'ensemble (Bagging & Boosting)
    - Validation croisée et sélection d'hyperparamètres (GridSearchCV)
    - Évaluation avancée : ROC, AUC, MAE, MSE, R²
    - Déploiement des modèles via Streamlit

    **Bibliothèques :** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pickle`, `streamlit`
    """)

    section("Jeu de données – Auto MPG")
    df = load_auto_mpg()
    metric_cards([
        ("Instances", len(df)),
        ("Caractéristiques", len(df.columns)-1),
        ("Cible", "mpg"),
        ("Valeurs manquantes", "0"),
    ])
    st.dataframe(df.head(8), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE : PARTIE 1 – CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Partie 1 – Classification":
    st.markdown("""
    <div class="hero-banner">
      <h1>📊 Partie 1 – Classification avec Méthodes d'Ensemble</h1>
      <p>Binarisation de la consommation (mpg) · Bagging · Random Forest · Gradient Boosting</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_auto_mpg()

    # Create binary target: 1 if mpg >= median, 0 otherwise
    median_mpg = df["mpg"].median()
    df["target"] = (df["mpg"] >= median_mpg).astype(int)

    info(f"Cible binaire créée : `1` si mpg ≥ {median_mpg:.1f} (efficace), `0` sinon (gourmand)")

    tabs = st.tabs(["A · Exploration", "B · Validation Croisée (CART)", "C · Bagging & Random Forest", "D · Boosting", "E · Importance des Variables", "F · Courbes ROC"])

    # ── TAB A : Exploration ──────────────────────────────────────────────────────
    with tabs[0]:
        section("A – Description et Visualisation des Données")
        col1, col2 = st.columns(2)
        with col1:
            sub("Dimensions et types")
            metric_cards([
                ("Classes", 2),
                ("Instances", len(df)),
                ("Classe 0 (gourmand)", int((df["target"]==0).sum())),
                ("Classe 1 (efficace)", int((df["target"]==1).sum())),
            ])
            st.dataframe(df.describe().round(2), use_container_width=True)
        with col2:
            sub("Distribution de la cible")
            fig, ax = styled_fig()
            counts = df["target"].value_counts()
            ax.bar(["Classe 0\n(gourmand)", "Classe 1\n(efficace)"], counts.values,
                   color=[ORANGE, BLUE], edgecolor="white", linewidth=1.5, width=.5)
            ax.set_ylabel("Nombre d'instances")
            ax.set_title("Distribution des classes", fontweight='bold')
            for i, v in enumerate(counts.values):
                ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        sub("Pairplot des variables numériques")
        features = ["cylinders","displacement","horsepower","weight","acceleration","model year"]
        fig2 = plt.figure(figsize=(10, 8))
        pair_df = df[features + ["target"]].copy()
        pair_df["target"] = pair_df["target"].map({0:"gourmand", 1:"efficace"})
        pg = sns.pairplot(pair_df, hue="target", palette={"gourmand": ORANGE, "efficace": BLUE},
                          plot_kws={"alpha": 0.5, "s": 20}, diag_kind="kde")
        pg.fig.suptitle("Nuages de points croisés", y=1.01, fontweight='bold')
        st.pyplot(pg.fig)
        plt.close()

        info("Les variables `weight`, `displacement` et `cylinders` sont fortement corrélées négativement avec mpg. `model year` montre une tendance positive (les voitures récentes sont plus économes).")

        sub("Corrélation entre variables")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        corr = df[features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, ax=ax3, square=True, linewidths=.5)
        ax3.set_title("Matrice de corrélation", fontweight='bold')
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # ── TAB B : Validation croisée CART ─────────────────────────────────────────
    with tabs[1]:
        section("B – Validation Croisée sur Arbre de Décision (CART)")

        features = ["cylinders","displacement","horsepower","weight","acceleration","model year","origin"]
        X = df[features].values
        y = df["target"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        col1, col2 = st.columns(2)
        with col1:
            sub("1. Classifieur constant (baseline)")
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_train, y_train)
            dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
            dummy_class = int(dummy.predict([[0]*7][0]))
            metric_cards([("Accuracy baseline", f"{dummy_acc:.3f}"), ("Classe prédite", str(dummy_class))])
            info("Le classifieur constant prédit toujours la classe majoritaire. C'est le référentiel à battre.")

        with col2:
            sub("2. Arbre de décision simple (max_depth=4)")
            dt_simple = DecisionTreeClassifier(max_depth=4, random_state=42)
            dt_simple.fit(X_train, y_train)
            dt_acc = accuracy_score(y_test, dt_simple.predict(X_test))
            metric_cards([("Accuracy arbre", f"{dt_acc:.3f}"), ("Gain vs baseline", f"+{dt_acc-dummy_acc:.3f}")])

        sub("3. GridSearchCV – Sélection de la profondeur optimale")
        param_grid = {"max_depth": list(range(1, 16)), "criterion": ["gini", "entropy"]}

        with st.spinner("GridSearchCV en cours..."):
            gs = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5,
                              scoring="accuracy", n_jobs=-1)
            gs.fit(X_train, y_train)

        best_params = gs.best_params_
        best_dt = gs.best_estimator_
        best_acc = accuracy_score(y_test, best_dt.predict(X_test))

        metric_cards([
            ("Meilleure profondeur", best_params["max_depth"]),
            ("Meilleur critère", best_params["criterion"]),
            ("Accuracy optimisée", f"{best_acc:.3f}"),
        ])

        sub("Erreur de validation croisée vs profondeur")
        fig, ax = styled_fig()
        results = gs.cv_results_
        for crit, col in [("gini", ORANGE), ("entropy", BLUE)]:
            mask_c = [p["criterion"] == crit for p in gs.cv_results_["params"]]
            depths = [gs.cv_results_["params"][i]["max_depth"] for i in range(len(mask_c)) if mask_c[i]]
            scores = [gs.cv_results_["mean_test_score"][i] for i in range(len(mask_c)) if mask_c[i]]
            pairs = sorted(zip(depths, scores))
            depths_s, scores_s = zip(*pairs)
            ax.plot(depths_s, scores_s, marker='o', label=crit, color=col, linewidth=2)
        ax.axvline(best_params["max_depth"], linestyle='--', color='gray', alpha=.6, label=f"Optimal: {best_params['max_depth']}")
        ax.set_xlabel("Profondeur de l'arbre")
        ax.set_ylabel("Accuracy (CV 5-fold)")
        ax.set_title("Validation croisée vs profondeur", fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        plt.close()

        sub("Matrice de confusion (arbre optimisé)")
        cm = confusion_matrix(y_test, best_dt.predict(X_test))
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=["Prédit 0","Prédit 1"], yticklabels=["Réel 0","Réel 1"])
        ax2.set_title("Matrice de confusion", fontweight='bold')
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── TAB C : Bagging & Random Forest ─────────────────────────────────────────
    with tabs[2]:
        section("C – Bagging et Random Forest")
        features = ["cylinders","displacement","horsepower","weight","acceleration","model year","origin"]
        X = df[features].values
        y = df["target"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        col1, col2 = st.columns(2)
        with col1:
            sub("Bagging (n_estimators = max_features = None)")
            B_range = [10, 50, 100, 200]
            bag_scores = []
            for B in B_range:
                bag = RandomForestClassifier(n_estimators=B, max_features=None, random_state=42, n_jobs=-1)
                bag.fit(X_train, y_train)
                bag_scores.append(accuracy_score(y_test, bag.predict(X_test)))

            fig, ax = styled_fig()
            ax.plot(B_range, bag_scores, marker='s', color=ORANGE, linewidth=2.5)
            ax.set_xlabel("Nombre d'arbres B")
            ax.set_ylabel("Accuracy (test)")
            ax.set_title("Bagging : performance vs B", fontweight='bold')
            st.pyplot(fig)
            plt.close()
            info("Avec max_features=None (tous les features), RandomForestClassifier réalise du Bagging pur.")

        with col2:
            sub("Random Forest optimisé (GridSearch sur p)")
            B_fixed = 100
            p_range = [1, 2, 3, 4, 5, 6, 7]
            gs_rf = GridSearchCV(
                RandomForestClassifier(n_estimators=B_fixed, oob_score=True, random_state=42, n_jobs=-1),
                {"max_features": p_range}, cv=5, scoring="accuracy", n_jobs=-1)
            with st.spinner("Optimisation RF..."):
                gs_rf.fit(X_train, y_train)

            best_rf = gs_rf.best_estimator_
            rf_acc  = accuracy_score(y_test, best_rf.predict(X_test))
            oob_err = 1 - best_rf.oob_score_

            metric_cards([
                ("max_features optimal", gs_rf.best_params_["max_features"]),
                ("Accuracy test", f"{rf_acc:.3f}"),
                ("Erreur OOB", f"{oob_err:.3f}"),
            ])

            fig2, ax2 = styled_fig()
            cv_means = gs_rf.cv_results_["mean_test_score"]
            ax2.plot(p_range, cv_means, marker='o', color=BLUE, linewidth=2.5)
            ax2.set_xlabel("max_features (p)")
            ax2.set_ylabel("Accuracy CV")
            ax2.set_title("RF : sélection de p", fontweight='bold')
            ax2.axvline(gs_rf.best_params_["max_features"], color='gray', linestyle='--', alpha=.6)
            st.pyplot(fig2)
            plt.close()

        sub("Comparaison Bagging vs Random Forest")
        best_bag = RandomForestClassifier(n_estimators=100, max_features=None, random_state=42, n_jobs=-1)
        best_bag.fit(X_train, y_train)
        metric_cards([
            ("Bagging Accuracy", f"{accuracy_score(y_test, best_bag.predict(X_test)):.3f}"),
            ("RF Accuracy", f"{rf_acc:.3f}"),
        ])

        # Save RF model
        with open("/tmp/auto-mpg_rf.pkl", "wb") as f:
            pickle.dump(best_rf, f)
        info("Modèle Random Forest optimisé sauvegardé.")

    # ── TAB D : Boosting ─────────────────────────────────────────────────────────
    with tabs[3]:
        section("D – Gradient Boosting")
        features = ["cylinders","displacement","horsepower","weight","acceleration","model year","origin"]
        X = df[features].values
        y = df["target"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        info("Paramètres cruciaux du Gradient Boosting : `n_estimators` (B), `learning_rate` (λ), `max_depth` (p), `subsample`. AdaBoost ↔ learning_rate=1, max_depth=1, loss='exponential'.")

        with st.spinner("Entraînement Gradient Boosting (early stopping)..."):
            gb = GradientBoostingClassifier(
                n_estimators=500, learning_rate=0.05,
                max_depth=3, subsample=0.8,
                validation_fraction=0.1, n_iter_no_change=20,
                random_state=42
            )
            gb.fit(X_train, y_train)

        gb_acc = accuracy_score(y_test, gb.predict(X_test))
        metric_cards([
            ("Estimators utilisés", gb.n_estimators_),
            ("Accuracy test", f"{gb_acc:.3f}"),
            ("learning_rate", "0.05"),
            ("max_depth", "3"),
        ])

        sub("Évolution de la déviance (train vs validation)")
        fig, ax = styled_fig()
        train_score = gb.train_score_
        ax.plot(range(1, len(train_score)+1), train_score, color=BLUE, label='Train', linewidth=1.5)
        ax.axvline(gb.n_estimators_, linestyle='--', color=ORANGE, label=f'Early stop @ {gb.n_estimators_}')
        ax.set_xlabel("Nombre d'arbres")
        ax.set_ylabel("Déviance")
        ax.set_title("Convergence Gradient Boosting", fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        plt.close()

        sub("Comparaison globale des classifieurs")
        # Arbre optimisé
        gs2 = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           {"max_depth": list(range(1,16))}, cv=5, n_jobs=-1)
        gs2.fit(X_train, y_train)
        dt_acc = accuracy_score(y_test, gs2.best_estimator_.predict(X_test))

        gs_rf2 = GridSearchCV(
            RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42, n_jobs=-1),
            {"max_features": [1,2,3,4,5,6,7]}, cv=5, n_jobs=-1)
        gs_rf2.fit(X_train, y_train)
        rf_acc2 = accuracy_score(y_test, gs_rf2.best_estimator_.predict(X_test))

        fig2, ax2 = styled_fig()
        models = ["Arbre CART", "Random Forest", "Gradient Boosting"]
        accs   = [dt_acc, rf_acc2, gb_acc]
        colors = [ORANGE, BLUE, "#0EA5E9"]
        bars = ax2.bar(models, accs, color=colors, width=0.4, edgecolor='white', linewidth=1.5)
        ax2.set_ylim(0.7, 1.0)
        ax2.set_ylabel("Accuracy (test)")
        ax2.set_title("Comparaison des classifieurs optimisés", fontweight='bold')
        for bar, acc in zip(bars, accs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .003,
                     f"{acc:.3f}", ha='center', fontweight='bold')
        st.pyplot(fig2)
        plt.close()

        # Save best model
        best_model = gb if gb_acc == max(accs) else gs_rf2.best_estimator_
        with open("/tmp/auto-mpg.pkl", "wb") as f:
            pickle.dump(best_model, f)
        info(f"Meilleur modèle de classification sauvegardé : {'Gradient Boosting' if gb_acc==max(accs) else 'Random Forest'}")

        buf = io.BytesIO()
        pickle.dump(best_model, buf)
        buf.seek(0)
        st.download_button("⬇️ Télécharger auto-mpg.pkl", buf, file_name="auto-mpg.pkl")

    # ── TAB E : Importance des variables ────────────────────────────────────────
    with tabs[4]:
        section("E – Sélection et Importance des Variables")
        features = ["cylinders","displacement","horsepower","weight","acceleration","model year","origin"]
        X = df[features].values
        y = df["target"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        models_imp = {
            "Arbre CART": DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.patch.set_facecolor('#FFFFFF')
        colors_list = [ORANGE, BLUE, "#0EA5E9"]

        for ax, (name, model), color in zip(axes, models_imp.items(), colors_list):
            model.fit(X_train, y_train)
            importances = model.feature_importances_
            idx = np.argsort(importances)
            ax.barh([features[i] for i in idx], importances[idx], color=color, alpha=.85)
            ax.set_title(name, fontweight='bold')
            ax.set_xlabel("Importance")
            ax.set_facecolor('#F9FAFB')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.suptitle("Importance des variables par classifieur", fontweight='bold', fontsize=14)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        info("Les variables `displacement`, `weight` et `horsepower` sont les plus importantes dans tous les classifieurs. `origin` et `cylinders` ont une importance secondaire.")

    # ── TAB F : Courbes ROC ──────────────────────────────────────────────────────
    with tabs[5]:
        section("F – Évaluation : Courbes ROC et AUC")
        features = ["cylinders","displacement","horsepower","weight","acceleration","model year","origin"]
        X = df[features].values
        y = df["target"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        roc_models = {
            "Arbre CART": DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        roc_colors = [ORANGE, BLUE, "#0EA5E9"]

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F9FAFB')

        auc_vals = {}
        for (name, model), color in zip(roc_models.items(), roc_colors):
            model.fit(X_train, y_train)
            scores = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, scores)
            roc_auc = auc(fpr, tpr)
            auc_vals[name] = roc_auc
            ax.plot(fpr, tpr, color=color, linewidth=2.5,
                    label=f"{name} (AUC = {roc_auc:.3f})")

        ax.plot([0,1],[0,1], 'k--', alpha=.4, label="Aléatoire")
        ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=11)
        ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=11)
        ax.set_title("Courbes ROC – Classifieurs optimisés", fontweight='bold', fontsize=13)
        ax.legend(loc="lower right")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

        metric_cards([(name, f"{val:.3f}") for name, val in auc_vals.items()])
        best_clf = max(auc_vals, key=auc_vals.get)
        info(f"Le meilleur classifieur selon l'AUC est **{best_clf}** avec AUC = {auc_vals[best_clf]:.3f}")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE : PARTIE 2 – RÉGRESSION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Partie 2 – Régression":
    st.markdown("""
    <div class="hero-banner">
      <h1>📈 Partie 2 – Régression sur Auto MPG</h1>
      <p>Prédiction de la consommation (mpg) · KNN · Arbres · Random Forest · Comparaison</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_auto_mpg()
    features = ["cylinders","displacement","horsepower","weight","acceleration","model year","origin"]
    X = df[features].values
    y = df["mpg"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tabs = st.tabs(["📂 Exploration", "🔵 KNN Régression", "🌳 Arbres & Forêts", "⚡ Boosting Régression", "📊 Comparaison & Déploiement"])

    # ── TAB 0 : Exploration ──────────────────────────────────────────────────────
    with tabs[0]:
        section("Description des Données")
        metric_cards([
            ("Instances", len(df)),
            ("Features", len(features)),
            ("mpg moyen", f"{df['mpg'].mean():.1f}"),
            ("mpg min", f"{df['mpg'].min():.1f}"),
            ("mpg max", f"{df['mpg'].max():.1f}"),
        ])

        col1, col2 = st.columns(2)
        with col1:
            sub("Distribution de mpg")
            fig, ax = styled_fig()
            ax.hist(df["mpg"], bins=25, color=ORANGE, edgecolor='white', alpha=.85)
            ax.axvline(df["mpg"].mean(), color=BLUE, linewidth=2, linestyle='--', label=f"Moyenne: {df['mpg'].mean():.1f}")
            ax.set_xlabel("mpg"); ax.set_ylabel("Fréquence")
            ax.set_title("Distribution de mpg", fontweight='bold')
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with col2:
            sub("Corrélation avec mpg")
            corr_mpg = df[features + ["mpg"]].corr()["mpg"].drop("mpg").sort_values()
            fig2, ax2 = styled_fig()
            colors_bar = [ORANGE if v < 0 else BLUE for v in corr_mpg.values]
            ax2.barh(corr_mpg.index, corr_mpg.values, color=colors_bar, alpha=.85)
            ax2.axvline(0, color='gray', linewidth=.8)
            ax2.set_title("Corrélation des features avec mpg", fontweight='bold')
            st.pyplot(fig2)
            plt.close()

        sub("Scatter plots clés")
        fig3, axes = plt.subplots(2, 3, figsize=(13, 7))
        fig3.patch.set_facecolor('#FFFFFF')
        for ax, feat in zip(axes.flatten(), features):
            ax.scatter(df[feat], df["mpg"], color=BLUE, alpha=.4, s=15)
            ax.set_xlabel(feat); ax.set_ylabel("mpg")
            ax.set_facecolor('#F9FAFB')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        fig3.suptitle("mpg en fonction de chaque variable", fontweight='bold', fontsize=13)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # ── TAB 1 : KNN ──────────────────────────────────────────────────────────────
    with tabs[1]:
        section("KNN Régression – Influence du paramètre k")

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        k_range = list(range(1, 31))
        mae_raw, mse_raw, r2_raw = [], [], []
        mae_sc,  mse_sc,  r2_sc  = [], [], []

        for k in k_range:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X_train, y_train)
            p = knn.predict(X_test)
            mae_raw.append(mean_absolute_error(y_test, p))
            mse_raw.append(mean_squared_error(y_test, p))
            r2_raw.append(r2_score(y_test, p))

            knn_sc = KNeighborsRegressor(n_neighbors=k)
            knn_sc.fit(X_train_sc, y_train)
            p_sc = knn_sc.predict(X_test_sc)
            mae_sc.append(mean_absolute_error(y_test, p_sc))
            mse_sc.append(mean_squared_error(y_test, p_sc))
            r2_sc.append(r2_score(y_test, p_sc))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor('#FFFFFF')
        metrics_plot = [
            ("MAE", mae_raw, mae_sc),
            ("MSE", mse_raw, mse_sc),
            ("R²",  r2_raw,  r2_sc),
        ]
        for ax, (name, raw, sc) in zip(axes, metrics_plot):
            ax.plot(k_range, raw, color=ORANGE, linewidth=2, marker='o', markersize=3, label="Sans normalisation")
            ax.plot(k_range, sc,  color=BLUE,   linewidth=2, marker='s', markersize=3, label="Normalisé")
            ax.set_xlabel("k"); ax.set_ylabel(name)
            ax.set_title(name, fontweight='bold')
            ax.set_facecolor('#F9FAFB')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(fontsize=8)
        fig.suptitle("KNN Régression : impact de k et de la normalisation", fontweight='bold', fontsize=13)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        best_k_sc = k_range[np.argmax(r2_sc)]
        best_knn  = KNeighborsRegressor(n_neighbors=best_k_sc)
        best_knn.fit(X_train_sc, y_train)
        preds = best_knn.predict(X_test_sc)

        metric_cards([
            ("k optimal", best_k_sc),
            ("MAE", f"{mean_absolute_error(y_test, preds):.3f}"),
            ("MSE", f"{mean_squared_error(y_test, preds):.3f}"),
            ("R²",  f"{r2_score(y_test, preds):.3f}"),
        ])

        info("La normalisation améliore sensiblement les performances du KNN, car il est basé sur les distances.")

        sub("Prédictions vs Valeurs réelles (KNN optimal)")
        fig2, ax2 = styled_fig()
        ax2.scatter(y_test, preds, color=BLUE, alpha=.6, s=25)
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax2.plot(lims, lims, 'r--', linewidth=1.5)
        ax2.set_xlabel("Valeurs réelles"); ax2.set_ylabel("Prédictions")
        ax2.set_title("Prédictions vs Réel (KNN)", fontweight='bold')
        st.pyplot(fig2)
        plt.close()

    # ── TAB 2 : Arbres & Forêts ──────────────────────────────────────────────────
    with tabs[2]:
        section("Arbres de Décision & Random Forest – Régression")
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        col1, col2 = st.columns(2)
        with col1:
            sub("Arbre de Décision – max_depth")
            depths = list(range(1, 16))
            r2_dt = []
            for d in depths:
                dt = DecisionTreeRegressor(max_depth=d, random_state=42)
                dt.fit(X_train, y_train)
                r2_dt.append(r2_score(y_test, dt.predict(X_test)))
            best_d = depths[np.argmax(r2_dt)]
            fig, ax = styled_fig()
            ax.plot(depths, r2_dt, marker='o', color=ORANGE, linewidth=2)
            ax.axvline(best_d, color='gray', linestyle='--', alpha=.6)
            ax.set_xlabel("Profondeur"); ax.set_ylabel("R²")
            ax.set_title("Arbre : R² vs profondeur", fontweight='bold')
            st.pyplot(fig)
            plt.close()
            metric_cards([("Profondeur optimale", best_d), ("R² arbre", f"{max(r2_dt):.3f}")])

        with col2:
            sub("Random Forest Régression")
            with st.spinner("GridSearch RF Régression..."):
                gs_rf = GridSearchCV(
                    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    {"max_features": [1, 2, 3, 4, 5]},
                    cv=5, scoring="r2", n_jobs=-1
                )
                gs_rf.fit(X_train, y_train)
            best_rf = gs_rf.best_estimator_
            rf_preds = best_rf.predict(X_test)
            metric_cards([
                ("max_features optimal", gs_rf.best_params_["max_features"]),
                ("R²", f"{r2_score(y_test, rf_preds):.3f}"),
                ("MAE", f"{mean_absolute_error(y_test, rf_preds):.3f}"),
            ])

            fig2, ax2 = styled_fig()
            ax2.scatter(y_test, rf_preds, color=BLUE, alpha=.6, s=25)
            lims = [min(y_test.min(), rf_preds.min()), max(y_test.max(), rf_preds.max())]
            ax2.plot(lims, lims, 'r--', linewidth=1.5)
            ax2.set_xlabel("Réel"); ax2.set_ylabel("Prédit")
            ax2.set_title("RF : Prédictions vs Réel", fontweight='bold')
            st.pyplot(fig2)
            plt.close()

    # ── TAB 3 : Boosting Régression ──────────────────────────────────────────────
    with tabs[3]:
        section("Gradient Boosting – Régression")
        with st.spinner("Entraînement GB Régression..."):
            gb = GradientBoostingRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=3,
                subsample=0.8, validation_fraction=0.1,
                n_iter_no_change=20, random_state=42
            )
            gb.fit(X_train, y_train)

        gb_preds = gb.predict(X_test)
        metric_cards([
            ("Estimators utilisés", gb.n_estimators_),
            ("MAE", f"{mean_absolute_error(y_test, gb_preds):.3f}"),
            ("MSE", f"{mean_squared_error(y_test, gb_preds):.3f}"),
            ("R²",  f"{r2_score(y_test, gb_preds):.3f}"),
        ])

        sub("Importance des variables")
        fig, ax = styled_fig()
        imp = gb.feature_importances_
        idx = np.argsort(imp)
        ax.barh([features[i] for i in idx], imp[idx], color=BLUE, alpha=.85)
        ax.set_title("Gradient Boosting : importance des variables", fontweight='bold')
        st.pyplot(fig)
        plt.close()

    # ── TAB 4 : Comparaison & Déploiement ───────────────────────────────────────
    with tabs[4]:
        section("Comparaison finale des modèles de régression")
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        all_models = {
            "KNN (k=7, normalisé)": (KNeighborsRegressor(n_neighbors=7), X_train_sc, X_test_sc),
            "Arbre CART": (DecisionTreeRegressor(max_depth=6, random_state=42), X_train, X_test),
            "Random Forest": (RandomForestRegressor(n_estimators=100, max_features=3, random_state=42), X_train, X_test),
            "Gradient Boosting": (GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42), X_train, X_test),
            "Ridge Regression": (Ridge(alpha=1.0), X_train_sc, X_test_sc),
        }

        results_rows = []
        for name, (model, Xtr, Xte) in all_models.items():
            model.fit(Xtr, y_train)
            p = model.predict(Xte)
            results_rows.append({
                "Modèle": name,
                "MAE": round(mean_absolute_error(y_test, p), 3),
                "MSE": round(mean_squared_error(y_test, p), 3),
                "R²":  round(r2_score(y_test, p), 3),
            })

        res_df = pd.DataFrame(results_rows).sort_values("R²", ascending=False).reset_index(drop=True)
        st.dataframe(res_df, use_container_width=True)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor('#FFFFFF')
        for ax, metric in zip(axes, ["MAE", "MSE", "R²"]):
            sorted_df = res_df.sort_values(metric, ascending=(metric != "R²"))
            ax.barh(sorted_df["Modèle"], sorted_df[metric],
                    color=[ORANGE, BLUE, "#0EA5E9", "#10B981", "#8B5CF6"][:len(sorted_df)], alpha=.85)
            ax.set_title(metric, fontweight='bold')
            ax.set_facecolor('#F9FAFB')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        fig.suptitle("Comparaison des modèles de régression", fontweight='bold', fontsize=13)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        best_name = res_df.iloc[0]["Modèle"]
        info(f"Meilleur modèle : **{best_name}** (R² = {res_df.iloc[0]['R²']})")

        # Save best model (Gradient Boosting)
        gb_final = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
        gb_final.fit(X_train, y_train)
        buf = io.BytesIO()
        pickle.dump({"model": gb_final, "scaler": None, "features": features}, buf)
        buf.seek(0)
        st.download_button("⬇️ Télécharger auto-mpg.pkl (régression)", buf, file_name="auto-mpg.pkl")

        sub("Prédiction interactive")
        col1, col2, col3 = st.columns(3)
        with col1:
            cyl  = st.slider("Cylinders", 3, 8, 4)
            disp = st.slider("Displacement", 70, 460, 150)
        with col2:
            hp   = st.slider("Horsepower", 45, 230, 100)
            wt   = st.slider("Weight", 1600, 5200, 2800)
        with col3:
            acc  = st.slider("Acceleration", 8.0, 25.0, 15.0)
            yr   = st.slider("Model Year", 70, 82, 76)
            ori  = st.selectbox("Origin", [1, 2, 3])

        if st.button("🔮 Prédire la consommation (mpg)"):
            inp = np.array([[cyl, disp, hp, wt, acc, yr, ori]])
            pred_mpg = gb_final.predict(inp)[0]
            st.markdown(f"""
            <div style="background:linear-gradient(90deg,{BLUE},{ORANGE});
                        padding:1.2rem 2rem;border-radius:12px;color:white;text-align:center;margin-top:1rem">
              <div style="font-size:2rem;font-weight:800">{pred_mpg:.1f} mpg</div>
              <div style="opacity:.85">Consommation estimée (Gradient Boosting)</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE : PARTIE 3 – DATASET LIBRE
# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════
# PAGE : PARTIE 3 – DATASET LIBRE (CENSUS 2015)
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Partie 3 – Dataset Libre":
    st.markdown("""
    <div class="hero-banner">
      <h1>🔬 Partie 3 – Dataset Census 2015</h1>
      <p>Analyse socio-démographique · Prédiction des revenus · Modèles avancés</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Chargement des données ──────────────────────────────────────────────
    @st.cache_data
    def load_census_data():
        """Charge et prépare les données du recensement"""
        df = pd.read_csv("census-data2015.csv")
        
        # Remplacer les chaînes vides par NaN
        df = df.replace('', np.nan)
        df = df.replace(' ', np.nan)
        
        # Colonnes numériques à convertir
        numeric_cols = ['TotalPop', 'Men', 'Women', 'Hispanic', 'White', 'Black', 
                        'Native', 'Asian', 'Pacific', 'Citizen', 'Income', 'IncomeErr',
                        'IncomePerCap', 'IncomePerCapErr', 'Poverty', 'ChildPoverty',
                        'Professional', 'Service', 'Office', 'Construction', 'Production',
                        'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp', 'WorkAtHome',
                        'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork', 
                        'SelfEmployed', 'FamilyWork', 'Unemployment']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Supprimer les lignes avec trop de valeurs manquantes
        df.dropna(thresh=len(df.columns)-5, inplace=True)
        
        # Créer une cible binaire pour la classification
        median_income = df['Income'].median()
        df['target_high_income'] = (df['Income'] >= median_income).astype(int)
        
        # Créer une cible de régression (log pour normaliser)
        df['target_income_log'] = np.log1p(df['Income'])
        
        info(f"Dataset chargé : {len(df)} tracts · {len(df.columns)} variables")
        info(f"Revenu médian : ${median_income:,.0f} · Classes équilibrées : {df['target_high_income'].value_counts().to_dict()}")
        
        return df

    df = load_census_data()
    
    # Features sélectionnées pour la modélisation
    base_features = ['TotalPop', 'Men', 'Women', 'Hispanic', 'White', 'Black', 
                     'Native', 'Asian', 'Pacific', 'Citizen', 'Poverty', 'ChildPoverty',
                     'Professional', 'Service', 'Office', 'Construction', 'Production',
                     'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp', 'WorkAtHome',
                     'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork', 
                     'SelfEmployed', 'FamilyWork', 'Unemployment']
    
    # Features disponibles (certaines peuvent manquer)
    available_features = [f for f in base_features if f in df.columns]
    
    tabs = st.tabs(["📊 A · Exploration", "🔧 B · Feature Engineering", 
                    "🤖 C · Modélisation", "📈 D · Évaluation", "🔮 E · Prédiction Interactive"])

    # ── TAB A : Exploration ──────────────────────────────────────────────────
    with tabs[0]:
        section("A – Exploration du Dataset Census 2015")
        
        col1, col2 = st.columns(2)
        with col1:
            sub("Aperçu général")
            metric_cards([
                ("Tracts", f"{len(df):,}"),
                ("États", df['State'].nunique()),
                ("Comtés", df['County'].nunique()),
                ("Variables", len(df.columns)),
            ])
            
            st.markdown("**5 premiers tracts**")
            st.dataframe(df[['State', 'County', 'TotalPop', 'Income', 'Poverty']].head(), use_container_width=True)
        
        with col2:
            sub("Distribution du revenu médian")
            fig, ax = styled_fig()
            ax.hist(df['Income'].dropna(), bins=50, color=BLUE, alpha=0.7, edgecolor='white')
            ax.axvline(df['Income'].median(), color=ORANGE, linewidth=2, linestyle='--', 
                       label=f"Médiane: ${df['Income'].median():,.0f}")
            ax.set_xlabel("Revenu médian ($)")
            ax.set_ylabel("Nombre de tracts")
            ax.set_title("Distribution des revenus", fontweight='bold')
            ax.legend()
            st.pyplot(fig)
            plt.close()
        
        sub("Variables clés - Statistiques descriptives")
        key_stats = ['TotalPop', 'Income', 'Poverty', 'Unemployment', 'MeanCommute', 'Professional']
        available_stats = [k for k in key_stats if k in df.columns]
        st.dataframe(df[available_stats].describe().round(2), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            sub("Top 10 États par nombre de tracts")
            state_counts = df['State'].value_counts().head(10)
            fig, ax = styled_fig()
            bars = ax.barh(state_counts.index, state_counts.values, color=BLUE, alpha=0.8)
            ax.set_xlabel("Nombre de tracts")
            ax.set_title("Distribution par état", fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            sub("Corrélation avec le revenu")
            if 'Income' in df.columns:
                corr_income = df[available_features + ['Income']].corr()['Income'].drop('Income').sort_values()
                fig, ax = styled_fig()
                colors = [ORANGE if c < 0 else BLUE for c in corr_income.values]
                ax.barh(corr_income.index[:15], corr_income.values[:15], color=colors, alpha=0.8)
                ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
                ax.set_xlabel("Corrélation avec le revenu")
                ax.set_title("Top 15 corrélations", fontweight='bold')
                st.pyplot(fig)
                plt.close()
        
        sub("Matrice de corrélation")
        corr_features = ['TotalPop', 'Income', 'Poverty', 'Unemployment', 'Professional', 
                         'Service', 'Office', 'Construction', 'Production', 'MeanCommute']
        corr_available = [f for f in corr_features if f in df.columns]
        
        fig, ax = styled_fig()
        fig.set_size_inches(10, 8)
        corr_matrix = df[corr_available].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Matrice de corrélation", fontweight='bold', fontsize=14)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        info("🔍 Observations : Le revenu est fortement corrélé positivement avec les emplois professionnels et négativement avec le taux de pauvreté et le chômage.")

    # ── TAB B : Feature Engineering ──────────────────────────────────────────
    with tabs[1]:
        section("B – Feature Engineering Avancé")
        
        st.markdown("""
        <div class="info-box">
        ℹ️ Création de nouvelles features à partir des variables existantes pour améliorer les performances des modèles.
        </div>
        """, unsafe_allow_html=True)
        
        # Création des nouvelles features
        df_fe = df.copy()
        
        with st.spinner("Création des features dérivées..."):
            # Ratios démographiques
            df_fe['MenWomenRatio'] = df_fe['Men'] / (df_fe['Women'] + 1)
            df_fe['ChildPovertyRatio'] = df_fe['ChildPoverty'] / (df_fe['Poverty'] + 1)
            
            # Structure de l'emploi
            df_fe['WhiteCollar'] = df_fe['Professional'] + df_fe['Office']
            df_fe['BlueCollar'] = df_fe['Construction'] + df_fe['Production'] + df_fe['Service']
            df_fe['WhiteBlueRatio'] = df_fe['WhiteCollar'] / (df_fe['BlueCollar'] + 1)
            df_fe['EmploymentRate'] = df_fe['Employed'] / (df_fe['TotalPop'] + 1) * 100
            
            # Transports
            df_fe['CarDependency'] = df_fe['Drive'] / (df_fe['Carpool'] + df_fe['Transit'] + df_fe['Walk'] + 1)
            df_fe['PublicTransitUse'] = df_fe['Transit'] / (df_fe['Drive'] + df_fe['Carpool'] + 1) * 100
            df_fe['WorkFromHome'] = df_fe['WorkAtHome'] / (df_fe['Employed'] + 1) * 100
            
            # Indicateurs économiques
            df_fe['IncomePerCapita'] = df_fe['Income'] / (df_fe['TotalPop'] + 1)
            df_fe['PovertySeverity'] = df_fe['Poverty'] * df_fe['ChildPoverty'] / 100
            df_fe['UnemploymentSeverity'] = df_fe['Unemployment'] * (1 - df_fe['Professional']/100)
            
            # Indice de diversité (Simpson)
            races = ['White', 'Black', 'Hispanic', 'Asian', 'Native', 'Pacific']
            race_cols = [r for r in races if r in df_fe.columns]
            if len(race_cols) > 1:
                race_data = df_fe[race_cols].fillna(0) / 100
                df_fe['DiversityIndex'] = 1 - (race_data ** 2).sum(axis=1)
        
        new_features = ['MenWomenRatio', 'WhiteCollar', 'BlueCollar', 'WhiteBlueRatio',
                        'EmploymentRate', 'CarDependency', 'PublicTransitUse', 'WorkFromHome',
                        'IncomePerCapita', 'PovertySeverity', 'UnemploymentSeverity']
        
        if 'DiversityIndex' in df_fe.columns:
            new_features.append('DiversityIndex')
        
        sub("Nouvelles features créées")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Features démographiques & emploi**")
            st.dataframe(df_fe[['MenWomenRatio', 'WhiteCollar', 'BlueCollar', 
                               'EmploymentRate', 'DiversityIndex']].describe().round(3), 
                        use_container_width=True)
        
        with col2:
            st.markdown("**Features transport & économie**")
            st.dataframe(df_fe[['CarDependency', 'PublicTransitUse', 'WorkFromHome',
                               'IncomePerCapita', 'PovertySeverity']].describe().round(3),
                        use_container_width=True)
        
        sub("Impact des nouvelles features sur le revenu")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.patch.set_facecolor('#FFFFFF')
        plot_features = ['WhiteBlueRatio', 'EmploymentRate', 'CarDependency', 
                        'WorkFromHome', 'DiversityIndex', 'PovertySeverity']
        
        for idx, (ax, feat) in enumerate(zip(axes.flatten(), plot_features)):
            if feat in df_fe.columns:
                ax.scatter(df_fe[feat], df_fe['Income'], alpha=0.3, s=5, color=BLUE)
                ax.set_xlabel(feat)
                ax.set_ylabel('Revenu')
                ax.set_facecolor('#F9FAFB')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Calcul et affichage de la corrélation
                corr = df_fe[[feat, 'Income']].corr().iloc[0,1]
                ax.set_title(f'{feat}\n(r = {corr:.3f})', fontweight='bold')
        
        fig.suptitle("Relation entre nouvelles features et revenu", fontweight='bold', fontsize=14)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Sélection des features finales
        final_features = available_features + [f for f in new_features if f in df_fe.columns]
        
        success = f"✅ {len(final_features)} features disponibles pour la modélisation"
        st.markdown(f'<div class="success-box" style="background:{BLUE}_lt;padding:0.8rem;border-radius:8px;margin:1rem 0;">{success}</div>',
                   unsafe_allow_html=True)

    # ── TAB C : Modélisation ─────────────────────────────────────────────────
    with tabs[2]:
        section("C – Modélisation Prédictive")
        
        col1, col2 = st.columns(2)
        with col1:
            task = st.radio("🎯 Type de tâche", 
                           ["Classification (Revenu élevé vs faible)",
                            "Régression (Prédire le revenu)"])
        
        with col2:
            test_size = st.slider("📊 Taille de l'ensemble de test", 0.1, 0.4, 0.2, 0.05)
            use_fe = st.checkbox("✅ Utiliser les features engineering", value=True)
        
        # Préparation des données
        if use_fe and 'df_fe' in locals():
            X_df = df_fe.copy()
        else:
            X_df = df.copy()
        
        # Sélection des features
        exclude_cols = ['CensusTract', 'State', 'County', 'Income', 'IncomeErr', 
                       'IncomePerCap', 'IncomePerCapErr', 'target_high_income', 
                       'target_income_log']
        
        feature_cols = [c for c in X_df.columns if c not in exclude_cols and 
                       X_df[c].dtype in ['int64', 'float64']]
        
        # Cible
        if "Classification" in task:
            y = X_df['target_high_income'].values
            target_name = "target_high_income"
            scoring = "accuracy"
        else:
            y = X_df['target_income_log'].values
            target_name = "target_income_log"
            scoring = "r2"
        
        X = X_df[feature_cols].fillna(X_df[feature_cols].median()).values
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=42, stratify=y if "Classification" in task else None)
        
        # Normalisation
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        sub("Sélection des modèles")
        
        models = {
            "KNN": KNeighborsClassifier if "Classification" in task else KNeighborsRegressor,
            "Arbre de décision": DecisionTreeClassifier if "Classification" in task else DecisionTreeRegressor,
            "Random Forest": RandomForestClassifier if "Classification" in task else RandomForestRegressor,
            "Gradient Boosting": GradientBoostingClassifier if "Classification" in task else GradientBoostingRegressor,
        }
        
        if "Classification" in task:
            models["Régression Logistique"] = LogisticRegression
            models["SVM"] = SVC
        else:
            models["Ridge"] = Ridge
            models["SVR"] = SVR
        
        results = []
        progress_bar = st.progress(0)
        
        for idx, (name, ModelClass) in enumerate(models.items()):
            with st.spinner(f"Entraînement de {name}..."):
                try:
                    # Paramètres par défaut adaptés
                    if name == "KNN":
                        model = ModelClass(n_neighbors=7)
                        model.fit(X_train_sc, y_train)
                        y_pred = model.predict(X_test_sc)
                    elif name in ["SVM", "SVR"]:
                        model = ModelClass()
                        model.fit(X_train_sc, y_train)
                        y_pred = model.predict(X_test_sc)
                    else:
                        model = ModelClass(random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    if "Classification" in task:
                        score = accuracy_score(y_test, y_pred)
                        metric = "Accuracy"
                    else:
                        score = r2_score(y_test, y_pred)
                        metric = "R²"
                    
                    results.append({
                        "Modèle": name,
                        metric: round(score, 4),
                        "Utilise normalisation": "Oui" if name in ["KNN", "SVM", "SVR"] else "Non"
                    })
                except Exception as e:
                    st.warning(f"Erreur avec {name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(models))
        
        progress_bar.empty()
        
        # Affichage des résultats
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values(list(results_df.columns[1]), ascending=False)
            st.dataframe(results_df, use_container_width=True)
            
            # Visualisation
            fig, ax = styled_fig()
            fig.set_size_inches(10, 5)
            metric_col = results_df.columns[1]
            colors = [BLUE if i == 0 else ORANGE for i in range(len(results_df))]
            bars = ax.bar(results_df['Modèle'], results_df[metric_col], color=colors, alpha=0.8)
            ax.set_ylabel(metric_col)
            ax.set_title(f"Comparaison des modèles - {metric_col}", fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, val in zip(bars, results_df[metric_col]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            best_model_name = results_df.iloc[0]['Modèle']
            best_score = results_df.iloc[0][metric_col]
            
            st.markdown(f"""
            <div style="background:linear-gradient(90deg,{BLUE},{ORANGE});
                        padding:1rem 2rem;border-radius:12px;color:white;text-align:center;margin:1rem 0">
              <div style="font-size:1.3rem;font-weight:600">🏆 Meilleur modèle : {best_model_name}</div>
              <div style="font-size:1.8rem;font-weight:800">{metric_col} = {best_score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB D : Évaluation approfondie ───────────────────────────────────────
    with tabs[3]:
        section("D – Évaluation Approfondie")
        
        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.selectbox("📌 Sélectionner un modèle", 
                                       [r['Modèle'] for r in results] if 'results' in locals() else ["Random Forest"])
        with col2:
            show_confusion = st.checkbox("📊 Afficher la matrice de confusion", value=True)
        
        if 'X_train' in locals() and 'X_test' in locals():
            # Ré-entraîner le modèle choisi
            if "Classification" in task:
                if model_choice == "KNN":
                    model = KNeighborsClassifier(n_neighbors=7)
                    model.fit(X_train_sc, y_train)
                    y_pred = model.predict(X_test_sc)
                    y_proba = model.predict_proba(X_test_sc)[:, 1]
                elif model_choice in ["SVM"]:
                    model = SVC(probability=True)
                    model.fit(X_train_sc, y_train)
                    y_pred = model.predict(X_test_sc)
                    y_proba = model.predict_proba(X_test_sc)[:, 1]
                else:
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    elif model_choice == "Gradient Boosting":
                        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                    elif model_choice == "Arbre de décision":
                        model = DecisionTreeClassifier(max_depth=10, random_state=42)
                    else:
                        model = LogisticRegression(random_state=42)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
            else:
                # Régression
                if model_choice == "KNN":
                    model = KNeighborsRegressor(n_neighbors=7)
                    model.fit(X_train_sc, y_train)
                    y_pred = model.predict(X_test_sc)
                elif model_choice in ["SVR"]:
                    model = SVR()
                    model.fit(X_train_sc, y_train)
                    y_pred = model.predict(X_test_sc)
                else:
                    if model_choice == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif model_choice == "Gradient Boosting":
                        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    elif model_choice == "Arbre de décision":
                        model = DecisionTreeRegressor(max_depth=10, random_state=42)
                    else:
                        model = Ridge(alpha=1.0)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
            
            if "Classification" in task:
                col1, col2, col3 = st.columns(3)
                with col1:
                    metric_cards([("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")])
                with col2:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    metric_cards([("Précision", f"{precision_score(y_test, y_pred):.3f}")])
                with col3:
                    metric_cards([("Rappel", f"{recall_score(y_test, y_pred):.3f}")])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if show_confusion:
                        sub("Matrice de confusion")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = styled_fig()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                   xticklabels=['Faible revenu', 'Haut revenu'],
                                   yticklabels=['Faible revenu', 'Haut revenu'])
                        ax.set_xlabel('Prédit')
                        ax.set_ylabel('Réel')
                        ax.set_title('Matrice de confusion', fontweight='bold')
                        st.pyplot(fig)
                        plt.close()
                
                with col2:
                    sub("Courbe ROC")
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig, ax = styled_fig()
                    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Aléatoire')
                    ax.set_xlabel('Taux faux positifs')
                    ax.set_ylabel('Taux vrais positifs')
                    ax.set_title('Courbe ROC', fontweight='bold')
                    ax.legend(loc='lower right')
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    st.pyplot(fig)
                    plt.close()
            
            else:
                # Métriques de régression
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    metric_cards([("R²", f"{r2_score(y_test, y_pred):.3f}")])
                with col2:
                    metric_cards([("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")])
                with col3:
                    metric_cards([("MSE", f"{mean_squared_error(y_test, y_pred):.3f}")])
                with col4:
                    metric_cards([("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    sub("Prédictions vs Valeurs réelles")
                    fig, ax = styled_fig()
                    ax.scatter(y_test, y_pred, alpha=0.5, s=10, color=BLUE)
                    
                    # Ligne parfaite
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Idéal')
                    
                    ax.set_xlabel('Valeurs réelles')
                    ax.set_ylabel('Prédictions')
                    ax.set_title('Prédictions vs Réel', fontweight='bold')
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    sub("Distribution des résidus")
                    residuals = y_test - y_pred
                    fig, ax = styled_fig()
                    ax.hist(residuals, bins=30, color=ORANGE, alpha=0.7, edgecolor='white')
                    ax.axvline(0, color=BLUE, linestyle='--', linewidth=2)
                    ax.set_xlabel('Résidus')
                    ax.set_ylabel('Fréquence')
                    ax.set_title('Distribution des résidus', fontweight='bold')
                    st.pyplot(fig)
                    plt.close()
            
            # Importance des features pour les modèles qui le supportent
            if hasattr(model, 'feature_importances_'):
                sub("Importance des variables")
                importances = model.feature_importances_
                indices = np.argsort(importances)[-15:]  # Top 15
                
                fig, ax = styled_fig()
                fig.set_size_inches(10, 6)
                ax.barh(range(len(indices)), importances[indices], color=BLUE, alpha=0.8)
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_cols[i] for i in indices])
                ax.set_xlabel('Importance')
                ax.set_title('Top 15 features les plus importantes', fontweight='bold')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

    # ── TAB E : Prédiction Interactive ───────────────────────────────────────
    with tabs[4]:
        section("E – Prédiction Interactive")
        
        st.markdown("""
        <div class="info-box">
        ℹ️ Utilisez les sliders ci-dessous pour définir les caractéristiques d'un census tract
        et obtenir une prédiction de revenu en temps réel.
        </div>
        """, unsafe_allow_html=True)
        
        # Ré-entraîner le meilleur modèle pour la prédiction
        if 'results' in locals() and not results_df.empty:
            best_model_name = results_df.iloc[0]['Modèle']
            
            with st.spinner(f"Préparation du modèle {best_model_name}..."):
                if "Classification" in task:
                    if best_model_name == "KNN":
                        final_model = KNeighborsClassifier(n_neighbors=7)
                        final_model.fit(X_train_sc, y_train)
                        use_scaler = True
                    elif best_model_name in ["SVM"]:
                        final_model = SVC(probability=True)
                        final_model.fit(X_train_sc, y_train)
                        use_scaler = True
                    else:
                        if best_model_name == "Random Forest":
                            final_model = RandomForestClassifier(n_estimators=100, random_state=42)
                        elif best_model_name == "Gradient Boosting":
                            final_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                        else:
                            final_model = LogisticRegression(random_state=42)
                        final_model.fit(X_train, y_train)
                        use_scaler = False
                else:
                    if best_model_name == "KNN":
                        final_model = KNeighborsRegressor(n_neighbors=7)
                        final_model.fit(X_train_sc, y_train)
                        use_scaler = True
                    elif best_model_name in ["SVR"]:
                        final_model = SVR()
                        final_model.fit(X_train_sc, y_train)
                        use_scaler = True
                    else:
                        if best_model_name == "Random Forest":
                            final_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif best_model_name == "Gradient Boosting":
                            final_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        else:
                            final_model = Ridge(alpha=1.0)
                        final_model.fit(X_train, y_train)
                        use_scaler = False
            
            # Interface de prédiction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_pop = st.slider("Population totale", 0, 20000, 5000, step=100)
                poverty = st.slider("Taux de pauvreté (%)", 0.0, 50.0, 15.0, step=0.5)
                professional = st.slider("Emplois pro (%)", 0.0, 80.0, 25.0, step=0.5)
            
            with col2:
                unemployment = st.slider("Chômage (%)", 0.0, 30.0, 8.0, step=0.5)
                white_collar = st.slider("Cols blancs (%)", 0.0, 80.0, 30.0, step=0.5)
                mean_commute = st.slider("Temps de trajet (min)", 5.0, 60.0, 25.0, step=1.0)
            
            with col3:
                car_dep = st.slider("Dépendance auto (%)", 0.0, 100.0, 80.0, step=1.0)
                diversity = st.slider("Indice de diversité", 0.0, 1.0, 0.5, step=0.05)
            
            if st.button("🔮 Prédire le revenu", use_container_width=True):
                # Construction du vecteur d'entrée
                input_dict = {
                    'TotalPop': total_pop,
                    'Poverty': poverty,
                    'Professional': professional,
                    'Unemployment': unemployment,
                    'MeanCommute': mean_commute,
                }
                
                # Ajouter les features calculées
                if 'WhiteCollar' in feature_cols:
                    input_dict['WhiteCollar'] = white_collar
                if 'CarDependency' in feature_cols:
                    input_dict['CarDependency'] = car_dep
                if 'DiversityIndex' in feature_cols:
                    input_dict['DiversityIndex'] = diversity
                
                # Créer le vecteur dans le bon ordre
                input_vector = []
                for col in feature_cols:
                    if col in input_dict:
                        input_vector.append(input_dict[col])
                    else:
                        # Valeur par défaut (médiane)
                        input_vector.append(X_df[col].median())
                
                input_array = np.array(input_vector).reshape(1, -1)
                
                # Appliquer le scaler si nécessaire
                if use_scaler:
                    input_array = scaler.transform(input_array)
                
                # Prédiction
                prediction = final_model.predict(input_array)[0]
                
                if "Classification" in task:
                    proba = final_model.predict_proba(input_array)[0]
                    result_text = "HAUT REVENU" if prediction == 1 else "FAIBLE REVENU"
                    prob_text = f"Probabilité : {max(proba):.1%}"
                    
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,{BLUE_DK},{ORANGE});
                                padding:1.5rem 2rem;border-radius:14px;color:white;text-align:center;margin-top:1rem">
                      <div style="font-size:1.2rem;opacity:0.9;">Prédiction</div>
                      <div style="font-size:2.5rem;font-weight:800">{result_text}</div>
                      <div style="font-size:1.1rem;margin-top:0.5rem;">{prob_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Revenir à l'échelle normale (inverse du log)
                    income_pred = np.expm1(prediction)
                    
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,{BLUE_DK},{ORANGE});
                                padding:1.5rem 2rem;border-radius:14px;color:white;text-align:center;margin-top:1rem">
                      <div style="font-size:1.2rem;opacity:0.9;">Revenu médian prédit</div>
                      <div style="font-size:3rem;font-weight:800">${income_pred:,.0f}</div>
                      <div style="font-size:1rem;margin-top:0.5rem;">par an</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Sauvegarde du modèle
            st.markdown("---")
            sub("Télécharger le modèle entraîné")
            
            buf = io.BytesIO()
            pickle.dump({
                'model': final_model,
                'scaler': scaler if use_scaler else None,
                'features': feature_cols,
                'task': task,
                'use_scaler': use_scaler
            }, buf)
            buf.seek(0)
            
            st.download_button(
                "⬇️ Télécharger le modèle final",
                buf,
                file_name="census_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
        st.markdown("---")
        sub("Télécharger le modèle final")
        buf = io.BytesIO()
        pickle.dump({"model": best_final, "features": extended_features, "params": best_params_f}, buf)
        buf.seek(0)
        st.download_button("⬇️ Télécharger modele_final.pkl", buf, file_name="modele_final_partie3.pkl",
                           mime="application/octet-stream")

        info("Le modèle final (Gradient Boosting avec features enrichies) offre les meilleures performances et est prêt pour le déploiement en production.")
