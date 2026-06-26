# ==============================================================================
# TESI DI LAUREA IN MACHINE LEARNING - GATTA GENNARO
# PIPELINE COMPLETA AVANZATA CON METRICHE DI EXPLAINABLE AI (XAI) E LATENZA
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Per il calcolo della latenza di inferenza

# Istruzione fondamentale per disattivare i messaggi di "FutureWarning"
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Strumenti di Scikit-Learn per il pre-processing, validazione, tuning e modelli
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Metriche di valutazione
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configurazione grafica standard per evitare crash o blocchi di rendering
plt.style.use('default') 
sns.set_theme(style="whitegrid") 

# ==============================================================================
# BLOCCO 1: CARICAMENTO DEL DATASET ED ANALISI ESPLORATIVA (EDA)
# ==============================================================================
print("\n" + "="*80)
print("--- FASE 1: CARICAMENTO DEL DATASET REALE ED ISPEZIONE ---")
print("="*80)

# Caricamento del file train.csv reale
df = pd.read_csv('train.csv')
print(f"[INFO] Dataset reale caricato con successo! Dimensioni: {df.shape[0]} righe, {df.shape[1]} colonne.")

# Calcolo statistico dello sbilanciamento delle classi per la discussione orale
conteggio_classi = df['Depression'].value_counts()
percentuali_classi = df['Depression'].value_counts(normalize=True) * 100
print(f"[EDA] Distribuzione target: Sani (No) = {conteggio_classi['No']} ({percentuali_classi['No']:.2f}%)")
print(f"[EDA] Distribuzione target: A Rischio (Yes) = {conteggio_classi['Yes']} ({percentuali_classi['Yes']:.2f}%)")

# Rimuoviamo la colonna 'index' se presente poiché è un identificativo e non una feature predittiva
if 'index' in df.columns:
    df = df.drop(columns=['index'])

# Generazione e salvataggio immagini EDA sullo sbilanciamento del target
plt.figure(figsize=(6, 4))
sns.countplot(x='Depression', data=df, order=['No', 'Yes'], palette="Set2")
plt.title('Distribuzione della Variabile Target (Depression) nel Dataset Reale', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('01_sbilanciamento_target.png', dpi=300)
plt.close()

# ==============================================================================
# BLOCCO 2: PRE-PROCESSING E STRUTTURAZIONE DATI
# ==============================================================================
print("\n" + "="*80)
print("--- FASE 2: PRE-PROCESSING E CODIFICA FEATURE ---")
print("="*80)

X = df.drop(columns=['Depression'])  
y = df['Depression']                 

# Trasformazione delle variabili categoriali (One-Hot Encoding)
X_encoded = pd.get_dummies(X, drop_first=True)
colonne_finali = X_encoded.columns
print(f"[INFO] Numero di feature dopo il One-Hot Encoding: {X_encoded.shape[1]}")

# Codifica del Target ('No' -> 0, 'Yes' -> 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split Stratificato di Train e Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded)
print(f"[INFO] Record di Addestramento (Train Set): {X_train.shape[0]}")
print(f"[INFO] Record di Validazione (Test Set): {X_test.shape[0]}")

# Standardizzazione Z-Score
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# BLOCCO 3: HYPERPARAMETER TUNING + BILANCIAMENTO DELLE CLASSI (CLASS WEIGHTING)
# ==============================================================================
print("\n" + "="*80)
print("--- FASE 3: TUNING IPERPARAMETRI CON TECNICHE DI BILANCIAMENTO ---")
print("="*80)

# 1. K-Nearest Neighbors
print("[1/4] Ottimizzazione k-NN (Pesi basati sulla distanza)...")
param_grid_knn = {'n_neighbors': [3, 5, 7, 11], 'weights': ['distance'], 'metric': ['euclidean', 'manhattan']}
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='f1', n_jobs=-1)
knn_grid.fit(X_train_scaled, y_train)

# 2. Random Forest
print("[2/4] Ottimizzazione Random Forest (Costrizione Penalità Bilanciata)...")
param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [5, 10, None], 'criterion': ['gini', 'entropy']}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)

# 3. Regressione Logistica
print("[3/4] Ottimizzazione Regressione Logistica (Costrizione Penalità Bilanciata)...")
param_grid_lr = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs']}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'), param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
lr_grid.fit(X_train_scaled, y_train)

# 4. Support Vector Machines
print("[4/4] Ottimizzazione Support Vector Machines (Costrizione Penalità Bilanciata)...")
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
svm_grid = GridSearchCV(SVC(random_state=42, probability=True, class_weight='balanced'), param_grid_svm, cv=5, scoring='f1', n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)

# ==============================================================================
# BLOCCO 4: REPORT DELLE METRICHE DI VALUTAZIONE E PROFILAZIONE DEI TEMPI
# ==============================================================================
print("\n" + "="*80)
print("--- FASE 4: REPORT PERFORMANCE E LATENZA COMPUTAZIONALE ---")
print("="*80)

modelli_ottimizzati = {
    'K-Nearest Neighbors (k-NN)': (knn_grid.best_estimator_, X_test_scaled),
    'Random Forest (RF)': (rf_grid.best_estimator_, X_test),
    'Logistic Regression (LR)': (lr_grid.best_estimator_, X_test_scaled),
    'Support Vector Machines (SVM)': (svm_grid.best_estimator_, X_test_scaled)
}

metriche_finali = {}

for nome, (modello, dataset_test) in modelli_ottimizzati.items():
    # Misurazione precisa del tempo di esecuzione dell'inferenza (Latenza)
    start_time = time.time()
    y_pred = modello.predict(dataset_test)
    end_time = time.time()
    latenza_msec = (end_time - start_time) * 1000  # Convertito in millisecondi
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metriche_finali[nome] = {
        'Accuracy': acc, 
        'Precision': prec, 
        'Recall': rec, 
        'F1-Score': f1,
        'Latenza (ms)': latenza_msec
    }
    
    print(f"\n────────────────────────────────────────────────────────────────────────")
    print(f"MATRICE DI CONFUSIONE & PERFORMANCE: {nome.upper()}")
    print(f"────────────────────────────────────────────────────────────────────────")
    print(f"  [Reale: Sano]      ->  Veri Negativi (TN): {tn:3d}  |  Falsi Positivi (FP): {fp:3d}")
    print(f"  [Reale: Depresso]  ->  Falsi Negativi (FN): {fn:3d}  |  Veri Positivi (TP): {tp:3d}")
    print(f"  " + "-"*64)
    print(f"  ├─ ACCURACY     = {acc:.4f}")
    print(f"  ├─ PRECISION    = {prec:.4f}")
    print(f"  ├─ RECALL       = {rec:.4f}")
    print(f"  ├─ F1-SCORE     = {f1:.4f}")
    print(f"  └─ LATENZA COMP. = {latenza_msec:.3f} ms")

df_risultati = pd.DataFrame(metriche_finali).T
print("\n" + "="*80)
print("TABELLA COMPARATIVA POST-BILANCIAMENTO")
print("="*80)
print(df_risultati.round(4))

# Esportazione automatica dei risultati in CSV per la tesi
df_risultati.to_csv("report_prestazioni_modelli.csv")
print("\n[INFO] Tabella delle metriche salvata localmente in 'report_prestazioni_modelli.csv'")

# Generazione grafico comparativo delle metriche principali
df_grafico = df_risultati.drop(columns=['Latenza (ms)'])
df_grafico.plot(kind='bar', figsize=(12, 6), cmap="Set2")
plt.title('Performance Comparativa dei Modelli Ottimizzati su Dati Reali', fontsize=12, fontweight='bold')
plt.ylabel('Punteggio (Score)')
plt.xticks(rotation=0)
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('03_confronto_completo_4_modelli.png', dpi=300)
plt.close()

# ==============================================================================
# BLOCCO 5: EXPLAINABLE AI (XAI) - IMPORTANZA DELLE FEATURE (RANDOM FOREST)
# ==============================================================================
print("\n" + "="*80)
print("--- FASE 5: EXPLAINABLE AI (FEATURE IMPORTANCE INTERPRETATION) ---")
print("="*80)

rf_best = rf_grid.best_estimator_
importances = rf_best.feature_importances_
indices = np.argsort(importances)[::-1]

print("Classifica delle feature più determinanti secondo l'algoritmo Random Forest:")
for f in range(min(5, len(colonne_finali))):
    print(f"  {f+1}. Feature: '{colonne_finali[indices[f]]}' -> Peso Matematico: {importances[indices[f]]:.4f}")

# Creazione del grafico di Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices[:10]], y=colonne_finali[indices[:10]], palette="Blues_r")
plt.title('Explainable AI: Top 10 Feature Predittive per lo Screening della Depressione', fontsize=12, fontweight='bold')
plt.xlabel('Grado di Importanza Relativa')
plt.ylabel('Feature Cliniche / Accademiche')
plt.tight_layout()
plt.savefig('04_feature_importance_xai.png', dpi=300)
plt.close()
print("[INFO] Grafico di interpretabilità salvato come '04_feature_importance_xai.png'")

# ==============================================================================
# BLOCCO 6: UTILIZZO DEI METODI (INFERENCE SU 3 CASI REALI ESTRATTI)
# ==============================================================================
print("\n" + "="*80)
print("--- FASE 6: UTILIZZO PRATICO DEI METODI SU 3 STUDENTI DEL DATASET ---")
print("="*80)

# Estrazione esatta dei primi 3 studenti reali presenti nel file CSV originario
nuovi_studenti_raw = pd.read_csv('train.csv').head(3)
diagnosi_reale_csv = nuovi_studenti_raw['Depression'].values

if 'index' in nuovi_studenti_raw.columns:
    nuovi_studenti_raw = nuovi_studenti_raw.drop(columns=['index'])
nuovi_studenti_features = nuovi_studenti_raw.drop(columns=['Depression'])

# Adeguamento delle colonne tramite One-Hot Encoding e allineamento strutturale
nuovi_studenti_encoded = pd.get_dummies(nuovi_studenti_features)
nuovi_studenti_encoded = nuovi_studenti_encoded.reindex(columns=colonne_finali, fill_value=0)

# Utilizzo delle funzioni .predict() e .predict_proba() sul modello Random Forest Ottimizzato
modello_scelto = rf_grid.best_estimator_
predizioni_secche = modello_scelto.predict(nuovi_studenti_encoded)
etichette_diagnosi = le.inverse_transform(predizioni_secche)
probabilita = modello_scelto.predict_proba(nuovi_studenti_encoded)

print("\nRISULTATI APPLICAZIONE DEI METODI SUI 3 PROFILI REALI:")
for i in range(len(nuovi_studenti_raw)):
    print(f"\n 👤 CASO STUDIO REALE {i+1}:")
    print(f"  ├─ Parametri Estratti: Età={nuovi_studenti_raw.loc[i, 'Age']}, Genere={nuovi_studenti_raw.loc[i, 'Gender']}, Ore Studio={nuovi_studenti_raw.loc[i, 'Study Hours']}h, Stress Finanziario={nuovi_studenti_raw.loc[i, 'Financial Stress']}/5.")
    print(f"  ├─ Diagnosi Effettiva nel CSV  -> {diagnosi_reale_csv[i].upper()}")
    print(f"  ├─ Metodo '.predict()':       Diagnosi Modello        -> RISCHIO: {etichette_diagnosi[i].upper()}")
    print(f"  └─ Metodo '.predict_proba()': Analisi delle Probabilità -> SANO (No): {probabilita[i][0]*100:.2f}% | RISCHIO (Yes): {probabilita[i][1]*100:.2f}%")

print("\n" + "="*80)
print("[ESECUZIONE COMPLETA] Analisi conclusa con successo. Pipeline terminata!")
print("="*80 + "\n")