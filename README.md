# ✈️ Flight Delay Detection / Flugverspätungs-Erkennung

Dieses Projekt nutzt Machine Learning zur Vorhersage von Flugverspätungen auf Basis von US-Flugdaten aus dem Jahr 2024.

*This project uses machine learning to predict flight delays based on 2024 U.S. flight data.*

## 📋 Projektübersicht / Project Overview

Das Projekt beinhaltet eine vollständige Machine Learning Pipeline:
- Datenvorverarbeitung und Feature Engineering
- Training eines XGBoost Klassifikators
- Modell-Evaluation und Leistungsanalyse
- Speicherung der Vorhersagen in einer SQLite-Datenbank

*The project includes a complete machine learning pipeline:*
- *Data preprocessing and feature engineering*
- *Training an XGBoost classifier*
- *Model evaluation and performance analysis*
- *Storing predictions in a SQLite database*

## 📊 Datenquelle / Data Source

### Flight Delay Dataset — 2024

Dieses Projekt verwendet den **Flight Delay Dataset — 2024**, der auf Kaggle verfügbar ist.

*This project uses the **Flight Delay Dataset — 2024**, available on Kaggle.*

🔗 **Kaggle Link:** [Flight Data 2024 Dataset](https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024/data)

### 📥 Download-Details / Download Details

Es gibt zwei Versionen des Datensatzes:

*There are two versions of the dataset:*

| Datei / File | Zeilen / Rows | Spalten / Columns | Größe / Size | Verwendung / Usage |
|--------------|---------------|-------------------|--------------|-------------------|
| `flight_data_2024.csv` | ~7 Millionen / ~7 million | 35 | ~1.31 GB | Vollständiger Datensatz / Full dataset |
| `flight_data_2024_sample.csv` | 10.000 | 35 | ~10 MB | Beispieldatensatz für Entwicklung / Sample for development |

### 🔍 Wichtige Features / Key Features

**Feature Engineering für Verspätungs-Ziel / Feature Engineering for Delay Target:**
- `DepDel15`: Abflugverspätung > 15 Minuten / Departure delay > 15 minutes
- `ArrDel15`: Ankunftsverspätung > 15 Minuten / Arrival delay > 15 minutes

Diese Spalten können als Basis für die Erstellung der Zielvariable `Delayed` verwendet werden.

*These columns can be used as the basis for creating the `Delayed` target variable.*

### 📌 Ursprung / Original Source

Die Daten stammen ursprünglich aus der **TranStats On-Time Performance-Datenbank** des US-Verkehrsministeriums (Bureau of Transportation Statistics - BTS).

*The data originally comes from the **TranStats On-Time Performance database** of the U.S. Department of Transportation (Bureau of Transportation Statistics - BTS).*

### 💾 SQLite-Datenbank / SQLite Database

Da der vollständige Datensatz über 1 GB groß ist, werden die Vorhersagen in einer SQLite-Datenbank (`flights2024.db`) gespeichert. Dies optimiert die Performance bei Abfragen und ermöglicht effizientes Datenmanagement.

*Since the full dataset is over 1 GB, predictions are stored in a SQLite database (`flights2024.db`). This optimizes query performance and enables efficient data management.*

## 📂 Projektstruktur / Project Structure

```
Flight-delay-detection-/
├── data/                                    # Datensätze / Datasets
│   └── flight_data_2024.csv.dvc           # DVC-verwaltete Daten / DVC-managed data
├── docs/                                    # Dokumentation / Documentation
│   └── flight_delay_insights_2024.png     # Visualisierungen / Visualizations
├── notebooks/                               # Jupyter Notebooks
│   └── flight_delay_prediction_analytics.ipynb  # Hauptanalyse / Main analysis
├── .gitignore                              # Git ignore Regeln / Git ignore rules
├── README.md                               # Projektdokumentation / Project documentation
└── requirements.txt                        # Python Abhängigkeiten / Python dependencies
```

## 🛠️ Technologie-Stack / Tech Stack

- **Python 3.13**
- **Pandas & NumPy** - Datenverarbeitung / Data processing
- **Scikit-learn** - Preprocessing und Metriken / Preprocessing and metrics
- **XGBoost** - Machine Learning Modell / Machine learning model
- **SQLAlchemy** - Datenbankanbindung / Database integration
- **Matplotlib & Seaborn** - Visualisierung / Visualization
- **DVC** - Daten-Versionskontrolle / Data version control

## ⚙️ Pipeline-Schritte / Pipeline Steps

### 1. Datenaufbereitung / Data Preparation
- Laden der Flugdaten aus `data/flight_data_2024.csv`
- Erstellung der Zielvariable `Delayed` (1 wenn `arr_delay > 15`, sonst 0)
- Speicherung in SQLite-Datenbank `flights2024.db`

*Loading flight data from `data/flight_data_2024.csv`*
*Creating target variable `Delayed` (1 if `arr_delay > 15`, otherwise 0)*
*Storing in SQLite database `flights2024.db`*

### 2. Preprocessing
- Filterung: Nur nicht-stornierte und nicht-umgeleitete Flüge
- Label-Encoding für kategoriale Variablen (`op_unique_carrier`, `origin`, `dest`)
- StandardScaler für numerische Features
- Train/Test Split (80/20)

*Filtering: Only non-cancelled and non-diverted flights*
*Label encoding for categorical variables*
*StandardScaler for numeric features*
*Train/test split (80/20)*

### 3. Modell-Training / Model Training
**XGBoost Classifier Hyperparameter:**
- `n_estimators=300`
- `max_depth=6`
- `learning_rate=0.05`

### 4. Vorhersagen / Predictions
- Vorhersage auf dem gesamten Datensatz
- Speicherung in SQLite-Tabelle `flight_preds_2024`

*Prediction on full dataset*
*Storage in SQLite table `flight_preds_2024`*

## 📊 Modell-Performance / Model Performance

| Metrik / Metric | Pünktlich / On-Time (0) | Verspätet / Delayed (1) |
|-----------------|------------------------|------------------------|
| Precision       | 0.94                   | 0.92                   |
| Recall          | 0.98                   | 0.73                   |
| F1-Score        | 0.96                   | 0.81                   |

**Gesamtgenauigkeit / Overall Accuracy: ✅ 93%**

## 🚀 Installation und Ausführung / Installation and Usage

### Voraussetzungen / Prerequisites
- Python 3.13 oder höher / or higher
- Git
- DVC (für Datenverwaltung / for data management)

### Schritte / Steps

1. **Repository klonen / Clone repository:**
   ```bash
   git clone https://github.com/AndreasTraut/Flight-delay-detection-.git
   cd Flight-delay-detection-
   ```

2. **Abhängigkeiten installieren / Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Daten abrufen / Pull data (with DVC):**
   ```bash
   dvc pull
   ```

4. **Notebook ausführen / Run notebook:**
   ```bash
   jupyter notebook notebooks/flight_delay_prediction_analytics.ipynb
   ```

## 📌 Nächste Schritte / Next Steps

- [ ] Hyperparameter-Tuning für besseren Recall bei verspäteten Flügen
- [ ] Feature Engineering mit Wetter- und Flughafen-Auslastungsdaten
- [ ] Deployment als Flask API oder Streamlit Dashboard
- [ ] Integration zusätzlicher Datenquellen

*Hyperparameter tuning for better recall on delayed flights*
*Feature engineering with weather and airport congestion data*
*Deployment as Flask API or Streamlit dashboard*
*Integration of additional data sources*

## 👤 Autor / Author

**Andreas Traut**

## 📄 Lizenz / License

Dieses Projekt steht unter der MIT-Lizenz.

*This project is licensed under the MIT License.*

