**✈️ Flight Delay Prediction (2024 Data)**
This project builds a machine learning pipeline to predict whether a flight will be delayed or on-time using the 2024 U.S. flight dataset.

The workflow includes:
Data loading & preprocessing
Feature engineering (creating a Delayed target variable)
Training an XGBoost classifier
Evaluating performance (accuracy, precision, recall, F1-score)

Saving predictions into a SQLite database

📂 Project Structure

├── flight_data_2024.csv        # Raw dataset
├── flights2024.db              # SQLite database
├── notebook.ipynb              # Jupyter Notebook with full workflow
├── README.md                   # Project documentation

⚙️ Steps in the Pipeline
1. Data Loading & Target Creation

Loaded flight_data_2024.csv using Pandas.

Created a new binary column:

Delayed = 1 if arr_delay > 15

Delayed = 0 otherwise

Stored cleaned data into a SQLite table (flights2024).

2. Querying & Preprocessing

Queried non-cancelled and non-diverted flights.

Encoded categorical columns (op_unique_carrier, origin, dest) using LabelEncoder.

Scaled numeric features with StandardScaler.

Split dataset into train/test (80/20).

3. Model Training

Trained an XGBoost Classifier with:

n_estimators=300

max_depth=6

learning_rate=0.05

Evaluated with accuracy and classification report.

Results:

Accuracy: ~93%

Strong recall for on-time flights, moderate recall for delayed flights.

4. Predictions & Database Saving

Generated predictions on the full dataset.

Saved results into a new SQLite table: flight_preds_2024.

📊 Model Performance (Test Set)
Metric	On-Time (0)	Delayed (1)
Precision	0.94	0.92
Recall	0.98	0.73
F1-Score	0.96	0.81

Overall Accuracy: ✅ 93%

🛠️ Tech Stack

Python 3.13

Pandas, NumPy

Scikit-learn

XGBoost

SQLite (via SQLAlchemy)

🚀 How to Run

Clone this repository:

git clone https://github.com/yourusername/flight-delay-prediction.git
cd flight-delay-prediction


Install dependencies:

pip install -r requirements.txt


Place the dataset (flight_data_2024.csv) in the project root.

Run the notebook:

jupyter notebook notebook.ipynb

📌 Next Steps / Improvements

Hyperparameter tuning for better recall on delayed flights.

Feature engineering with weather & airport congestion data.

Deploy as a Flask API or Streamlit dashboard for real-time predictions.

👩‍💻 Author

Swarada Kulkarni
Data Analyst | Aspiring Data Scientist

