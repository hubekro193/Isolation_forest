# --- DATA PARAMETERS ---

# •  sex
# •  date
# •  age - Wiek osoby badanej.
# •  height - Wzrost osoby badanej, mierzony w centymetrach.
# •  weight - Waga osoby badanej, mierzona w kilogramach.
# •  discipline - Dyscyplina sportowa badanej osoby.

# •  AeT - Próg aerobowy, czyli intensywność wysiłku [hr], przy której organizm zaczyna korzystać z procesów tlenowych do produkcji energii.
# •  AnT - Próg anaerobowy, czyli intensywność wysiłku [hr], przy której organizm zaczyna produkować energię głównie z procesów beztlenowych.
# •  vo2max - Maksymalna ilość tlenu, jaką organizm może zużyć podczas intensywnego wysiłku.
# •  vo2_at - Ilości tlenu zużywanego na progu anaerobowym (AnT).
# •  ve - Wentylacja minutowa, czyli ilość powietrza wdychanego i wydychanego przez płuca w ciągu jednej minuty.
# •  r - Wskaźnik oddechowy, czyli stosunek objętości wydychanego dwutlenku węgla do objętości wdychanego tlenu.
# •  hrmax - Maksymalne tętno, czyli najwyższa liczba uderzeń serca na minutę podczas maksymalnego wysiłku.
# •  rf - Częstotliwość oddechów na minutę.
# •  vo2max_l_m - VO₂ max w przeliczeniu na masę ciała, co pozwala na bardziej precyzyjne porównanie wydolności między osobami o różnej masie ciała.

# •  hr_6 - Tętno przy prędkości 6 km/h.
# •  la_6 - Poziom kwasu mlekowego przy prędkości 8 km/h. --- PUSTA KOLUMNA---
# •  hr_8 - Tętno przy prędkości 8 km/h.
# •  la_8 - Poziom kwasu mlekowego przy prędkości 8 km/h.
# •  hr_10 - Tętno przy prędkości 10 km/h.
# •  la_10 - Poziom kwasu mlekowego przy prędkości 10 km/h.
# •  hr_12 - Tętno przy prędkości 12 km/h.
# •  la_12 - Poziom kwasu mlekowego przy prędkości 12 km/h.
# •  hr_14 - Tętno przy prędkości 14 km/h.
# •  la_14 - Poziom kwasu mlekowego przy prędkości 14 km/h.
# •  hr_16 - Tętno przy prędkości 16 km/h.
# •  la_16 - Poziom kwasu mlekowego przy prędkości 16 km/h.
# •  hr_18 - Tętno przy prędkości 18 km/h.
# •  la_18 - Poziom kwasu mlekowego przy prędkości 18 km/h.
# •  hr_20 - Tętno przy prędkości 20 km/h.
# •  la_20 - Poziom kwasu mlekowego przy prędkości 20 km/h.
# •  hr_22 - Tętno przy prędkości 22 km/h.
# •  la_22 - Poziom kwasu mlekowego przy prędkości 22 km/h.

# •  z2 – tętno na poziomie rozwoju wytrzymałości tlenowej,
# •  z3 – tętno na poziomie kształtowania mocy tlenowej (strefa mieszana),
# •  z4 – tętno na poziomie progu tlenowo-beztlenowego,
# •  z5 – tętno w strefie wytrzymałości beztlenowej.

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Funkcja '{func.__name__}' wykonała się w czasie: {elapsed_time:.6f} sekund")
        return result

    return wrapper


@measure_time
def load_and_inspect_data(file_path):
    """
    Load the data and display initial statistics.
    """
    data = pd.read_csv(file_path)
    print("Liczba pustych wartości w kolumnach do sprawdzenia:")
    columns_to_check = ['hr_14', 'la_14', 'hr_16', 'la_16', 'hr_18', 'la_18', 'hr_20', 'la_20', 'hr_22', 'la_22']
    empty_counts = data[columns_to_check].isnull().sum()
    print(empty_counts)
    print("\nWartość % pustych pozycji w kolumnach:")
    print((empty_counts / len(data)) * 100)
    return data


@measure_time
def clean_data(data):
    """
    Clean and preprocess the dataset.
    """
    # Drop unnecessary columns
    columns_to_drop = ['date', 'hr_6', 'la_6', 'hr_16', 'la_16', 'hr_18', 'la_18', 'hr_20', 'la_20', 'hr_22', 'la_22']
    data.drop(columns_to_drop, axis=1, inplace=True)

    # Fill missing numeric values with column means
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # Encode categorical columns
    label_encoder = LabelEncoder()
    for col in ['sex', 'discipline', 'rf']:
        if col in data.columns:
            data[col] = label_encoder.fit_transform(data[col])

    return data


@measure_time
def prepare_data(data):
    """
    Split the data into features and labels, and apply scaling.
    """
    X = data.drop(columns=['AeT', 'AnT'])
    y = data[['AeT', 'AnT']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


@measure_time
def train_isolation_forest(X_train, contamination=0.103, n_estimators=500):
    """
    Train the Isolation Forest model.
    """
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=2137)
    model.fit(X_train)
    return model


@measure_time
def detect_anomalies(data, model, scaler):
    """
    Detect anomalies using the trained model.
    """
    data_for_prediction = data.drop(columns=['AeT', 'AnT'])
    data['anomaly'] = np.where(model.predict(scaler.transform(data_for_prediction)) == -1, 1, 0)
    data['anomaly_label'] = data['anomaly'].map({1: 'Anomaly', 0: 'Normal'})
    return data


def plot_anomalies(data, feature_x, feature_y):
    """
    Visualize anomalies using a scatter plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=data,
        x=feature_x,
        y=feature_y,
        hue='anomaly_label',
        palette={'Normal': 'blue', 'Anomaly': 'red'},
        alpha=0.7
    )
    plt.title("Wykrywanie anomalii - Isolation Forest", fontsize=14)
    plt.xlabel(feature_x, fontsize=12)
    plt.ylabel(feature_y, fontsize=12)
    plt.legend(title='Typ')
    plt.grid(True)
    plt.show()


@measure_time
def evaluate_model_with_labels(data, model, scaler, ground_truth_col='ground_truth'):
    """
    Ocena modelu Isolation Forest za pomocą klasycznych metryk, gdy dostępne są prawdziwe etykiety.
    """
    data_for_prediction = data.drop(columns=['AeT', 'AnT', 'anomaly', 'anomaly_label'], errors='ignore')
    predictions = model.predict(scaler.transform(data_for_prediction))
    predictions = np.where(predictions == -1, 1, 0)  # Konwersja -1 na 1 (anomalie) i 1 na 0 (normalne)

    ground_truth = data[ground_truth_col]
    print("Confusion Matrix:")
    print(confusion_matrix(ground_truth, predictions))

    print("\nClassification Report:")
    print(classification_report(ground_truth, predictions))


@measure_time
def evaluate_model_without_labels(data):
    """
    Ocena modelu bez etykiet: analiza proporcji anomalii i ich rozkładu.
    """
    num_anomalies = data['anomaly'].sum()
    total_observations = len(data)
    anomaly_ratio = num_anomalies / total_observations

    #Już raz wyświetlone te metryki
    #print("Liczba anomalii:", num_anomalies)
    #print("Liczba normalnych obserwacji:", total_observations - num_anomalies)
    #print("Proporcja anomalii w danych: {:.2%}".format(anomaly_ratio))

    # Wizualizacja histogramu proporcji
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x='anomaly_label', palette={'Normal': 'blue', 'Anomaly': 'red'})
    plt.title("Rozkład normalnych i anomalii")
    plt.xlabel("Typ")
    plt.ylabel("Liczba obserwacji")
    plt.grid(True, axis='y')
    plt.show()


file_path = 'journal.pone.0309427.s001.csv'

# Load and inspect data
data = load_and_inspect_data(file_path)

# Clean and preprocess the data
data = clean_data(data)

# Prepare data for model training
X_train, X_test, y_train, y_test, scaler = prepare_data(data)

# Train Isolation Forest
iso_forest = train_isolation_forest(X_train)

# Detect anomalies
data = detect_anomalies(data, iso_forest, scaler)

# Display summary
print("\nPodsumowanie wyników:")
print(f"Liczba próbek: {len(data)}")
print(f"Liczba anomalii: {data['anomaly'].sum()}")
print(f"Liczba normalnych: {len(data) - data['anomaly'].sum()}")
print(f"Odsetek anomalii: {data['anomaly'].mean() * 100:.2f}%")
print(f"Odsetek normalnych: {(1 - data['anomaly'].mean()) * 100:.2f}%")

# Optionally display anomalies
while True:
    show_anomalies = input("Czy chcesz zobaczyć wykryte anomalie? (Y/N): ").strip().upper()
    if show_anomalies == 'Y':
        anomalies = data[data['anomaly'] == 1]
        print("Wykryte anomalie:")
        print(anomalies)
        break
    elif show_anomalies == 'N':
        print("Anomalie nie zostaną wyświetlone.")
        break
    else:
        print("Nieprawidłowa odpowiedź. Wpisz 'Y' lub 'N'.")

# Visualize anomalies
plot_anomalies(data, feature_x='vo2max', feature_y='hrmax')

# Evaluate model without labels
evaluate_model_without_labels(data)

# If ground truth labels are available, use the evaluation with labels
if 'ground_truth' in data.columns:
    evaluate_model_with_labels(data, iso_forest, scaler, ground_truth_col='ground_truth')