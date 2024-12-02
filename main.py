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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data
data = pd.read_csv('journal.pone.0309427.s001.csv')


#check how many empty rows
columns_to_check = ['hr_14', 'la_14', 'hr_16', 'la_16', 'hr_18', 'la_18', 'hr_20', 'la_20', 'hr_22', 'la_22']
empty_counts = data[columns_to_check].isnull().sum()
filled_counts = data[columns_to_check].notnull().sum()
print("Liczba pustych wartości w kolumnach:")
print(empty_counts)
print("\nLiczba wypełnionych wartości w kolumnach:")
print(filled_counts)
print("\nWartość % pustych pozycji w kolumnach:")
print( (empty_counts/len(data))*100 )

while True:
    answer = input("Czy chcesz kontynuować? (Y/N): ").strip().upper()  # Pobierz odpowiedź i przetwórz ją
    if answer == 'Y':
        print("Kontynuujemy program...")
        break  # Wyjdź z pętli, aby kontynuować
    elif answer == 'N':
        print("Kończymy program.")
        exit()  # Zakończ program
    else:
        print("Nieprawidłowa odpowiedź. Wpisz 'Y' lub 'N'.")

# Drop unnecessary columns
# hr_6 and la_6 - The study did not measure lactate at a speed of 6 km/h because the exercise test started at a speed of 8 km/h.
columns_to_drop = ['date', 'hr_6', 'la_6','hr_16', 'la_16', 'hr_18', 'la_18', 'hr_20', 'la_20', 'hr_22', 'la_22']
data.drop(columns_to_drop, axis=1, inplace=True)

# Fill in column 'la_6' with value 0
# - The study did not measure lactate at a speed of 6 km/h because the exercise test started at a speed of 8 km/h.
#   Probably no presence of lactic acid at a speed of 6 km/h
#data['la_6'] = 0


# Fill missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Transform text columns into numeric values
label_encoder = LabelEncoder()

# Transform the 'sex' column
if 'sex' in data.columns:
    data['sex'] = label_encoder.fit_transform(data['sex'])

# Transform the 'discipline' column
if 'discipline' in data.columns:
    data['discipline'] = label_encoder.fit_transform(data['discipline'])

# date column has been dropped
# # Convert the date column
# if 'date' in data.columns:
#     data['date'] = pd.to_datetime(data['date'], errors='coerce')
#     data['date'] = data['date'].map(lambda x: x.toordinal() if pd.notnull(x) else 0)

# Encode the 'rf' column if it exists
if 'rf' in data.columns:
    data['rf'] = label_encoder.fit_transform(data['rf'])

# Encode the 'rf' column if it exists
if 'rf' in data.columns:
    data['rf'] = label_encoder.fit_transform(data['rf'])

# Check which columns have NaN after conversion
print(data.isnull().sum())

# Prepare features and labels
X = data.drop(columns=['AeT', 'AnT'])  # Parameters on which the prediction is based
y = data[['AeT', 'AnT']]  # We are looking for aerobic (AeT) and anaerobic (AnT) thresholds

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data is ready to be used with the machine learning algorithm.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
