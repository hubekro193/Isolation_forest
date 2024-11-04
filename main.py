import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data
data = pd.read_csv('pone.0309427.s001.csv')

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

# Convert the date column
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['date'] = data['date'].map(lambda x: x.toordinal() if pd.notnull(x) else 0)

# Check which columns have NaN after conversion
print(data.isnull().sum())

# Prepare features and labels
X = data.drop(columns=['la_6', 'la_8', 'la_10', 'la_12', 'la_14', 'la_16', 'la_18', 'la_20', 'la_22'])  # Modify based on the goal
y = data['la_6']  # Example target, change as needed

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data is ready to be used with the machine learning algorithm.")
