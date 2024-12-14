# Machine Learning Model for Disease Prediction

## Overview

This project implements a machine learning model for predicting disease outcomes using the **Random Forest Classifier**. The process includes preprocessing the data and training the model to handle class imbalances by adjusting class weights. 

## Files

- `Preprocess_Data.py`: Contains the data preprocessing steps (handling missing values, encoding categorical features, scaling features, etc.).
- `ML_Model.py`: Contains the code for training the machine learning model using the Random Forest Classifier and evaluating its performance.
- `Patients Data ( Used for Heart Disease Prediction ).csv`: The dataset used for training and testing the model.
- `requirements.txt`: A list of required Python packages for the project.
- `README.md`: This documentation file.

## Setup

To set up the project environment, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/ideniaa/HealthConditionsPatterns.git
   ```
   
2. Navigate to the project directory:
   ```
   cd HealthConditionsPatterns
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Preprocessing the Data

Before training the model, run the `Preprocess_Data.py` script to prepare the data. This step handles missing values, encodes categorical variables, scales the data, and splits it into training and testing sets.

Run the preprocessing script:
```bash
python Preprocess_Data.py
```

This will save the preprocessed data (`X_train_scaled`, `y_train`) to be used in the next step.

### 2. Train the Model

After preprocessing, run the `ML_Model.py` script to train the **Random Forest Classifier**. The model will be trained on the preprocessed data, with class weights applied to address class imbalance.

Run the model training script:
```bash
python ML_Model.py
```

This will output the model’s performance metrics, such as accuracy, precision, recall, and F1-score.

### Example Code

**Preprocessing (`Preprocess_Data.py`)**:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the data
data = pd.read_csv('Patients Data ( Used for Heart Disease Prediction ).csv')

# Encode categorical variables
encoder = LabelEncoder()
data['encoded_disease'] = encoder.fit_transform(data['disease'])

# Handle missing values, scale data, and split into train and test sets
data.fillna(method='ffill', inplace=True)
X = data.drop('encoded_disease', axis=1)
y = data['encoded_disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data
# (You can optionally save these to files or pass them directly to the model script)
```

**Model Training (`ML_Model.py`)**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# Compute class weights using the labels from the training data
class_weights = compute_class_weight('balanced', classes=y_train.unique(), y=y_train)
class_weight_dict = dict(zip(y_train.unique(), class_weights))

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

## Results

After running the model training script, you'll receive evaluation metrics, including accuracy, precision, recall, and F1-score. 

Example output:
```
Accuracy: 0.98
Macro Avg:
    Precision: 0.64
    Recall: 0.66
    F1-score: 0.64
Weighted Avg:
    Precision: 0.98
    Recall: 0.98
    F1-score: 0.98
```

## Dependencies

The following Python libraries are required for this project:
- `pandas`
- `scikit-learn`
- `numpy`
- `matplotlib` (for visualization, if needed)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
