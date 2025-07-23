Employee Salary Prediction using Random Forest Classifier
This project predicts whether an employee earns more than 50K or less than or equal to 50K based on their demographic and work-related features. It uses the Random Forest Classifier (RFC), a robust ensemble machine learning algorithm.

 Dataset
The dataset is a synthetic collection of employee data, designed to resemble the structure of real-world HR datasets.
Features:
age (numerical)

education (categorical: Bachelors, Masters, PhD, etc.)

department (categorical: Sales, Engineering, HR, Marketing, etc.)

experience (numerical, in years)

salary (target: 0 = ≤50K, 1 = >50K)

 Objective
Build a machine learning model to predict employee salary class

Use feature preprocessing to handle categorical and numerical features

Evaluate model performance using accuracy and classification metrics

Export predictions and save the trained model for future use

 Algorithm Used
 Random Forest Classifier
An ensemble of decision trees

Robust to overfitting

Handles both numerical and categorical features

Used as a reliable baseline for classification tasks

⚙️ Technologies
Python 3.9+

scikit-learn

pandas

joblib

 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/salary-prediction-rfc.git
cd salary-prediction-rfc
Create a virtual environment and activate it:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # on Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the model training script:

bash
Copy
Edit
python salary_prediction.py

 Output
model.pkl – Trained Random Forest model saved using joblib

predicted_salaries.csv – Contains actual and predicted salary classifications for test data

 Example Output
yaml
Copy
Edit
Model Accuracy: 0.87

Classification Report:
               precision    recall  f1-score   support
           0       0.85      0.90      0.87        30
           1       0.89      0.83      0.86        30
 Custom Prediction Script
Create a file named predict_custom.py:

python
Copy
Edit
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("model.pkl")

# Define custom input
custom_input = pd.DataFrame([{
    "age": 35,
    "education": "Masters",
    "department": "Engineering",
    "experience": 8
}])

# Predict salary class
prediction = model.predict(custom_input)[0]
print("Predicted Salary Class:", ">50K" if prediction == 1 else "<=50K")

