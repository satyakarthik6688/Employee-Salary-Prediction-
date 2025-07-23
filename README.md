# Employee-Salary-Prediction-
Employee Salary Prediction using machine learning
Salary Prediction Project using SVM
This machine learning project predicts whether an individual's salary is above or below a certain threshold based on demographic and job-related features. It uses a Support Vector Machine (SVM) classifier and a custom synthetic dataset.

Dataset
The dataset contains the following features:

age (numerical)

education (categorical: Bachelors, Masters, PhD)

department (categorical: Sales, Engineering, HR, Marketing)

experience (numerical, in years)

salary (target: 0 = ≤50K, 1 = >50K)

 Objective
To build a machine learning pipeline that:

Preprocesses categorical and numerical data

Trains a Support Vector Machine (SVM) model

Predicts salary classification

Exports predictions to a CSV file
Tech Stack
Python 3.9+

scikit-learn

pandas

joblib

🧠 Algorithm Used
✅ Support Vector Machine (SVM)
Kernel: RBF (Radial Basis Function)

γ (gamma): scale

C (regularization): 1.0

SVM is used due to its effectiveness on small to medium-sized datasets with non-linear decision boundaries.

🛠️ How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/salary-prediction-svm.git
cd salary-prediction-svm
Create a virtual environment:

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
Run the script:

bash
Copy
Edit
python salary_prediction.py
 Output Files
model.pkl – Trained SVM model

predicted_salaries.csv – File containing actual vs predicted labels on test data

 Example Output
yaml
Copy
Edit
Model Accuracy: 0.85

Classification Report:
               precision    recall  f1-score   support
           0       0.88      0.84      0.86        25
           1       0.83      0.88      0.85        25
🙋 Custom Input Testing
To test your own inputs:

python
Copy
Edit
import joblib
import pandas as pd

model = joblib.load("model.pkl")
custom_input = pd.DataFrame([{
    'age': 35,
    'education': 'Masters',
    'department': 'Engineering',
    'experience': 8
}])
prediction = model.predict(custom_input)
print("Predicted Salary Class:", prediction[0])

Saves the trained model for later use
