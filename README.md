Online Course Completion Prediction

This project uses machine learning to predict whether a student will complete an online course based on their demographics, engagement metrics, and activity data. The model helps identify at-risk students early, enabling targeted interventions.

📌 Project Overview

Online education platforms collect a wide range of learner activity data — from time spent on the platform to the number of videos watched and assignments submitted. This project leverages that data to:

Predict course completion likelihood

Analyze key factors affecting student engagement

Provide actionable insights to educators and administrators


🛠 Tech Stack

Language: Python 3.x

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Modeling: Gradient Boosting, Logistic Regression, Random Forest

Environment: Jupyter Notebook


📂 Project Structure

├── notebooks/
│   └── Online_Course_Completion_Prediction.ipynb  # Main analysis & modeling notebook
├── README.md                                       # Project documentation

(Dataset not included in repository.)

🚀 Features

Data Cleaning & Preprocessing: Handles missing values and encodes categorical variables

Feature Engineering: Adds BMI and engagement score

Scaling: Standardization of numerical features

Modeling: Gradient Boosting Classifier for best performance

Evaluation: Accuracy, confusion matrix, classification report


📊 Model Performance (Example)

Model	Accuracy

Logistic Regression	0.85
Random Forest	0.90
Gradient Boosting	0.92


📜 How to Run

1. Clone the repository

   git clone https://github.com/SahanaCodes27/Online-Course-Completion-ML.git
   cd Online-Course-Completion-ML


2. Install dependencies

   pip install -r requirements.txt


3. Open the Jupyter Notebook

   jupyter notebook notebooks/Online_Course_Completion_Prediction.ipynb


4. Update dataset path in Step 2 of the notebook with your local .csv file.
