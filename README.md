Online Course Completion Prediction

This project uses machine learning to predict whether a student will complete an online course based on their demographics, engagement metrics, and activity data.
The goal is to help identify at-risk students early, enabling targeted interventions by educators.


---

📌 Project Overview

Online learning platforms generate rich data about learners — from time spent on the platform to assignments submitted.
This project leverages that data to:

Predict course completion likelihood

Analyze factors influencing student engagement

Support educators in making data-driven interventions



---

🛠 Tech Stack

Language: Python 3.10+

Dependency Management: Poetry

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Modeling: Logistic Regression, Random Forest, Gradient Boosting

Environment: Jupyter Notebook



---

📂 Project Structure

Online-Course-Completion-ML/
├── notebooks/
│   └── Online_Course_Completion_Prediction.ipynb   # Main analysis & modeling notebook
├── pyproject.toml                                   # Poetry project configuration
├── poetry.lock                                      # Poetry dependency lock file
├── README.md                                        # Project documentation

(Dataset not included in repository.)


---

🚀 Features

Data Cleaning & Preprocessing – Handle missing values, encode categorical variables

Feature Engineering – Create new variables like engagement score

Scaling – Standardization of numerical features

Modeling – Logistic Regression (baseline), Random Forest, Gradient Boosting

Evaluation – Accuracy, confusion matrix, classification report



---

📊 Model Performance (Example)

Model	Accuracy

Logistic Regression	0.85
Random Forest	0.90
Gradient Boosting	0.92


(Values are illustrative — see notebook for actual results.)


---

📜 How to Run

1. Clone the repository

git clone https://github.com/SahanaCodes27/Online-Course-Completion-ML.git
cd Online-Course-Completion-ML


2. Install dependencies with Poetry

poetry install


3. Open the Jupyter Notebook

poetry run jupyter notebook notebooks/Online_Course_Completion_Prediction.ipynb


4. Update dataset path in Step 2 of the notebook with your local .csv file.
