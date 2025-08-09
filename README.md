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

Modeling: Logistic Regression, Decision Tree, Gradient Boosting

Environment: Jupyter Notebook, CLI script



---

📂 Project Structure

Online-Course-Completion-ML/
├── notebooks/
│   └── Online_Course_Completion_Prediction.ipynb   # Main analysis & modeling notebook
├── models/                                         # Saved ML models
├── train_model.py                                  # CLI script for training & saving models
├── pyproject.toml                                  # Poetry project configuration
├── poetry.lock                                     # Poetry dependency lock file
├── README.md                                       # Project documentation

> Note: Dataset is not included in the repository.




---

🚀 Features

Data Cleaning & Preprocessing – Handle missing values, encode categorical variables

Feature Engineering – Create engagement-related features

Scaling – Standardization of numerical features

Modeling – Logistic Regression (baseline), Decision Tree, Gradient Boosting

Evaluation – Accuracy, confusion matrix, classification report

CLI Training Script – Train & save models without opening Jupyter



---

📜 How to Run

1️⃣ Clone the Repository

git clone https://github.com/SahanaCodes27/Online-Course-Completion-ML.git
cd Online-Course-Completion-ML

2️⃣ Install Dependencies

poetry install

3️⃣ Run Jupyter Notebook (Optional)

poetry run jupyter notebook notebooks/Online_Course_Completion_Prediction.ipynb

4️⃣ Run Training Script via CLI

python3 train_model.py --data online_course_completion.csv --model gradient_boosting

Arguments:

--data → Path to CSV dataset

--model → Model to train (logistic_regression, decision_tree, gradient_boosting)


Example:

python3 train_model.py --data online_course_completion.csv --model gradient_boosting

Output:

Prints accuracy and top 10 important features

Saves trained model to models/ directory



---

📊 Model Performance (Example)

Model	Accuracy

Logistic Regression	0.85
Decision Tree	0.90
Gradient Boosting	0.96


(Values are illustrative — see actual script output.)


---

✍️Author

Sahana
