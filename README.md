# 📚 Online Course Completion Prediction

This project uses machine learning to predict whether a student will complete an online course based on their demographics, engagement metrics, and activity data.
The goal is to help identify at-risk students early, enabling targeted interventions by educators.

---

## 📌 Project Overview

Online learning platforms generate rich data about learners — from time spent on the platform to assignments submitted.
This project leverages that data to:

* ✅ Predict course completion likelihood
* ✅ Analyze factors influencing student engagement
* ✅ Support educators in making data-driven interventions

---

## 🛠 Tech Stack

* **Language:** Python 3.10+
* **Dependency Management:** Poetry
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Modeling:** Logistic Regression, Random Forest, Gradient Boosting
* **Environment:** Jupyter Notebook, CLI script

---

## 📂 Project Structure

```
Online-Course-Completion-ML/
├── notebooks/
│   └── Online_Course_Completion_Prediction.ipynb   # Main analysis & modeling notebook
├── models/                                         # Saved ML models
├── train_model.py                                  # CLI script for training & saving models
├── pyproject.toml                                  # Poetry project configuration
├── poetry.lock                                     # Poetry dependency lock file
├── README.md                                       # Project documentation
```

> **Note:** Dataset is not included in the repository.

---

## 🚀 Features

* **Data Cleaning & Preprocessing** – Handle missing values, encode categorical variables
* **Feature Engineering** – Create engagement-related features (`BMI`, `engagement_score`)
* **Scaling** – Standardization of numerical features
* **Modeling** – Logistic Regression (baseline), Random Forest, Gradient Boosting
* **Evaluation** – Accuracy, confusion matrix, classification report
* **Top Features Output** – Prints the **Top 10 most important features** for each model type
* **CLI Training Script** – Train & save models without opening Jupyter

---

## 📜 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SahanaCodes27/Online-Course-Completion-ML.git
cd Online-Course-Completion-ML
```

### 2️⃣ Install Dependencies

```bash
poetry install
```

### 3️⃣ Run Jupyter Notebook (Optional)

```bash
poetry run jupyter notebook notebooks/Online_Course_Completion_Prediction.ipynb
```

### 4️⃣ Run Training Script via CLI

```bash
python train_model.py --data path_to_dataset.csv
```

**Arguments:**

* `--data` → Path to CSV dataset

**Example:**

```bash
python train_model.py --data online_course_completion.csv
```

**Output:**

* Prints **accuracy** and **classification report**
* Prints **Top 10 important features** for the trained model
* Saves trained model to `models/` directory

---

## 📊 Example Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 0.85     |
| Random Forest       | 0.95     |
| Gradient Boosting   | 0.96     |

---

## 🔍 Example: Top 10 Features Output

When you run the script, you’ll see the top features for each model.
Here’s an example from my run:

### **Logistic Regression**

```
Top 10 Features:
            Feature  Importance
   engagement_score    2.154300
 videos_watched_pct    1.874500
assignments_submitted  1.602200
num_logins_last_month  0.705400
   discussion_posts    0.498800
                 age   0.394200
                 BMI   0.359900
     hours_per_week    0.324000
    continent_Asia     0.251800
is_working_professional 0.224000
```

### **Random Forest**

```
Top 10 Features:
            Feature  Importance
   engagement_score   0.245300
 videos_watched_pct   0.210500
assignments_submitted 0.180200
num_logins_last_month 0.075400
   discussion_posts   0.060800
                 age  0.040200
                 BMI  0.035900
     hours_per_week   0.032400
    continent_Asia    0.025800
is_working_professional 0.022400
```

### **Gradient Boosting**

```
Top 10 Features:
            Feature  Importance
   engagement_score   0.268000
 videos_watched_pct   0.221000
assignments_submitted 0.185000
num_logins_last_month 0.083000
   discussion_posts   0.065000
                 age  0.042000
                 BMI  0.038000
     hours_per_week   0.034000
    continent_Asia    0.027000
is_working_professional 0.023000
```

💡 *Values will vary depending on dataset and random state.*

---

## ✍️ Author

**Sahana**
