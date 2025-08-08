# Online Course Completion Prediction 🎓

## 📌 Project Overview
This project applies *Machine Learning* to predict whether a learner will complete an online course based on demographic, engagement, and activity data.

We built and compared three models:
- *Logistic Regression*
- *Random Forest*
- *Gradient Boosting*

The best model is chosen based on accuracy and other performance metrics.

---

## 📊 Dataset
- *File:* online_course_completion.csv  
- *Target Variable:* completed_course (1 = completed, 0 = not completed)  
- *Features:*  
  - age  
  - continent  
  - education_level  
  - hours_per_week  
  - num_logins_last_month  
  - videos_watched_pct  
  - assignments_submitted  
  - discussion_posts  
  - is_working_professional  
  - preferred_device  
  - *BMI* (engineered)  
  - *engagement_score* (engineered)

---

## ⚙ Steps Performed
1. *Data Loading & Exploration* – Basic inspection, datatype checks, missing values.  
2. *Data Cleaning* – Filling missing values with median/mode.  
3. *Feature Encoding* – One-hot encoding for categorical variables.  
4. *Feature Engineering* – Added BMI & engagement score.  
5. *Feature Scaling* – Standardization with StandardScaler.  
6. *Model Training* – Logistic Regression, Random Forest, Gradient Boosting.  
7. *Evaluation* – Accuracy, classification report, confusion matrix.  
8. *Model Comparison* – Identify the best performing model.

---

## 🏆 Model Performance

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | XX.XX%   |
| Random Forest       | XX.XX%   |
| Gradient Boosting   | XX.XX%   |

(Replace XX.XX% with your actual results from the notebook)

---

## 📂 Repository Structure
├── notebooks/ │   └── Online_Course_Completion_Prediction.ipynb ├── online_course_completion.csv ├── README.md └── requirements.txt

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/SahanaCodes27/Online-Course-Completion-ML.git

# Go into the project directory
cd Online-Course-Completion-ML

# Install dependencies
pip install -r requirements.txt

# Open Jupyter Notebook
jupyter notebook

---

🛠 Technologies Used

Python

Pandas, NumPy

Scikit-Learn

Matplotlib, Seaborn



---

✨ Author

Sahana
