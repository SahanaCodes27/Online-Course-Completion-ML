import argparse
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


class TrainModel:
    def __init__(self, data_path, model_type):
        self.data_path = data_path
        self.model_type = model_type
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.data_path)

        # Feature engineering
        df['BMI'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
        df['engagement_score'] = (
            df['videos_watched_pct'] +
            df['assignments_submitted'] +
            df['discussion_posts']
        )

        features = [
            'age', 'continent', 'education_level', 'hours_per_week',
            'num_logins_last_month', 'videos_watched_pct',
            'assignments_submitted', 'discussion_posts',
            'is_working_professional', 'preferred_device', 'BMI', 'engagement_score'
        ]
        X = df[features]
        y = df['completed_course']

        X = pd.get_dummies(X, columns=['continent', 'education_level', 'preferred_device'], drop_first=True)
        X.fillna(X.median(), inplace=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.load_data()

        if self.model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        print(f"\nTraining {self.model_type}...")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{self.model_type}.pkl"
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple machine learning models.")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset CSV file.")
    args = parser.parse_args()

    model_types = ["logistic_regression", "random_forest", "gradient_boosting"]

    for model_type in model_types:
        trainer = TrainModel(data_path=args.data, model_type=model_type)
        trainer.train()
        trainer.save_model()
