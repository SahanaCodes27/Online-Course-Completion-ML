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
        self.feature_names = []

        # Always save inside repo's models folder
        self.repo_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.repo_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

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

        # One-hot encode categorical features
        X = pd.get_dummies(X, columns=['continent', 'education_level', 'preferred_device'], drop_first=True)

        # Save feature names
        self.feature_names = X.columns.tolist()

        # Fill missing values
        X.fillna(X.median(), inplace=True)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.load_data()

        # Select model
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

        self.print_top_features()

    def print_top_features(self):
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = abs(self.model.coef_[0])
        else:
            print("\nThis model does not support feature importance.")
            return

        feature_importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print("\nTop 10 Features:")
        print(feature_importance_df.head(10).to_string(index=False))

    def save_model(self):
        model_path = os.path.join(self.models_dir, f"{self.model_type}.pkl")
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
