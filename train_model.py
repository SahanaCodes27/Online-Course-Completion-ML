import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

class TrainModel:
    def __init__(self, data_path, model_type):
        self.data_path = data_path
        self.model_type = model_type
        self.model = None
        self.feature_names = None

    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        target_column = "completed_course"
        X = df.drop(columns=[target_column])
        y = df[target_column]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def choose_model(self):
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier()
        elif self.model_type == "hist_gradient_boosting":
            self.model = HistGradientBoostingClassifier()
        else:
            raise ValueError("Unsupported model type")

    def train(self):
        X_train, X_test, y_train, y_test = self.load_data()

        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        self.choose_model()

        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', self.model)
        ])

        print(f"Training model: {self.model_type}...")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{self.model_type}_model.pkl"
        joblib.dump(clf, model_path)
        print(f"Model saved to {model_path}")

        # Feature importances
        self.print_feature_importances(preprocessor, clf.named_steps['classifier'], numeric_features, categorical_features)

    def print_feature_importances(self, preprocessor, model, numeric_features, categorical_features):
        if hasattr(model, "feature_importances_"):
            # Get transformed feature names
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
            all_feature_names = np.concatenate([numeric_features, cat_feature_names])

            # Match importances with names
            importances = model.feature_importances_
            sorted_idx = importances.argsort()[::-1]

            print("\nTop 10 Most Important Features:")
            for idx in sorted_idx[:10]:
                print(f"{all_feature_names[idx]}: {importances[idx]:.4f}")
        else:
            print("\nModel does not support feature importances.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--model", required=True, choices=["gradient_boosting", "hist_gradient_boosting"], help="Model type")
    args = parser.parse_args()

    trainer = TrainModel(args.data, args.model)
    trainer.train()
