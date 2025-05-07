import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pickle
import warnings

warnings.filterwarnings('ignore')

class FinalPricingRecommender:
    def __init__(self, model_path=None):
        self.model = None
        self.preprocessor = None
        if model_path:
            self.load_model(model_path)

    def load_data(self, filepath):
        """Load data with robust parsing"""
        try:
            df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
            print("Data loaded successfully")
            return self._clean_data(df)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def _clean_data(self, df):
        """Enhanced data cleaning"""
        # Convert price
        df['price'] = pd.to_numeric(
            df['price'].astype(str).str.replace('[\$,]', '', regex=True),
            errors='coerce'
        )
        df = df[df['price'].notna() & (df['price'] > 0)]

        # Create target variable
        df['log_price'] = np.log1p(df['price'])

        # Feature engineering
        df['amenities_count'] = df['amenities'].apply(
            lambda x: len(eval(x)) if pd.notna(x) and isinstance(x, str) else 0
        )

        # Date features
        if 'host_since' in df.columns:
            df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
            df['host_days'] = (pd.Timestamp.now() - df['host_since']).dt.days
            df['host_days'] = df['host_days'].fillna(df['host_days'].median())

        return df

    def prepare_model_data(self, df):
        """Prepare features and target"""
        # Select features
        numeric_features = [
            'accommodates', 'bedrooms', 'amenities_count',
            'minimum_nights', 'review_scores_rating', 'review_scores_accuracy',
            'review_scores_cleanliness', 'review_scores_checkin',
            'review_scores_communication', 'review_scores_location',
            'review_scores_value'
        ]

        categorical_features = [
            'neighbourhood', 'property_type', 'room_type'
        ]

        # Only use existing columns
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]

        # Select data
        X = df[numeric_features + categorical_features]
        y = df['log_price']

        return X, y, numeric_features, categorical_features

    def train_model(self, X, y, numeric_features, categorical_features):
        """Train model with proper preprocessing"""
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ))
        ])

        print("Training model...")
        pipeline.fit(X_train, y_train)

        # Store model and feature names
        self.model = pipeline
        self.feature_names = (
            numeric_features +
            list(pipeline.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names_out(categorical_features))
        )

        # Evaluate
        predictions = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"\nModel RMSE: {rmse:.4f}")

        return rmse

    def save_model(self, filename):
        """Save model with preprocessor"""
        save_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

    def load_model(self, filename):
        """Load saved model"""
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)
        self.model = save_data['model']
        self.preprocessor = save_data['preprocessor']
        self.feature_names = save_data['feature_names']

    def suggest_price(self, input_data):
        """Make prediction with input validation"""
        try:
            # Create input DataFrame with all expected features
            input_df = pd.DataFrame([input_data])

            # Predict
            log_price = self.model.predict(input_df)[0]
            return round(np.expm1(log_price), 2)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

# Example usage
if __name__ == "__main__":
    print("Initializing final recommender...")
    recommender = FinalPricingRecommender()

    print("Loading data...")
    df = recommender.load_data('/content/airbnb_listing.csv')

    if df is not None:
        print("Preparing model data...")
        X, y, numeric_features, categorical_features = recommender.prepare_model_data(df)

        print("Training model...")
        rmse = recommender.train_model(X, y, numeric_features, categorical_features)
        recommender.save_model('final_airbnb_model.pkl')

        # Sample prediction
        sample_listing = {
            'accommodates': 2,
            'bedrooms': 1,
            'amenities_count': 3,
            'minimum_nights': 2,
            'review_scores_rating': 95,
            'review_scores_accuracy': 9,
            'review_scores_cleanliness': 9,
            'review_scores_checkin': 10,
            'review_scores_communication': 10,
            'review_scores_location': 10,
            'review_scores_value': 9,
            'neighbourhood': 'Buttes-Montmartre',
            'property_type': 'Entire apartment',
            'room_type': 'Entire place'
        }

        print("\nMaking prediction...")
        price = recommender.suggest_price(sample_listing)
        print(f"\nSuggested price: ${price} per night")