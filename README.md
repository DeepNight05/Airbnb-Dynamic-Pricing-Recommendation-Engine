Final Project: Airbnb Dynamic Pricing Recommendation Engine
# Airbnb Dynamic Pricing Recommendation Engine

## Introduction
In the competitive short-term rental market, Airbnb hosts must price their listings optimally to maximize occupancy and revenue while maintaining customer satisfaction. This project aims to develop a **dynamic pricing recommendation engine** using machine learning, supported by an interactive **Tableau dashboard** for visual exploration of key pricing factors.

## Abstract
This end-to-end project integrates **Python**, **Excel**, and **Tableau** to analyze historical Airbnb data and recommend optimal pricing based on various factors such as location, seasonality, and listing attributes. We use a **Random Forest Regressor** trained on curated features like room type, property type, number of amenities, and review scores. To handle skewed pricing data, price predictions are logarithmically scaled. The results are presented via a **user-friendly Tableau dashboard**, allowing hosts to explore trends and fine-tune pricing using filters and sliders.

## Tools & Technologies Used
- **Python**: Data preprocessing, feature engineering, machine learning model training (`scikit-learn`), and predictions.
- **Excel**: Light data cleaning and inspection.
- **Tableau**: Interactive dashboard for price insights.

## Steps Involved

### 1. Data Collection & Preprocessing
- Loaded CSV dataset using `pandas` with encoding and formatting corrections.
- Cleaned and transformed price columns using a **logarithmic scale**.
- Engineered features such as:
  - Host experience (days active)
  - Number of amenities
  - Review scores
- Handled missing values with appropriate imputation techniques.
- Applied one-hot encoding for categorical variables.

### 2. Model Building
- Trained a **RandomForestRegressor** on selected numeric and categorical features.
- Used `Pipeline` and `ColumnTransformer` from `scikit-learn` for:
  - Scaling
  - Encoding
  - Imputation
- Evaluated performance using **Root Mean Squared Error (RMSE)**.

### 3. Price Suggestion Engine
- Implemented a class `FinalPricingRecommender` that handles:
  - Data loading
  - Preprocessing
  - Model training and saving
  - Predicting optimal prices for sample listings

### 4. Dashboard Development
- Exported prediction results and key attributes (city, property_type, review_scores_rating, etc.) to Excel/CSV.
- Developed a Tableau dashboard with:
  - Filters for city, property type, and review scores
  - Charts for:
    - Average price by city
    - High-priced property types
    - Review score vs price
    - Map view of listings

## Dashboard Preview
> A screenshot or link to the Tableau dashboard can be added here.

## Conclusion
This project successfully demonstrates the power of combining machine learning with interactive data visualization to drive data-backed pricing decisions in the Airbnb marketplace. The Tableau dashboard provides transparency and empowers hosts to fine-tune their strategies based on real insights.
