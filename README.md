# DineScope — Restaurant Intelligence

Streamlit dashboard for the supplied Zomato restaurant dataset. DineScope includes preprocessing, EDA, four rating regressors, four price category classifiers, a test-set model comparison, and interactive predictions. The left navigation includes Dashboard, Visual Analytics, Rating Prediction, Price Tier Prediction, and Model Comparison.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep `Zomato Restaurant Dataset.csv` beside `app.py`.

## Project workflow

1. **Cleaning and preprocessing:** remove duplicate rows, fill missing categorical values, coerce numeric columns, impute any remaining input gaps in the pipelines, and remove records missing target values. Rating regression excludes records whose aggregate rating is 0 (`Not rated`).
2. **EDA:** review row counts, rating and price range distributions, top cities, missingness, and cleaned data preview.
3. **Rating regression:** Linear Regression, Ridge Regression, Decision Tree Regressor, and Random Forest Regressor predict `Aggregate rating`.
4. **Price category classification:** Logistic Regression, Decision Tree Classifier (Gini), Random Forest Classifier, and K-Nearest Neighbors predict `Price range` (1–4).
5. **Model comparison:** rating regressors use MAE, RMSE, and R²; price classifiers use accuracy, weighted precision, weighted recall, and weighted F1 on a fixed 80/20 holdout split.
6. **Website:** Streamlit tabs present the overview, EDA, both prediction forms, and model comparison. Prediction inputs appear in a horizontal panel above the tabs; the app uses a light, high-contrast theme from `.streamlit/config.toml`.

Restaurant IDs, names, addresses, aggregate rating labels, and other post-outcome fields are not model inputs. Cost is recorded across multiple countries and currencies, so interpret it only with its currency and as a dataset estimate.

After changing the app or theme, stop the existing server with **Ctrl+C** and restart it with `python -m streamlit run app.py` so Streamlit reloads the light theme.
