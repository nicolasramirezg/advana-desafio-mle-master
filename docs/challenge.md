# Challenge – Software Engineer (ML & LLMs)

## 1. Final Model

For the final implementation I used a **Logistic Regression** with class balancing, trained on the **10 most relevant features** from the notebook.  
I chose this model because it offers very similar performance to XGBoost but is simpler, faster to train, and easier to maintain in production.

Balancing the classes significantly improved recall for the minority class (`delay = 1`), which is operationally more important than overall accuracy, since the main goal is to detect potential delays.

---

## 2. Design and Implementation

The code is organized in a single class called `DelayModel`, following a clean and modular structure.  
It includes three main methods:

- **`preprocess()`**: creates and transforms the columns needed for the model (`high_season`, `min_diff`, `period_day`), applies one-hot encoding, and filters the top 10 features.  
- **`fit()`**: trains the Logistic Regression model using dynamic class weights.  
- **`predict()`**: handles unseen categories safely by reindexing columns and predicting over the consistent schema.

This setup allows the same code to be used both for training and for serving predictions through an API.

---

## 3. Top 10 Features Used

OPERA_Latin American Wings
MES_7
MES_10
OPERA_Grupo LATAM
MES_12
TIPOVUELO_I
MES_4
MES_11
OPERA_Sky Airline
OPERA_Copa Air


---

## 4. Production-Ready Considerations

- The preprocessing handles missing or malformed dates with safe fallbacks.
- New unseen categories are automatically filled with zeroes (`reindex` safeguard).
- Random seed fixed (`42`) for reproducibility.
- The model can still predict even if `fit()` was not called (for robustness in API contexts).
- All tests (`make model-test`) pass successfully with **~90 % coverage**.

## 5. Testing ci/cd deployment comment
## Testing 2