# insurance-charges-app

This interactive web app predicts personalized health insurance charges based on user demographics and lifestyle features. It also provides SHAP-based model explainability to help understand how each feature contributes to the expected cost.

 **Live App**: [nkume-insurance-predictor.streamlit.app](https://nkume-insurance-predictor.streamlit.app)

---

##  Features

- Predict charges using a trained Random Forest model
-  Input individual patient details for real-time predictions
-  Upload a CSV file for batch predictions
-  SHAP explanations:
  -  Local explanation (force plot) for individual predictions
  -  Global explanation (summary plot) for feature importance
-  Download predictions as CSV

---

##  Technologies Used

- [Python 3.x](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [SHAP](https://github.com/slundberg/shap)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Joblib](https://joblib.readthedocs.io/)

---

##  File Structure
 insurance-charges-app/
├── streamlit_app.py # Main Streamlit application
├── random_forest_model.pkl # Trained ML model
├── scaler.pkl # StandardScaler used during training
├── feature_columns.pkl # Feature order used for inference
├── requirements.txt # Python dependencies
└── README.md # Project documentation

Model Overview
Best model: Random Forest Regressor

Evaluation Metrics:

R² ≈ 0.85

RMSE ≈ 4600

MAE ≈ 2600

Author
Friday Nkume
2423902
GitHub: PFUN4

