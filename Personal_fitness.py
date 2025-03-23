import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Page Title
st.title("DDoS Attack Prediction using Ensemble Learning")
st.write("Upload your dataset and train an ensemble model to predict DDoS attacks.")

# Sidebar for Uploading Dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load Dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Preprocessing
    st.write("### Data Preprocessing")
    
    # Check for missing values
    if st.checkbox("Show missing values"):
        st.write(data.isnull().sum())

    # Drop missing values
    data = data.dropna()

    # Select Features and Target
    st.write("### Select Features and Target")
    features = st.multiselect("Select features", data.columns[:-1], default=data.columns[:-1].tolist())
    target = st.selectbox("Select target column", data.columns, index=len(data.columns) - 1)

    X = data[features]
    y = data[target]

    # Encode Categorical Data
    st.write("### Encoding Categorical Data")
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.write(f"Categorical columns detected: {list(categorical_cols)}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        st.write("Categorical columns have been one-hot encoded.")
        st.write(X.head())

    # Encode Target Variable (if necessary)
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        st.write("Target variable has been label encoded.")

    # Split Data
    st.write("### Train-Test Split")
    test_size = st.slider("Select test size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model Selection
    st.write("### Model Selection")
    st.write("Choose ensemble models to train:")
    rf = st.checkbox("Random Forest", value=True)
    xgb = st.checkbox("XGBoost", value=True)
    gbm = st.checkbox("Gradient Boosting", value=True)

    # Train Models
    if st.button("Train Model"):
        st.write("### Training Models...")
        estimators = []

        if rf:
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            estimators.append(('Random Forest', rf_model))
            st.write("Random Forest trained!")

        if xgb:
            xgb_model = XGBClassifier(n_estimators=100, random_state=42)
            xgb_model.fit(X_train, y_train)
            estimators.append(('XGBoost', xgb_model))
            st.write("XGBoost trained!")

        if gbm:
            gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gbm_model.fit(X_train, y_train)
            estimators.append(('Gradient Boosting', gbm_model))
            st.write("Gradient Boosting trained!")

        # Ensemble Model (Voting)
        if len(estimators) > 0:
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            ensemble.fit(X_train, y_train)
            st.write("Ensemble model trained!")

            # Evaluate Model
            st.write("### Model Evaluation")
            y_pred = ensemble.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Save Model
            joblib.dump(ensemble, "ddos_ensemble_model.pkl")
            st.write("Model saved as `ddos_ensemble_model.pkl`.")

            # Download Model
            with open("ddos_ensemble_model.pkl", "rb") as f:
                st.download_button("Download Model", f, file_name="ddos_ensemble_model.pkl")

else:
    st.write("Please upload a dataset to get started.")
