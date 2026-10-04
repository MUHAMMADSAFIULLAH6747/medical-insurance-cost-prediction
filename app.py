"""Streamlit app: Medical Insurance Cost Prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

from src.preprocess import encode_single_input

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "insurance_cost_model.joblib"


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python src/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def main() -> None:
    st.set_page_config(
        page_title="Medical Insurance Cost Prediction",
        page_icon="💊",
        layout="centered",
    )

    st.title("Medical Insurance Cost Prediction")
    st.markdown(
        "Estimate annual medical insurance charges from age, BMI, smoking status, "
        "dependents, gender, and U.S. region using a trained **Multiple Linear Regression** model."
    )

    bundle = load_bundle()
    model = bundle["model"]
    metrics = bundle.get("metrics", {})

    with st.sidebar:
        st.header("Model Performance")
        if metrics:
            st.metric("Test R²", f"{metrics['r2_test'] * 100:.2f}%")
            st.metric("MAE", f"${metrics['mae']:,.2f}")
            st.metric("RMSE", f"${metrics['rmse']:,.2f}")
        st.caption("Trained on the Kaggle Medical Cost Personal Dataset (1,337 clean rows).")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=64, value=30)
        bmi = st.slider("BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
        children = st.slider("Number of Children", min_value=0, max_value=5, value=0)
    with col2:
        sex = st.selectbox("Gender", options=["male", "female"])
        smoker = st.selectbox("Smoker", options=["no", "yes"])
        region = st.selectbox(
            "Region",
            options=["northeast", "northwest", "southeast", "southwest"],
        )

    predict = st.button("Predict Insurance Cost", type="primary", use_container_width=True)

    if predict:
        features = encode_single_input(age, sex, bmi, children, smoker, region)
        prediction = float(model.predict(features)[0])
        prediction = max(prediction, 0.0)

        st.success(f"Estimated Insurance Cost: **${prediction:,.2f}**")
        st.write("### Input Summary")
        st.json(
            {
                "age": age,
                "bmi": bmi,
                "children": children,
                "gender": sex,
                "smoker": smoker,
                "region": region,
                "predicted_charges_usd": round(prediction, 2),
            }
        )

        if smoker == "yes":
            st.info(
                "Smoking status is the strongest cost driver in this model "
                "(coefficient ≈ +$23,078)."
            )

    with st.expander("How this works"):
        st.markdown(
            """
1. User inputs are encoded to match training features (`sex`, `smoker`, region one-hot).
2. The saved Linear Regression model returns continuous USD charges.
3. Metrics shown in the sidebar come from an 80/20 train-test split (`random_state=42`).
            """
        )


if __name__ == "__main__":
    main()
