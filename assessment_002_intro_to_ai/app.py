# Bank Campaign Predictor - MVP web app
# run with:  streamlit run app.py

import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

DB_FILE = "bank.db"
THRESHOLD = 0.5

# months that exist in the training data (the bank made no calls in jan and feb)
MONTHS_IN_DATA = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTHS_NOT_IN_DATA = ["jan", "feb"]

st.set_page_config(page_title="Bank Campaign Predictor", page_icon="🏦")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            customer_name TEXT,
            age INTEGER,
            job TEXT,
            marital TEXT,
            education TEXT,
            poutcome TEXT,
            campaign INTEGER,
            probability REAL,
            prediction TEXT
        )
    """)
    return conn


def save_prediction(name, data, prob, label):
    conn = get_connection()
    conn.execute(
        "INSERT INTO predictions (created_at, customer_name, age, job, marital, education, poutcome, campaign, probability, prediction) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M"), name, data["age"], data["job"], data["marital"],
         data["education"], data["poutcome"], data["campaign"], round(prob, 3), label),
    )
    conn.commit()
    conn.close()


def load_history():
    conn = get_connection()
    history = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return history


model = load_model()

st.title("🏦 Bank Campaign Predictor")
st.write("Fill in the customer details to see if they are likely to say **yes** to the term deposit campaign.")

tab1, tab2 = st.tabs(["New prediction", "Saved predictions"])

with tab1:
    with st.form("customer_form"):
        name = st.text_input("Customer name (optional)")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
                                       "self-employed", "services", "student", "technician", "unemployed", "unknown"])
            marital = st.selectbox("Marital status", ["married", "single", "divorced", "unknown"])
            education = st.selectbox("Education", ["university.degree", "high.school", "professional.course", "basic.9y",
                                                   "basic.6y", "basic.4y", "illiterate", "unknown"])
            default = st.selectbox("Has credit in default?", ["no", "yes", "unknown"])
            housing = st.selectbox("Has housing loan?", ["no", "yes", "unknown"])
            loan = st.selectbox("Has personal loan?", ["no", "yes", "unknown"])
        with col2:
            contact = st.selectbox("Contact type", ["cellular", "telephone"])
            month = st.selectbox("Month of contact", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                                      "oct", "nov", "dec"], index=4)
            day = st.selectbox("Day of week", ["mon", "tue", "wed", "thu", "fri"])
            campaign = st.number_input("Number of contacts in this campaign", min_value=1, max_value=50, value=1)
            previous = st.number_input("Number of contacts in previous campaigns", min_value=0, max_value=10, value=0)
            poutcome = st.selectbox("Result of previous campaign", ["nonexistent", "failure", "success"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        data = {
            "age": age, "job": job, "marital": marital, "education": education, "default": default,
            "housing": housing, "loan": loan, "contact": contact, "month": month, "day_of_week": day,
            "campaign": campaign, "previous": previous, "poutcome": poutcome,
            "contacted_before": 1 if previous > 0 else 0,
        }
        if month in MONTHS_NOT_IN_DATA:
            # the dataset has no calls in jan/feb, so the model knows nothing about these months.
            # instead of guessing, I try the customer in every month the model knows and take the average
            rows = pd.DataFrame([{**data, "month": m} for m in MONTHS_IN_DATA])
            prob = model.predict_proba(rows)[:, 1].mean()
            st.info("No data for this month in the dataset, so the prediction is the average of all other months.")
        else:
            prob = model.predict_proba(pd.DataFrame([data]))[0][1]
        label = "yes" if prob >= THRESHOLD else "no"

        if label == "yes":
            st.success(f"✅ Likely to say YES ({prob:.0%}). Recommended: call this customer.")
        else:
            st.warning(f"❌ Likely to say NO ({prob:.0%}). Lower priority for the campaign.")
        st.progress(float(prob))

        save_prediction(name, data, prob, label)
        st.caption("Prediction saved in the database.")

with tab2:
    history = load_history()
    if history.empty:
        st.info("No predictions yet.")
    else:
        st.write(f"Total predictions: {len(history)}  |  Predicted yes: {(history['prediction'] == 'yes').sum()}")
        st.dataframe(history, width="stretch")

st.divider()
st.caption("MVP for Introduction to AI (AIN5301EN). Model: Random Forest trained on the Bank Marketing dataset.")