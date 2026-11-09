# Bank Campaign Predictor (MVP)

Introduction to AI (AIN5301EN) – Portfolio project

A simple web app that helps a bank decide **which customers to call** in a term deposit campaign.
The bank employee enters some customer details and the app predicts if the customer is likely to say **yes** or **no**.

## What is inside

| File | What it is |
|------|------------|
| `bank_campaign_model.ipynb` | Jupyter notebook: data loading, preprocessing, training, evaluation |
| `app.py` | The MVP web app (Streamlit) |
| `model.pkl` | The trained model (Random Forest) used by the app |
| `data/bank-additional-full.csv` | Bank Marketing dataset (41,188 customers) |
| `requirements.txt` | Python libraries |

## How to run

```bash
pip install -r requirements.txt

# 1. (optional) train the model again - creates model.pkl and bank.db
jupyter notebook bank_campaign_model.ipynb

# 2. start the web app
streamlit run app.py
```

The app opens in the browser at `http://localhost:8501`.

## How it works

1. The CSV is loaded into a SQLite database (`bank.db`, table `customers`).
2. The notebook reads the data from the database, cleans it and trains 3 models
   (Logistic Regression, Decision Tree, Random Forest).
3. The best model (Random Forest) is saved as `model.pkl`.
4. The web app loads `model.pkl`, makes a prediction for the customer from the form
   and saves every prediction in the `predictions` table of the same database.

## Results (test set, 8,238 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.797 | 0.297 | 0.587 | 0.394 | 0.765 |
| Decision Tree | 0.870 | 0.422 | 0.412 | 0.417 | 0.728 |
| **Random Forest** | 0.840 | 0.365 | 0.577 | **0.447** | **0.782** |

A model that always says "no" gets 88.7% accuracy but finds 0 customers, so I mainly look at Recall, F1 and ROC AUC.

## Dataset

Bank Marketing dataset (Moro, Cortez & Rita, 2014), available on Kaggle and the UCI Machine Learning Repository.

## Use of Generative AI

Generative AI (Claude, Anthropic) was used in this project.
