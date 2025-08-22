import os                      # >>> n8n
import json                    # >>> n8n
import requests                # >>> n8n
import streamlit as st
import sqlite3
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# DB Helpers

DB_PATH = "health_agent.sqlite3"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def exec_sql(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    return cur

def query_df(conn, sql, params=()):
    return pd.read_sql_query(sql, conn, params=params)


# Initialize DB

def init_db():
    conn = get_conn()
    exec_sql(conn, """
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        created_at TEXT,
        age INTEGER,
        height_cm REAL,
        weight_kg REAL,
        bmi REAL,
        dizzy_headache INTEGER,
        tiredness INTEGER,
        smoker INTEGER,
        diabetic INTEGER,
        cholesterol_high INTEGER,
        physically_active INTEGER,
        rule_score REAL,
        rule_band TEXT,
        ml_band TEXT,
        ml_confidence REAL,
        report TEXT
    )
    """)
    # >>> n8n: add optional columns (safe no-ops if they exist)
    try: exec_sql(conn, "ALTER TABLE records ADD COLUMN plan TEXT")
    except Exception: pass
    try: exec_sql(conn, "ALTER TABLE records ADD COLUMN email_sent_at TEXT")
    except Exception: pass
    return conn


# Risk Score (Rule Based)

def rule_based_score(age,bmi,dizzy,tired,smoker,diabetic,chol,active):
    score=0
    if age>=60: score+=3
    elif age>=45: score+=2
    if bmi>=35: score+=3
    elif bmi>=30: score+=2
    elif bmi>=25: score+=1
    if dizzy: score+=2
    if tired: score+=1
    if smoker: score+=2
    if diabetic: score+=2
    if chol: score+=2
    if not active: score+=1
    band="Low"
    if score>=8: band="High"
    elif score>=4: band="Moderate"
    return score, band

def risk_badge(band:str)->str:
    colors={"Low":"🟢 Low","Moderate":"🟡 Moderate","High":"🔴 High"}
    return colors.get(band,"❓ Unknown")


# ML Model (KNN on SQLite Data)

def knn_predict(df, features):
    if len(df)<5:   # not enough data to train
        return None, None

    X=df[["age","bmi","dizzy_headache","tiredness","smoker",
          "diabetic","cholesterol_high","physically_active"]]
    y=df["rule_band"]  # use rule-based band as labels

    le=LabelEncoder()
    y_enc=le.fit_transform(y)

    knn=KNeighborsClassifier(n_neighbors=min(5,len(df)))
    knn.fit(X,y_enc)

    pred=knn.predict([features])[0]
    proba=knn.predict_proba([features])[0]

    predicted_band=le.inverse_transform([pred])[0]
    confidence=float(np.max(proba))

    return predicted_band, confidence

# n8n integration helpers

def get_n8n_webhook_url():
    """Priority: st.secrets -> env -> sidebar session."""
    url = https://dynamo889.app.n8n.cloud/webhook-test/health-remedies
    try:
        url = st.secrets.get("N8N_WEBHOOK_URL")
    except Exception:
        pass
    if not url:
        url = os.getenv("N8N_WEBHOOK_URL")
    if not url:
        url = st.session_state.get("n8n_url")
    return url

def send_to_n8n(webhook_url: str, payload: dict, timeout: int = 20):
    try:
        resp = requests.post(webhook_url, json=payload, timeout=timeout)
        ok = 200 <= resp.status_code < 300
        return ok, resp.status_code, resp.text
    except Exception as e:
        return False, None, str(e)


# Streamlit UI

st.set_page_config("🧬 Health Risk Agent","🧬",layout="wide")
conn=init_db()

# Sidebar navigation + n8n config  >>> n8n
with st.sidebar:
    menu=st.radio("📌 Navigation",["Home","Form","Dashboard","Reports"])
    with st.expander("⚙️Agentic AI"):
        st.text_input(
            "n8n Webhook URL",
            key="n8n_url",
            placeholder="https://<your-subdomain>.n8n.cloud/webhook/health-remedies",
        )
        auto_send_dashboard = st.checkbox(
            "Auto-send remedies", value=False
        )
       


# Home Page

if menu=="Home":
    st.title("🌿 Health Risk Screener (Rule + ML)")
    st.write("""
    👉 Answer simple questions.  
    👉 We calculate risk using **Rule-based logic** + **KNN ML** from past data.  
    👉 If ML confidence ≥70%, we use ML result. Otherwise, fallback to rule-based.  
    """)


# Form Page

elif menu=="Form":
    st.title("📝 Health Checkup Form")

    with st.form("health_form"):
        name=st.text_input("👤 Your Name")
        email=st.text_input("📧 Your Email")

        age=st.number_input("🎂 Age",18,100,30)
        height=st.number_input("📏 Height (cm)",100.0,220.0,165.0,step=0.5)
        weight=st.number_input("⚖️ Weight (kg)",30.0,150.0,60.0,step=0.5)

        st.markdown("### 🩺 Health Questions (Yes/No)")
        dizzy=st.radio("Do you often feel dizzy or get headaches?",["No","Yes"])=="Yes"
        tired=st.radio("Do you feel tired most of the time?",["No","Yes"])=="Yes"
        smoker=st.radio("Do you smoke?",["No","Yes"])=="Yes"
        diabetic=st.radio("Do you have diabetes?",["No","Yes"])=="Yes"
        chol=st.radio("Have you been told you have high cholesterol?",["No","Yes"])=="Yes"
        active=st.radio("Do you do physical work / exercise regularly?",["Yes","No"])=="Yes"

        submitted = st.form_submit_button("🔍 Analyze Health")

        if submitted:
            bmi=weight/((height/100)**2)
            score,rule_band=rule_based_score(age,bmi,dizzy,tired,smoker,diabetic,chol,active)

            # Get historical data
            df_all=query_df(conn,"SELECT * FROM records")
            features=[age,bmi,int(dizzy),int(tired),int(smoker),int(diabetic),int(chol),int(active)]
            ml_band,confidence=knn_predict(df_all,features)

            # Decide final
            final_band=rule_band
            source="Rule-based"
            if ml_band is not None and confidence>=0.7:
                final_band=ml_band
                source=f"ML (KNN, {confidence*100:.1f}% confident)"

            report=f"""
Health Risk Report ({datetime.utcnow().isoformat()}Z)

👤 Name: {name}  
📧 Email: {email}  

Age: {age}, Height: {height} cm, Weight: {weight} kg, BMI: {bmi:.1f}  

➡️ Rule-based Risk: {rule_band}  
➡️ ML Predicted Risk: {ml_band if ml_band else 'Not enough data'}  
➡️ Final Risk Level: {final_band} ({source})

⚠️ This is a screening tool, not a medical diagnosis.
"""

            exec_sql(conn,"""INSERT INTO records 
            (name,email,created_at,age,height_cm,weight_kg,bmi,
            dizzy_headache,tiredness,smoker,diabetic,cholesterol_high,physically_active,
            rule_score,rule_band,ml_band,ml_confidence,report)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (name,email,datetime.utcnow().isoformat(),age,height,weight,bmi,
             int(dizzy),int(tired),int(smoker),int(diabetic),int(chol),int(active),
             score,rule_band,ml_band,float(confidence) if confidence else None,report))

            st.success("✅ Report Generated!")
            st.markdown(f"### Final Risk Level: {risk_badge(final_band)}")
            st.code(report,language="text")


# Dashboard

elif menu=="Dashboard":
    st.title("📊 Dashboard")

    df=query_df(conn,"SELECT * FROM records ORDER BY id DESC")
    if df.empty:
        st.info("No records yet.")
    else:
        latest=df.iloc[0]
        st.markdown(f"### Latest Final Risk: {risk_badge(latest['rule_band'])}")
        st.dataframe(df[["created_at","name","email","age","bmi","rule_band","ml_band","ml_confidence"]])

        # >>> n8n: Send remedies email for the latest record
        st.markdown("#### 📩 Remedies Email")
        col1, col2 = st.columns([1,2])
        with col1:
            send_clicked = st.button("Get Remedies & Email", use_container_width=True)
        

        def _send_latest_to_n8n(latest_row):
            webhook_url = get_n8n_webhook_url()
            if not webhook_url:
                st.warning("n8n Webhook URL not set. Add it in the sidebar or st.secrets.")
                return
            if not latest_row.get("email"):
                st.error("No email found for this record.")
                return

            payload = {
                "id": int(latest_row["id"]),
                "name": latest_row["name"],
                "email": latest_row["email"],
                "created_at": latest_row["created_at"],
                "summary": {
                    "age": int(latest_row["age"]),
                    "bmi": round(float(latest_row["bmi"]), 1),
                    "rule_band": latest_row["rule_band"],
                    "ml_band": latest_row["ml_band"],
                    "final_band": latest_row["rule_band"] if pd.isna(latest_row["ml_band"]) else latest_row["ml_band"],
                },
                "report_text": latest_row["report"],
                # Give the LLM more context if desired:
                "features": {
                    "dizzy_headache": int(latest_row["dizzy_headache"]),
                    "tiredness": int(latest_row["tiredness"]),
                    "smoker": int(latest_row["smoker"]),
                    "diabetic": int(latest_row["diabetic"]),
                    "cholesterol_high": int(latest_row["cholesterol_high"]),
                    "physically_active": int(latest_row["physically_active"]),
                },
                "llm_task": "Generate a personalized wellness & remedies plan based on risk and features and return HTML for email."
            }

            with st.spinner("📨 Contacting n8n to generate & send remedies email..."):
                ok, status, body = send_to_n8n(webhook_url, payload)

            if ok:
                plan_text = None
                try:
                    # If your n8n workflow responds with JSON { plan_text, plan_html }
                    data = json.loads(body)
                    plan_text = data.get("plan_text")
                except Exception:
                    pass

                exec_sql(conn,
                    "UPDATE records SET email_sent_at=?, plan=? WHERE id=?",
                    (datetime.utcnow().isoformat(), plan_text, int(latest_row["id"]))
                )
                st.success("📧 Email sent")
                if plan_text:
                    with st.expander("Preview of AI Remedies (from n8n response)"):
                        st.text(plan_text)
            else:
                st.error(f"n8n send failed (status={status}). Details: {str(body)[:300]}")

        if send_clicked:
            _send_latest_to_n8n(latest)

        # Optional auto-send when opening dashboard and not yet sent
        if auto_send_dashboard and (("email_sent_at" not in latest) or pd.isna(latest["email_sent_at"])):
            _send_latest_to_n8n(latest)


# Reports

elif menu=="Reports":
    st.title("🗂 Reports")
    df=query_df(conn,"SELECT report, plan FROM records ORDER BY id DESC LIMIT 1")
    if not df.empty:
        st.subheader("Latest Report")
        st.code(df.iloc[0]["report"],language="text")
        if "plan" in df.columns and pd.notna(df.iloc[0]["plan"]):
            st.subheader("Latest AI Remedies (saved)")
            st.text(df.iloc[0]["plan"])
    else:
        st.info("No reports yet.")
