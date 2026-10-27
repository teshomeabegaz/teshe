
import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(
    page_title="Health Informatics System",
    page_icon="🏥",
    layout="wide"
)

# Temporary database for fictional practice records
DATABASE = "/tmp/health_informatics.db"


def initialize_database():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Patients (
                PatientID INTEGER PRIMARY KEY AUTOINCREMENT,
                FullName TEXT NOT NULL,
                Age INTEGER NOT NULL,
                Sex TEXT NOT NULL,
                CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_patient(full_name, age, sex):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute(
            """
            INSERT INTO Patients (FullName, Age, Sex)
            VALUES (?, ?, ?)
            """,
            (full_name, age, sex)
        )


def get_patients():
    with sqlite3.connect(DATABASE) as conn:
        return pd.read_sql_query(
            """
            SELECT PatientID, FullName, Age, Sex, CreatedAt
            FROM Patients
            ORDER BY PatientID DESC
            """,
            conn
        )


initialize_database()

st.title("🏥 Health Informatics System")
st.write("Patient Registration and Database Management")
st.caption("Practice application — use fictional data only.")
st.divider()

form_column, records_column = st.columns([1, 1.4])

with form_column:
    st.subheader("Register a Test Patient")

    with st.form("patient_form", clear_on_submit=True):
        full_name = st.text_input("Full name")
        age = st.number_input(
            "Age", min_value=0, max_value=120, value=25
        )
        sex = st.selectbox(
            "Sex", ["Female", "Male", "Other", "Unknown"]
        )

        submitted = st.form_submit_button(
            "💾 Save Record", type="primary"
        )

        if submitted:
            if not full_name.strip():
                st.warning("Please enter a name.")
            else:
                try:
                    save_patient(
                        full_name.strip(), int(age), sex
                    )
                    st.success("Practice record saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save record: {e}")

with records_column:
    st.subheader("Saved Records")

    try:
        df = get_patients()
        st.metric("Total records", len(df))
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load records: {e}")
