
import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

st.set_page_config(
    page_title="Health Data Analysis Tool",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Health Data Analysis Tool")
st.caption("Python • Pandas • Streamlit • Health Informatics")

st.info(
    "For learning and practice. Upload de-identified or "
    "synthetic datasets only; do not upload identifiable patient data."
)

# Patient practice database
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


def get_patients():
    with sqlite3.connect(DATABASE) as conn:
        return pd.read_sql_query(
            "SELECT PatientID, FullName, Age, Sex, CreatedAt "
            "FROM Patients ORDER BY PatientID DESC",
            conn
        )


def save_patient(name, age, sex):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute(
            "INSERT INTO Patients (FullName, Age, Sex) VALUES (?, ?, ?)",
            (name, age, sex)
        )


initialize_database()

tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard",
    "📁 Analyze Dataset",
    "🧮 Python Statistics"
])

with tab1:
    st.header("Patient Practice Dashboard")

    patients = get_patients()
    col1, col2, col3 = st.columns(3)

    col1.metric("Total records", len(patients))

    if not patients.empty:
        col2.metric("Mean age", f"{patients['Age'].mean():.1f}")
        col3.metric("Age standard deviation",
                    f"{patients['Age'].std():.2f}"
                    if len(patients) > 1 else "N/A")

        st.subheader("Records by sex")
        st.bar_chart(patients["Sex"].value_counts())

        st.subheader("Registered practice records")
        st.dataframe(patients, use_container_width=True)
    else:
        st.write("No practice records yet.")

    st.subheader("Register a fictional test patient")

    with st.form("patient_form", clear_on_submit=True):
        name = st.text_input("Name or fictional ID")
        age = st.number_input("Age", 0, 120, 25)
        sex = st.selectbox(
            "Sex", ["Female", "Male", "Other", "Unknown"]
        )
        submitted = st.form_submit_button("Save practice record")

        if submitted:
            if not name.strip():
                st.warning("Please enter a name or fictional ID.")
            else:
                save_patient(name.strip(), int(age), sex)
                st.success("Practice record saved.")
                st.rerun()

with tab2:
    st.header("Upload a Health Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(
                f"Loaded {df.shape[0]} rows and {df.shape[1]} columns."
            )

            st.subheader("Dataset preview")
            st.dataframe(df.head(100), use_container_width=True)

            st.subheader("Dataset information")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Missing cells", int(df.isna().sum().sum()))

            st.subheader("Descriptive statistics")
            st.dataframe(
                df.describe(include="all").transpose(),
                use_container_width=True
            )

            st.subheader("Missing values by column")
            missing = df.isna().sum().rename("Missing values")
            st.dataframe(missing.to_frame())

            numeric = df.select_dtypes(include="number")

            if not numeric.empty:
                st.subheader("Explore a numeric variable")
                variable = st.selectbox(
                    "Select variable", numeric.columns
                )

                left, right = st.columns(2)

                with left:
                    st.write("Distribution")
                    st.bar_chart(
                        numeric[variable].dropna()
                        .value_counts().sort_index()
                        .head(50)
                    )

                with right:
                    st.write("Summary")
                    st.write(numeric[variable].describe())

                if len(numeric.columns) >= 2:
                    st.subheader("Correlation matrix")
                    st.dataframe(
                        numeric.corr(),
                        use_container_width=True
                    )

                    st.write("Correlation heatmap values")
                    st.caption(
                        "Correlations range from -1 to +1. "
                        "Correlation does not establish causation."
                    )

            st.subheader("Frequency table")
            categorical = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

            if categorical:
                selected = st.selectbox(
                    "Select a categorical variable", categorical
                )
                frequencies = (
                    df[selected].fillna("Missing")
                    .value_counts()
                    .rename_axis(selected)
                    .reset_index(name="Frequency")
                )
                st.dataframe(frequencies, use_container_width=True)
                st.bar_chart(frequencies.set_index(selected))

            st.download_button(
                "Download descriptive statistics (CSV)",
                data=df.describe(include="all").transpose()
                .to_csv().encode("utf-8"),
                file_name="descriptive_statistics.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Unable to analyze this file: {error}")

with tab3:
    st.header("Learn Python Statistics")

    st.write(
        "Enter values separated by commas to calculate basic "
        "descriptive statistics."
    )

    numbers_text = st.text_input(
        "Numeric values",
        value="20, 25, 30, 35, 40"
    )

    if st.button("Calculate statistics"):
        try:
            values = [
                float(item.strip())
                for item in numbers_text.split(",")
                if item.strip()
            ]

            if not values:
                st.warning("Enter at least one number.")
            else:
                series = pd.Series(values)

                a, b, c = st.columns(3)
                a.metric("Count", len(series))
                b.metric("Mean", f"{series.mean():.2f}")
                c.metric("Median", f"{series.median():.2f}")

                st.write("Full statistical summary")
                st.dataframe(series.describe().to_frame("Value"))

                st.write("Values chart")
                st.line_chart(series)

        except ValueError:
            st.error("Enter valid numbers separated by commas.")
