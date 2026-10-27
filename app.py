
import streamlit as st
import pyodbc
import pandas as pd

st.set_page_config(
    page_title="Health Informatics System",
    page_icon="🏥",
    layout="wide"
)

# SQL Server connection settings
SERVER = r"localhost"
DATABASE = "HealthInformaticsPractice"
DRIVER = "ODBC Driver 18 for SQL Server"

CONNECTION_STRING = (
    f"DRIVER={{{DRIVER}}};"
    f"SERVER={SERVER};"
    "DATABASE=master;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)


def initialize_database():
    # Create the practice database if necessary
    with pyodbc.connect(
        CONNECTION_STRING, autocommit=True
    ) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            IF DB_ID(N'HealthInformaticsPractice') IS NULL
                CREATE DATABASE HealthInformaticsPractice;
        """)
        cursor.close()

    # Create the table in the practice database
    with pyodbc.connect(
        CONNECTION_STRING.replace(
            "DATABASE=master;",
            f"DATABASE={DATABASE};"
        )
    ) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            IF OBJECT_ID(N'dbo.Patients', N'U') IS NULL
            BEGIN
                CREATE TABLE dbo.Patients (
                    PatientID INT IDENTITY(1,1) PRIMARY KEY,
                    FullName NVARCHAR(100) NOT NULL,
                    Age INT NOT NULL,
                    Sex NVARCHAR(10) NOT NULL,
                    CreatedAt DATETIME2
                        NOT NULL DEFAULT SYSDATETIME()
                );
            END
        """)
        conn.commit()
        cursor.close()


def get_connection():
    return pyodbc.connect(
        CONNECTION_STRING.replace(
            "DATABASE=master;",
            f"DATABASE={DATABASE};"
        )
    )


st.title("🏥 Health Informatics System")
st.write("Patient Registration and Database Management")
st.divider()

# Initialize database
try:
    initialize_database()
    st.success("SQL Server connected successfully.")
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Two-column layout
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
                    with get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            """
                            INSERT INTO dbo.Patients
                                (FullName, Age, Sex)
                            VALUES (?, ?, ?)
                            """,
                            full_name.strip(),
                            int(age),
                            sex
                        )
                        conn.commit()

                    st.success("Record saved to SQL Server!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Could not save record: {e}")

with records_column:
    st.subheader("Saved Records")

    try:
        with get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT PatientID, FullName, Age, Sex, CreatedAt
                FROM dbo.Patients
                ORDER BY PatientID DESC
                """,
                conn
            )

        st.metric("Total records", len(df))
        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load records: {e}")

st.caption("Practice application — use fictional data only.")
