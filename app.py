
import sqlite3
import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm
from supabase import create_client



# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title=(
        "MU-CHS Faculty of Public Health | "
        "Health Informatics Department | "
        "Health Data Analysis Tool"
    ),
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------
# SUPABASE CONNECTION AND USER LOGIN
# --------------------------------------------------

def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        return None

    return create_client(url, key)


if "supabase_client" not in st.session_state:
    st.session_state.supabase_client = get_supabase_client()

supabase = st.session_state.supabase_client

if supabase is None:
    st.error(
        "Supabase connection settings are missing. "
        "Please check SUPABASE_URL and SUPABASE_KEY "
        "in Render Environment Variables."
    )
    st.stop()

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("Health Informatics — Sign In")

    login_tab, register_tab = st.tabs(
        ["Sign In", "Create Account"]
    )

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email address")
            password = st.text_input(
                "Password", type="password"
            )
            login_clicked = st.form_submit_button("Sign In")

        if login_clicked:
            try:
                result = supabase.auth.sign_in_with_password(
                    {"email": email, "password": password}
                )
                st.session_state.user = result.user
                st.rerun()
            except Exception:
                st.error(
                    "Sign-in failed. Check your email and password."
                )

    with register_tab:
        with st.form("register_form"):
            full_name = st.text_input("Full name")
            new_email = st.text_input("Email address")
            new_password = st.text_input(
                "Create password", type="password"
            )
            register_clicked = st.form_submit_button(
                "Create Account"
            )

        if register_clicked:
            if not full_name.strip():
                st.error("Please enter your name.")
            elif len(new_password) < 8:
                st.error(
                    "Use a password with at least 8 characters."
                )
            else:
                try:
                    result = supabase.auth.sign_up(
                        {
                            "email": new_email,
                            "password": new_password,
                            "options": {
                                "data": {"full_name": full_name}
                            },
                        }
                    )
                    if result.user and not result.session:
                        st.success(
                            "Account created. Check your email "
                            "to confirm your account, then sign in."
                        )
                    elif result.user:
                        st.success(
                            "Account created. You can now sign in."
                        )
                except Exception:
                    st.error(
                        "Registration failed. Check the email "
                        "and try again."
                    )

    st.stop()

st.sidebar.success(
    f"Signed in: {st.session_state.user.email}"
)

if st.sidebar.button("Sign Out"):
    try:
        supabase.auth.sign_out()
    finally:
        st.session_state.user = None
        st.rerun()


st.title("🏥 MU-CHS Faculty of Public Health")
st.subheader("Health Informatics Department")
st.header("Health Data Analysis Tool")

st.caption(
    "A teaching and learning platform for health data "
    "management, statistical analysis, and clinical research."
)

st.markdown(
    "**Developed and provided by Teshome Abegaz**"
)

st.info(
    "For educational use. Use fictional or de-identified data only. "
    "Do not upload identifiable patient information."
)



# --------------------------------------------------
# 2. DATABASE FUNCTIONS
# --------------------------------------------------

DB_PATH = "/tmp/health_informatics.db"


def initialize_database():
    """Create the local practice database if it does not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id TEXT NOT NULL,
                age INTEGER,
                sex TEXT,
                diagnosis TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def get_patients():
    """Retrieve fictional practice records."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT id, study_id, age, sex, diagnosis, created_at
            FROM patients
            ORDER BY id DESC
            """,
            conn,
        )


def save_patient(study_id, age, sex, diagnosis):
    """Save a practice record."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO patients (study_id, age, sex, diagnosis)
            VALUES (?, ?, ?, ?)
            """,
            (study_id, age, sex, diagnosis),
        )


initialize_database()


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def load_dataset(uploaded_file):
    """Read a CSV or Excel file into a pandas DataFrame."""
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Please upload a CSV or Excel file.")


def get_confidence_level():
    """Return the selected confidence level as a decimal."""
    return {
        "90%": 0.90,
        "95%": 0.95,
        "99%": 0.99,
    }


def show_file_upload(key):
    """Display a dataset upload control."""
    return st.file_uploader(
        "Upload a CSV or Excel dataset",
        type=["csv", "xlsx", "xls"],
        key=key,
    )


# --------------------------------------------------
# 4. APPLICATION TABS
# --------------------------------------------------


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Dashboard",
        "📁 Analyze Dataset",
        "🧮 Python Statistics",
        "📈 Statistical Tests",
        "📝 Data Collection Forms",
    ]
)



# --------------------------------------------------
# TAB 1: DASHBOARD
# --------------------------------------------------

with tab1:
    st.header("Practice Health Data Dashboard")

    st.write(
        "Register fictional practice records and explore "
        "basic health information."
    )

    patients = get_patients()

    col1, col2, col3 = st.columns(3)

    col1.metric("Practice Records", len(patients))

    if not patients.empty:
        ages = pd.to_numeric(patients["age"], errors="coerce")
        col2.metric(
            "Average Age",
            f"{ages.mean():.1f}" if ages.notna().any() else "N/A",
        )
        col3.metric(
            "Recorded Diagnoses",
            int(patients["diagnosis"].nunique()),
        )
    else:
        col2.metric("Average Age", "N/A")
        col3.metric("Recorded Diagnoses", 0)

    st.subheader("Add a Fictional Practice Record")

    with st.form("practice_patient_form"):
        study_id = st.text_input(
            "Study ID",
            placeholder="Example: STUDY001",
        )

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=30,
            step=1,
        )

        sex = st.selectbox(
            "Sex",
            ["Female", "Male", "Other", "Not recorded"],
        )

        diagnosis = st.selectbox(
            "Example diagnosis",
            [
                "Hypertension",
                "Diabetes",
                "Pneumonia",
                "Malaria",
                "Trauma",
                "Other",
            ],
        )

        submitted = st.form_submit_button("Save Practice Record")

        if submitted:
            if not study_id.strip():
                st.error("Please enter a study ID.")
            else:
                save_patient(
                    study_id.strip(),
                    int(age),
                    sex,
                    diagnosis,
                )
                st.success("Practice record saved.")
                st.rerun()

    st.subheader("Practice Records")

    patients = get_patients()

    if not patients.empty:
        st.dataframe(
            patients.drop(columns=["created_at"]),
            use_container_width=True,
        )

        csv_data = patients.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Practice Records (CSV)",
            data=csv_data,
            file_name="practice_health_records.csv",
            mime="text/csv",
        )
    else:
        st.info("No practice records have been added yet.")


# --------------------------------------------------
# TAB 2: DATASET ANALYSIS
# --------------------------------------------------

with tab2:
    st.header("Analyze a Health Dataset")

    st.write(
        "Upload a CSV or Excel dataset to examine its structure, "
        "descriptive statistics, missing values, distributions, "
        "and correlations."
    )

    uploaded_file = show_file_upload("analysis_dataset")

    if uploaded_file is not None:
        try:
            df = load_dataset(uploaded_file)

            if df.empty:
                st.warning("The uploaded dataset is empty.")
            else:
                st.subheader("Dataset Preview")
                st.dataframe(df.head(20), use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", df.shape[0])
                c2.metric("Columns", df.shape[1])
                c3.metric(
                    "Missing Cells",
                    int(df.isna().sum().sum()),
                )

                st.subheader("Column Information")

                info_df = pd.DataFrame(
                    {
                        "Column": df.columns.astype(str),
                        "Data Type": [
                            str(dtype) for dtype in df.dtypes
                        ],
                        "Missing Values": df.isna().sum().values,
                        "Missing (%)": (
                            df.isna().mean().values * 100
                        ).round(2),
                        "Unique Values": df.nunique(
                            dropna=True
                        ).values,
                    }
                )

                st.dataframe(info_df, use_container_width=True)

                st.subheader("Descriptive Statistics")

                descriptive = df.describe(
                    include="all"
                ).transpose()

                st.dataframe(
                    descriptive,
                    use_container_width=True,
                )

                descriptive_csv = descriptive.to_csv().encode("utf-8")

                st.download_button(
                    "Download Descriptive Statistics",
                    data=descriptive_csv,
                    file_name="descriptive_statistics.csv",
                    mime="text/csv",
                    key="download_descriptive",
                )

                numeric_df = df.select_dtypes(
                    include=np.number
                )

                if not numeric_df.empty:
                    st.subheader("Numeric Variable Distribution")

                    numeric_column = st.selectbox(
                        "Select a numeric variable",
                        numeric_df.columns.tolist(),
                        key="distribution_column",
                    )

                    selected_values = numeric_df[
                        numeric_column
                    ].dropna()

                    if not selected_values.empty:
                        fig, ax = plt.subplots()
                        ax.hist(
                            selected_values,
                            bins="auto",
                            edgecolor="black",
                        )
                        ax.set_xlabel(numeric_column)
                        ax.set_ylabel("Frequency")
                        ax.set_title(
                            f"Distribution of {numeric_column}"
                        )
                        st.pyplot(fig)
                        plt.close(fig)

                        st.write(
                            "Mean:",
                            round(float(selected_values.mean()), 3),
                        )
                        st.write(
                            "Median:",
                            round(float(selected_values.median()), 3),
                        )
                        st.write(
                            "Standard deviation:",
                            round(
                                float(selected_values.std()),
                                3,
                            )
                            if len(selected_values) > 1
                            else "Not available for one observation",
                        )

                    if len(numeric_df.columns) >= 2:
                        st.subheader("Correlation Matrix")

                        correlation = numeric_df.corr(
                            method="pearson"
                        )

                        st.dataframe(
                            correlation.round(3),
                            use_container_width=True,
                        )

                        st.write("Correlation Heatmap")

                        fig, ax = plt.subplots(
                            figsize=(8, 5)
                        )

                        image = ax.imshow(
                            correlation,
                            vmin=-1,
                            vmax=1,
                            aspect="auto",
                        )

                        ax.set_xticks(
                            range(len(correlation.columns))
                        )
                        ax.set_xticklabels(
                            correlation.columns,
                            rotation=45,
                            ha="right",
                        )
                        ax.set_yticks(
                            range(len(correlation.index))
                        )
                        ax.set_yticklabels(correlation.index)

                        fig.colorbar(image, ax=ax)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                else:
                    st.info(
                        "No numeric columns were detected "
                        "for numeric analysis."
                    )

                st.subheader("Missing Values by Column")

                missing = df.isna().sum()
                missing = missing[missing > 0]

                if not missing.empty:
                    st.bar_chart(missing)
                else:
                    st.success(
                        "No missing values were detected."
                    )

        except Exception as error:
            st.error(
                f"Could not analyze the file: {error}"
            )


# --------------------------------------------------
# TAB 3: PYTHON STATISTICS
# --------------------------------------------------

with tab3:
    st.header("Learn Python Statistics")

    st.write(
        "Enter numeric values separated by commas to calculate "
        "basic descriptive statistics."
    )

    numbers_text = st.text_input(
        "Numeric values",
        value="20, 25, 30, 35, 40",
    )

    if st.button("Calculate Statistics"):
        try:
            values = [
                float(item.strip())
                for item in numbers_text.split(",")
                if item.strip()
            ]

            if not values:
                st.warning("Enter at least one number.")

            elif not np.isfinite(values).all():
                st.error(
                    "Please enter finite numbers only; "
                    "NaN and infinity are not supported."
                )

            else:
                series = pd.Series(values)

                a, b, c = st.columns(3)

                a.metric("Count", len(series))
                b.metric("Mean", f"{series.mean():.2f}")
                c.metric("Median", f"{series.median():.2f}")

                st.subheader("Full Statistical Summary")

                st.dataframe(
                    series.describe().to_frame("Value"),
                    use_container_width=True,
                )

                st.subheader("Values Chart")

                st.line_chart(series)

        except ValueError:
            st.error(
                "Enter valid numbers separated by commas."
            )


# --------------------------------------------------
# TAB 4: STATISTICAL TESTS
# --------------------------------------------------

with tab4:
    st.header("Statistical Tests for Health Research")

    st.write(
        "Upload a dataset to calculate confidence intervals, "
        "compare means, test associations, and fit binary "
        "logistic regression models."
    )

    st.warning(
        "Choose variables carefully. Statistical tests require "
        "appropriate study designs and assumptions. Results from "
        "small or biased samples may be unreliable."
    )

    statistical_file = show_file_upload(
        "statistical_dataset"
    )

    if statistical_file is not None:
        try:
            data = load_dataset(statistical_file)

            if data.empty:
                st.warning("The uploaded dataset is empty.")

            else:
                st.subheader("Dataset Preview")
                st.dataframe(
                    data.head(10),
                    use_container_width=True,
                )

                confidence_label = st.selectbox(
                    "Confidence level",
                    ["90%", "95%", "99%"],
                    index=1,
                )

                confidence = get_confidence_level()[
                    confidence_label
                ]

                alpha = 1 - confidence

                st.divider()

                # ------------------------------------------
                # 4A. CONFIDENCE INTERVAL FOR A MEAN
                # ------------------------------------------

                st.subheader(
                    "1. Confidence Interval for a Mean"
                )

                numeric_columns = data.select_dtypes(
                    include=np.number
                ).columns.tolist()

                if numeric_columns:
                    mean_column = st.selectbox(
                        "Select a numeric variable",
                        numeric_columns,
                        key="mean_ci_column",
                    )

                    mean_values = pd.to_numeric(
                        data[mean_column],
                        errors="coerce",
                    ).dropna()

                    mean_values = mean_values[
                        np.isfinite(mean_values)
                    ]

                    if len(mean_values) >= 2:
                        mean_value = float(mean_values.mean())
                        standard_error = float(
                            stats.sem(mean_values)
                        )

                        lower, upper = stats.t.interval(
                            confidence,
                            df=len(mean_values) - 1,
                            loc=mean_value,
                            scale=standard_error,
                        )

                        c1, c2, c3 = st.columns(3)
                        c1.metric(
                            "Sample Mean",
                            f"{mean_value:.3f}",
                        )
                        c2.metric(
                            "Lower CI",
                            f"{lower:.3f}",
                        )
                        c3.metric(
                            "Upper CI",
                            f"{upper:.3f}",
                        )

                        st.caption(
                            f"{confidence_label} t-based confidence "
                            f"interval for the population mean."
                        )

                    elif len(mean_values) == 1:
                        st.info(
                            "At least two valid observations "
                            "are needed to calculate this interval."
                        )
                else:
                    st.info(
                        "No numeric columns are available "
                        "for a mean confidence interval."
                    )

                st.divider()

                # ------------------------------------------
                # 4B. CONFIDENCE INTERVAL FOR A PROPORTION
                # ------------------------------------------

                st.subheader(
                    "2. Confidence Interval for a Proportion"
                )

                st.write(
                    "Enter the number of events and the total "
                    "number of observations."
                )

                p1, p2 = st.columns(2)

                with p1:
                    events = st.number_input(
                        "Number of events",
                        min_value=0,
                        value=20,
                        step=1,
                        key="proportion_events",
                    )

                with p2:
                    total = st.number_input(
                        "Total observations",
                        min_value=1,
                        value=100,
                        step=1,
                        key="proportion_total",
                    )

                if st.button(
                    "Calculate Proportion CI",
                    key="proportion_button",
                ):
                    if events > total:
                        st.error(
                            "The number of events cannot exceed "
                            "the total observations."
                        )
                    else:
                        proportion = events / total

                        lower_p, upper_p = proportion_confint(
                            count=int(events),
                            nobs=int(total),
                            alpha=alpha,
                            method="wilson",
                        )

                        p1, p2, p3 = st.columns(3)

                        p1.metric(
                            "Proportion",
                            f"{proportion:.3f}",
                        )
                        p2.metric(
                            "Lower CI",
                            f"{lower_p:.3f}",
                        )
                        p3.metric(
                            "Upper CI",
                            f"{upper_p:.3f}",
                        )

                        st.caption(
                            f"{confidence_label} Wilson confidence "
                            "interval."
                        )

                st.divider()

                # ------------------------------------------
                # 4C. INDEPENDENT-SAMPLES WELCH T-TEST
                # ------------------------------------------

                st.subheader(
                    "3. Independent-Samples t-Test"
                )

                st.write(
                    "Compare the means of a numeric variable "
                    "between two independent groups."
                )

                if numeric_columns and len(data.columns) >= 2:
                    t_col1, t_col2 = st.columns(2)

                    with t_col1:
                        outcome_column = st.selectbox(
                            "Numeric outcome",
                            numeric_columns,
                            key="ttest_outcome",
                        )

                    group_columns = [
                        column
                        for column in data.columns
                        if column != outcome_column
                    ]

                    with t_col2:
                        group_column = st.selectbox(
                            "Grouping variable",
                            group_columns,
                            key="ttest_group",
                        )

                    group_values = data[
                        group_column
                    ].dropna().unique().tolist()

                    if len(group_values) == 2:
                        group_a = st.selectbox(
                            "First group",
                            group_values,
                            key="ttest_group_a",
                        )

                        group_b = (
                            group_values[1]
                            if group_a == group_values[0]
                            else group_values[0]
                        )

                        st.write(
                            f"Second group: **{group_b}**"
                        )

                        if st.button(
                            "Run Independent t-Test",
                            key="ttest_button",
                        ):
                            work = data[
                                [outcome_column, group_column]
                            ].copy()

                            work[outcome_column] = pd.to_numeric(
                                work[outcome_column],
                                errors="coerce",
                            )

                            work = work.dropna()

                            values_a = work.loc[
                                work[group_column] == group_a,
                                outcome_column,
                            ].to_numpy(dtype=float)

                            values_b = work.loc[
                                work[group_column] == group_b,
                                outcome_column,
                            ].to_numpy(dtype=float)

                            values_a = values_a[
                                np.isfinite(values_a)
                            ]
                            values_b = values_b[
                                np.isfinite(values_b)
                            ]

                            if (
                                len(values_a) < 2
                                or len(values_b) < 2
                            ):
                                st.error(
                                    "Each group needs at least "
                                    "two valid observations."
                                )
                            else:
                                test_result = stats.ttest_ind(
                                    values_a,
                                    values_b,
                                    equal_var=False,
                                )

                                mean_a = float(
                                    np.mean(values_a)
                                )
                                mean_b = float(
                                    np.mean(values_b)
                                )

                                difference = mean_a - mean_b

                                var_a = np.var(
                                    values_a,
                                    ddof=1,
                                )
                                var_b = np.var(
                                    values_b,
                                    ddof=1,
                                )

                                se_difference = np.sqrt(
                                    var_a / len(values_a)
                                    + var_b / len(values_b)
                                )

                                numerator = (
                                    var_a / len(values_a)
                                    + var_b / len(values_b)
                                ) ** 2

                                denominator = (
                                    (var_a / len(values_a)) ** 2
                                    / (len(values_a) - 1)
                                    + (var_b / len(values_b)) ** 2
                                    / (len(values_b) - 1)
                                )

                                if denominator > 0:
                                    degrees_freedom = (
                                        numerator / denominator
                                    )
                                else:
                                    degrees_freedom = (
                                        len(values_a)
                                        + len(values_b)
                                        - 2
                                    )

                                critical_value = stats.t.ppf(
                                    1 - alpha / 2,
                                    df=degrees_freedom,
                                )

                                lower_diff = (
                                    difference
                                    - critical_value
                                    * se_difference
                                )

                                upper_diff = (
                                    difference
                                    + critical_value
                                    * se_difference
                                )

                                c1, c2, c3 = st.columns(3)

                                c1.metric(
                                    f"Mean: {group_a}",
                                    f"{mean_a:.3f}",
                                )

                                c2.metric(
                                    f"Mean: {group_b}",
                                    f"{mean_b:.3f}",
                                )

                                c3.metric(
                                    "Mean Difference",
                                    f"{difference:.3f}",
                                )

                                st.write(
                                    f"Welch t-statistic: "
                                    f"{test_result.statistic:.4f}"
                                )

                                st.write(
                                    f"p-value: "
                                    f"{test_result.pvalue:.6g}"
                                )

                                st.write(
                                    f"{confidence_label} CI for mean "
                                    f"difference: "
                                    f"({lower_diff:.3f}, "
                                    f"{upper_diff:.3f})"
                                )

                                if test_result.pvalue < alpha:
                                    st.success(
                                        "The difference is statistically "
                                        f"significant at alpha = "
                                        f"{alpha:.3f}."
                                    )
                                else:
                                    st.info(
                                        "The difference is not "
                                        "statistically significant at "
                                        f"alpha = {alpha:.3f}."
                                    )

                                st.caption(
                                    "Welch's t-test does not assume "
                                    "equal population variances. "
                                    "The groups should be independent."
                                )

                    else:
                        st.info(
                            "The grouping variable must have exactly "
                            "two observed categories for this test."
                        )

                st.divider()

                # ------------------------------------------
                # 4D. CHI-SQUARE TEST OF ASSOCIATION
                # ------------------------------------------

                st.subheader(
                    "4. Chi-Square Test of Association"
                )

                st.write(
                    "Test whether two categorical variables "
                    "are associated."
                )

                categorical_columns = [
                    column
                    for column in data.columns
                    if (
                        not pd.api.types.is_numeric_dtype(
                            data[column]
                        )
                        or data[column].nunique(dropna=True) <= 10
                    )
                ]

                if len(categorical_columns) >= 2:
                    chi_col1, chi_col2 = st.columns(2)

                    with chi_col1:
                        categorical_a = st.selectbox(
                            "First categorical variable",
                            categorical_columns,
                            key="chi_column_a",
                        )

                    possible_b = [
                        column
                        for column in categorical_columns
                        if column != categorical_a
                    ]

                    with chi_col2:
                        categorical_b = st.selectbox(
                            "Second categorical variable",
                            possible_b,
                            key="chi_column_b",
                        )

                    if st.button(
                        "Run Chi-Square Test",
                        key="chi_button",
                    ):
                        contingency_data = data[
                            [categorical_a, categorical_b]
                        ].dropna()

                        if contingency_data.empty:
                            st.error(
                                "No complete observations are available."
                            )
                        else:
                            contingency_table = pd.crosstab(
                                contingency_data[categorical_a],
                                contingency_data[categorical_b],
                            )

                            st.write("Contingency Table")
                            st.dataframe(
                                contingency_table,
                                use_container_width=True,
                            )

                            if (
                                contingency_table.shape[0] < 2
                                or contingency_table.shape[1] < 2
                            ):
                                st.error(
                                    "Both variables must have at least "
                                    "two observed categories."
                                )
                            else:
                                (
                                    chi2,
                                    p_value,
                                    degrees_freedom,
                                    expected,
                                ) = stats.chi2_contingency(
                                    contingency_table
                                )

                                st.write(
                                    f"Chi-square statistic: {chi2:.4f}"
                                )
                                st.write(
                                    f"Degrees of freedom: "
                                    f"{degrees_freedom}"
                                )
                                st.write(
                                    f"p-value: {p_value:.6g}"
                                )

                                expected_df = pd.DataFrame(
                                    expected,
                                    index=contingency_table.index,
                                    columns=contingency_table.columns,
                                )

                                st.write(
                                    "Expected Cell Frequencies"
                                )
                                st.dataframe(
                                    expected_df.round(2),
                                    use_container_width=True,
                                )

                                if (expected < 5).any():
                                    st.warning(
                                        "Some expected cell counts are "
                                        "below 5. The chi-square "
                                        "approximation may be unreliable. "
                                        "Consider an exact test or "
                                        "combining categories only when "
                                        "scientifically justified."
                                    )

                                if p_value < alpha:
                                    st.success(
                                        "Evidence of an association "
                                        f"at alpha = {alpha:.3f}."
                                    )
                                else:
                                    st.info(
                                        "Insufficient evidence of an "
                                        "association at "
                                        f"alpha = {alpha:.3f}."
                                    )

                                st.caption(
                                    "Association does not establish "
                                    "causation. Observations should be "
                                    "independent for the usual test."
                                )

                else:
                    st.info(
                        "At least two categorical variables are "
                        "needed for the chi-square test."
                    )

                st.divider()

                # ------------------------------------------
                # 4E. BINARY LOGISTIC REGRESSION
                # ------------------------------------------

                st.subheader(
                    "5. Binary Logistic Regression"
                )

                st.write(
                    "Model a binary outcome coded as 0 and 1 using "
                    "numeric predictor variables."
                )

                if len(numeric_columns) >= 2:
                    logistic_outcome = st.selectbox(
                        "Binary outcome variable (0/1)",
                        numeric_columns,
                        key="logistic_outcome",
                    )

                    predictor_options = [
                        column
                        for column in numeric_columns
                        if column != logistic_outcome
                    ]

                    predictors = st.multiselect(
                        "Select numeric predictor variables",
                        predictor_options,
                        default=(
                            predictor_options[:1]
                            if predictor_options
                            else []
                        ),
                        key="logistic_predictors",
                    )

                    if st.button(
                        "Run Logistic Regression",
                        key="logistic_button",
                    ):
                        if not predictors:
                            st.error(
                                "Select at least one predictor."
                            )
                        else:
                            logistic_data = data[
                                [logistic_outcome] + predictors
                            ].copy()

                            for column in logistic_data.columns:
                                logistic_data[column] = pd.to_numeric(
                                    logistic_data[column],
                                    errors="coerce",
                                )

                            logistic_data = logistic_data.replace(
                                [np.inf, -np.inf],
                                np.nan,
                            ).dropna()

                            observed_outcomes = set(
                                logistic_data[
                                    logistic_outcome
                                ].unique()
                            )

                            if observed_outcomes != {0, 1}:
                                st.error(
                                    "The outcome must contain both "
                                    "0 and 1, with no other values. "
                                    "Recode the outcome before analysis."
                                )

                            elif (
                                len(logistic_data)
                                <= len(predictors) + 2
                            ):
                                st.error(
                                    "There are too few complete records "
                                    "for the number of predictors selected."
                                )

                            else:
                                X = logistic_data[
                                    predictors
                                ].astype(float)

                                X = sm.add_constant(
                                    X,
                                    has_constant="add",
                                )

                                y = logistic_data[
                                    logistic_outcome
                                ].astype(int)

                                try:
                                    model = sm.Logit(
                                        y,
                                        X,
                                    ).fit(
                                        disp=False,
                                        maxiter=200,
                                    )

                                    coefficients = model.params
                                    confidence_intervals = (
                                        model.conf_int(
                                            alpha=alpha
                                        )
                                    )
                                    p_values = model.pvalues

                                    results_table = pd.DataFrame(
                                        {
                                            "Coefficient": coefficients,
                                            "Odds Ratio": np.exp(
                                                coefficients
                                            ),
                                            "OR CI Lower": np.exp(
                                                confidence_intervals[0]
                                            ),
                                            "OR CI Upper": np.exp(
                                                confidence_intervals[1]
                                            ),
                                            "p-value": p_values,
                                        }
                                    )

                                    st.write(
                                        "Logistic Regression Results"
                                    )

                                    st.dataframe(
                                        results_table.round(4),
                                        use_container_width=True,
                                    )

                                    st.write(
                                        f"Observations used: "
                                        f"{int(model.nobs)}"
                                    )

                                    st.write(
                                        f"Pseudo R-squared: "
                                        f"{model.prsquared:.4f}"
                                    )

                                    st.write(
                                        f"Log-likelihood: "
                                        f"{model.llf:.4f}"
                                    )

                                    st.caption(
                                        f"Odds ratio confidence intervals: "
                                        f"{confidence_label}. The odds "
                                        "ratios are adjusted for the other "
                                        "predictors included in the model."
                                    )

                                    st.warning(
                                        "Check sample size, model "
                                        "convergence, multicollinearity, "
                                        "linearity of the logit for "
                                        "continuous predictors, and "
                                        "possible separation. Odds ratios "
                                        "are not risk ratios."
                                    )

                                except Exception as error:
                                    st.error(
                                        "The logistic regression could "
                                        "not be fitted. Check the sample "
                                        "size, predictor values, and "
                                        f"model suitability. Details: "
                                        f"{error}"
                                    )

                else:
                    st.info(
                        "At least two numeric columns are needed: "
                        "one binary outcome and one predictor."
                    )

        except Exception as error:
            st.error(
                f"Could not process the dataset: {error}"
            )


# --------------------------------------------------
# TAB 5: DATA COLLECTION FORMS
# --------------------------------------------------

with tab5:
    st.header("Data Collection Forms")

    st.write(
        "Create your own questionnaires and manage your forms. "
        "Forms are associated with your signed-in account."
    )

    current_user_id = st.session_state.user.id

    create_tab, manage_tab = st.tabs(
        ["Create a Form", "My Forms"]
    )

    with create_tab:
        st.subheader("Create a New Questionnaire")

        with st.form("create_custom_form"):
            form_title = st.text_input(
                "Form title",
                placeholder="Example: Health Survey",
            )

            form_description = st.text_area(
                "Form description",
                placeholder="Describe the purpose of this survey",
            )

            question_count = st.number_input(
                "Number of questions",
                min_value=1,
                max_value=30,
                value=3,
                step=1,
            )

            questions = []

            for i in range(int(question_count)):
                st.markdown(f"**Question {i + 1}**")

                label = st.text_input(
                    "Question text",
                    key=f"form_question_label_{i}",
                )

                field_type = st.selectbox(
                    "Question type",
                    [
                        "text",
                        "textarea",
                        "number",
                        "date",
                        "dropdown",
                        "radio",
                        "checkbox",
                        "yes_no",
                    ],
                    format_func=lambda x: {
                        "text": "Short text",
                        "textarea": "Long text",
                        "number": "Number",
                        "date": "Date",
                        "dropdown": "Dropdown menu",
                        "radio": "Radio buttons",
                        "checkbox": "Checkbox choices",
                        "yes_no": "Yes / No",
                    }[x],
                    key=f"form_question_type_{i}",
                )

                required = st.checkbox(
                    "Required question",
                    key=f"form_question_required_{i}",
                )

                options_text = ""

                if field_type in ["dropdown", "radio", "checkbox"]:
                    options_text = st.text_input(
                        "Options (separate with commas)",
                        placeholder="Example: Yes, No, Not sure",
                        key=f"form_question_options_{i}",
                    )

                questions.append(
                    {
                        "label": label.strip(),
                        "field_type": field_type,
                        "required": required,
                        "options_text": options_text,
                    }
                )

                st.divider()

            create_clicked = st.form_submit_button(
                "Create Form",
                type="primary",
            )

        if create_clicked:
            if not form_title.strip():
                st.error("Please enter a form title.")

            elif any(not q["label"] for q in questions):
                st.error("Please enter text for every question.")

            elif any(
                q["field_type"] in ["dropdown", "radio", "checkbox"]
                and len([
                    option.strip()
                    for option in q["options_text"].split(",")
                    if option.strip()
                ]) < 2
                for q in questions
            ):
                st.error(
                    "Dropdown, radio, and checkbox questions "
                    "need at least two options."
                )

            else:
                try:
                    form_result = (
                        supabase.table("forms")
                        .insert(
                            {
                                "owner_id": current_user_id,
                                "title": form_title.strip(),
                                "description": form_description.strip(),
                                "is_published": False,
                            }
                        )
                        .execute()
                    )

                    if not form_result.data:
                        st.error("The form could not be saved.")
                    else:
                        new_form_id = form_result.data[0]["id"]

                        field_rows = []

                        for position, question in enumerate(questions):
                            options = None

                            if question["field_type"] in [
                                "dropdown", "radio", "checkbox"
                            ]:
                                options = [
                                    option.strip()
                                    for option in question["options_text"].split(",")
                                    if option.strip()
                                ]

                            field_rows.append(
                                {
                                    "form_id": new_form_id,
                                    "label": question["label"],
                                    "field_type": question["field_type"],
                                    "options": options,
                                    "required": question["required"],
                                    "position": position,
                                }
                            )

                        (
                            supabase.table("form_fields")
                            .insert(field_rows)
                            .execute()
                        )

                        st.success(
                            "Form created successfully! "
                            "It is saved as a draft."
                        )
                        st.rerun()

                except Exception as error:
                    st.error(
                        "Could not save the form. Check the "
                        "Supabase table structure and access policies. "
                        f"Details: {error}"
                    )

    with manage_tab:
        st.subheader("My Forms")

        try:
            forms_result = (
                supabase.table("forms")
                .select("*")
                .eq("owner_id", current_user_id)
                .order("created_at", desc=True)
                .execute()
            )

            my_forms = forms_result.data or []

            if not my_forms:
                st.info(
                    "You have not created any forms yet. "
                    "Use the Create a Form tab to begin."
                )
            else:
                for form_item in my_forms:
                    with st.expander(form_item["title"]):
                        st.write(
                            form_item.get("description") or
                            "No description provided."
                        )

                        st.write(
                            "Status:",
                            "Published"
                            if form_item.get("is_published")
                            else "Draft",
                        )

                        fields_result = (
                            supabase.table("form_fields")
                            .select("*")
                            .eq("form_id", form_item["id"])
                            .order("position")
                            .execute()
                        )

                        for field in fields_result.data or []:
                            st.write(
                                f'{field["position"] + 1}. '
                                f'{field["label"]} '
                                f'({field["field_type"]})'
                            )

        except Exception as error:
            st.error(
                "Could not load your forms. Check the Supabase "
                f"permissions and database tables. Details: {error}"
            )

# --------------------------------------------------
# 5. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Health Informatics & Data Analysis | Educational use only. "
    "Verify statistical assumptions and interpret results in "
    "the context of the study design."
)
