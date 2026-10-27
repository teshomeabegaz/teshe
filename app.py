
import os
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from supabase import create_client

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="MU-CHS Health Data Analysis Tool",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2rem; font-weight: 700;}
    .subtitle {color: #16806a; font-size: 1.1rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# CONFIGURATION
# Set these in Render Environment Variables or Streamlit
# secrets. Never put a Supabase secret/service-role key here.
# ---------------------------------------------------------
def get_config(name):
    try:
        return st.secrets.get(name, os.getenv(name, ""))
    except Exception:
        return os.getenv(name, "")


SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_KEY = get_config("SUPABASE_KEY")


@st.cache_resource
def get_supabase_client(url, key):
    return create_client(url, key)


supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = get_supabase_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:
        st.error(f"Could not connect to Supabase: {exc}")

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
defaults = {
    "user": None,
    "access_token": None,
    "refresh_token": None,
    "dataset": None,
    "dataset_name": None,
    "analysis_result": None,
    "saved_analysis_id": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Restore the Supabase authenticated session after reruns.
if supabase and st.session_state.access_token and st.session_state.refresh_token:
    try:
        supabase.auth.set_session(
            st.session_state.access_token,
            st.session_state.refresh_token,
        )
    except Exception:
        st.session_state.user = None
        st.session_state.access_token = None
        st.session_state.refresh_token = None


def save_auth_session(response):
    session = getattr(response, "session", None)
    user = getattr(response, "user", None)

    if session and user:
        st.session_state.access_token = session.access_token
        st.session_state.refresh_token = session.refresh_token
        st.session_state.user = {
            "id": user.id,
            "email": user.email,
        }
        return True
    return False


def sign_out():
    try:
        if supabase:
            supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.user = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.rerun()


# ---------------------------------------------------------
# LOGIN AND REGISTRATION
# ---------------------------------------------------------
if not supabase:
    st.title("🏥 MU-CHS Faculty of Public Health")
    st.subheader("Health Informatics Department")
    st.error(
        "Login is not configured yet. Add SUPABASE_URL and "
        "SUPABASE_KEY to your Render environment variables."
    )
    st.info(
        "Use your Supabase project's publishable/anon key. "
        "Do not use a service-role or secret key in this app."
    )
    st.stop()

if not st.session_state.user:
    st.title("🏥 MU-CHS Faculty of Public Health")
    st.subheader("Health Informatics Department")
    st.header("Health Data Analysis Tool")
    st.write(
        "A teaching and learning platform for health data management, "
        "statistical analysis, and clinical research."
    )

    login_tab, register_tab = st.tabs(["Sign in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email address", key="login_email")
            password = st.text_input(
                "Password", type="password", key="login_password"
            )
            login_submit = st.form_submit_button("Sign in", type="primary")

        if login_submit:
            try:
                response = supabase.auth.sign_in_with_password(
                    {"email": email.strip(), "password": password}
                )
                if save_auth_session(response):
                    st.success("Signed in successfully.")
                    st.rerun()
                else:
                    st.error(
                        "Sign-in did not return a session. If you just "
                        "registered, check your email for a confirmation link."
                    )
            except Exception as exc:
                st.error(f"Sign-in failed: {exc}")

    with register_tab:
        with st.form("register_form"):
            new_email = st.text_input("Email address", key="register_email")
            new_password = st.text_input(
                "Password (at least 8 characters)",
                type="password",
                key="register_password",
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="register_confirm",
            )
            register_submit = st.form_submit_button("Create account")

        if register_submit:
            if not new_email.strip() or not new_password:
                st.warning("Enter an email address and password.")
            elif len(new_password) < 8:
                st.warning("Use a password with at least 8 characters.")
            elif new_password != confirm_password:
                st.warning("The passwords do not match.")
            else:
                try:
                    response = supabase.auth.sign_up(
                        {
                            "email": new_email.strip(),
                            "password": new_password,
                        }
                    )
                    if save_auth_session(response):
                        st.success("Account created and signed in.")
                        st.rerun()
                    else:
                        st.success(
                            "Registration submitted. Check your email to "
                            "confirm your account, then sign in."
                        )
                except Exception as exc:
                    st.error(f"Registration failed: {exc}")

    st.caption("For educational use. Use fictional or de-identified data only.")
    st.stop()

# ---------------------------------------------------------
# COMMON HELPERS
# ---------------------------------------------------------
user = st.session_state.user
user_id = user["id"]


def current_dataset():
    return st.session_state.dataset


def parse_numbers(text):
    text = text.replace(";", ",").replace("\n", ",")
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        for item in part.split():
            values.append(float(item))
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Enter at least one valid numeric value.")
    return values


def show_p_value(p_value, alpha):
    st.metric("p-value", f"{p_value:.6g}")
    if p_value < alpha:
        st.success(
            f"Statistically significant at α = {alpha}. "
            "Interpret this result in the context of the study."
        )
    else:
        st.info(
            f"Not statistically significant at α = {alpha}. "
            "This does not prove that there is no effect or association."
        )


def load_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if filename.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="xlrd")
    raise ValueError("Unsupported file type.")


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def save_analysis(title, analysis_type, details):
    """Save analysis metadata/results to the user's own Supabase row."""
    record = {
        "user_id": user_id,
        "title": title.strip(),
        "analysis_type": analysis_type,
        "dataset_name": st.session_state.dataset_name or "",
        "details": details,
    }
    try:
        analysis_id = st.session_state.saved_analysis_id
        if analysis_id:
            response = (
                supabase.table("saved_analyses")
                .update(record)
                .eq("id", analysis_id)
                .eq("user_id", user_id)
                .execute()
            )
        else:
            response = (
                supabase.table("saved_analyses")
                .insert(record)
                .execute()
            )
            if response.data:
                st.session_state.saved_analysis_id = response.data[0]["id"]
        st.success("Analysis saved.")
    except Exception as exc:
        st.error(
            "Could not save. Check that the saved_analyses table and "
            f"Row Level Security policies are configured. Details: {exc}"
        )


def display_dataset_status():
    df = current_dataset()
    if df is None:
        st.warning("No dataset is loaded. Open Analyze Dataset to upload a file.")
        return False
    st.caption(f"Current dataset: {st.session_state.dataset_name}")
    st.write(f"{df.shape[0]:,} rows · {df.shape[1]:,} columns")
    return True


# ---------------------------------------------------------
# HEADER AND NAVIGATION
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🏥 MU-CHS Faculty of Public Health</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Health Informatics Department</div>',
    unsafe_allow_html=True,
)
st.title("Health Data Analysis Tool")
st.write(
    "A teaching and learning platform for health data management, "
    "statistical analysis, and clinical research."
)
st.markdown("**Developed and provided by Teshome Abegaz**")
st.caption(
    "For educational use. Use fictional or de-identified data only. "
    "Do not upload identifiable patient information."
)

with st.sidebar:
    st.markdown("### Account")
    st.write(f"Signed in: **{user['email']}**")
    if st.button("Sign out", use_container_width=True):
        sign_out()

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "📁 Analyze Dataset",
            "🧮 Python Statistics",
            "📈 Statistical Tests",
            "💾 Saved Analyses",
        ],
        label_visibility="collapsed",
    )

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
if page == "📊 Dashboard":
    st.header("📊 Dashboard")
    st.write(
        "Welcome to the Health Data Analysis Tool. Upload a dataset, "
        "explore descriptive statistics, and perform common statistical tests."
    )

    c1, c2, c3 = st.columns(3)
    df = current_dataset()

    with c1:
        st.metric("Dataset", "Loaded" if df is not None else "Not loaded")
    with c2:
        st.metric("Rows", f"{df.shape[0]:,}" if df is not None else "—")
    with c3:
        st.metric("Columns", f"{df.shape[1]:,}" if df is not None else "—")

    st.subheader("Available tools")
    st.markdown(
        """
        - **Analyze Dataset:** Upload CSV, XLSX, or XLS files and preview data.
        - **Python Statistics:** Calculate descriptive statistics and create histograms.
        - **Statistical Tests:** Calculate confidence intervals, compare means,
          test associations, calculate correlations, and fit binary logistic regression.
        - **Saved Analyses:** Reopen and update saved analysis records.
        """
    )

# ---------------------------------------------------------
# ANALYZE DATASET
# ---------------------------------------------------------
elif page == "📁 Analyze Dataset":
    st.header("📁 Analyze Dataset")
    st.write(
        "Upload a CSV or Excel dataset. Maximum file size: 200 MB per file."
    )
    uploaded_file = st.file_uploader(
        "Choose a dataset",
        type=["csv", "xlsx", "xls"],
        max_upload_size=200,
        key="dataset_uploader",
    )

    if uploaded_file is not None:
        try:
            df = load_uploaded_file(uploaded_file)
            st.session_state.dataset = df
            st.session_state.dataset_name = uploaded_file.name
            st.session_state.analysis_result = None
            st.session_state.saved_analysis_id = None
            st.success(f"Loaded {uploaded_file.name} successfully.")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")

    df = current_dataset()
    if df is not None:
        st.subheader("Dataset preview")
        st.dataframe(df.head(100), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

        st.subheader("Column information")
        info_df = pd.DataFrame(
            {
                "Column": df.columns.astype(str),
                "Data type": [str(dtype) for dtype in df.dtypes],
                "Missing values": df.isna().sum().values,
                "Unique values": df.nunique(dropna=True).values,
            }
        )
        st.dataframe(info_df, use_container_width=True)

        st.subheader("Download dataset")
        st.download_button(
            "Download loaded data as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="dataset_export.csv",
            mime="text/csv",
        )

        if st.button("Clear current dataset"):
            st.session_state.dataset = None
            st.session_state.dataset_name = None
            st.session_state.analysis_result = None
            st.session_state.saved_analysis_id = None
            st.rerun()

# ---------------------------------------------------------
# PYTHON STATISTICS
# ---------------------------------------------------------
elif page == "🧮 Python Statistics":
    st.header("🧮 Python Statistics")
    st.write("Calculate descriptive statistics from typed values or a dataset.")

    source = st.radio(
        "Choose data source",
        ["Use uploaded dataset", "Enter numbers manually"],
        horizontal=True,
    )

    if source == "Enter numbers manually":
        text_values = st.text_area(
            "Enter numbers separated by commas, spaces, or new lines",
            "12, 15, 18, 20, 22, 25, 28, 30",
        )
        try:
            values = parse_numbers(text_values)
        except Exception as exc:
            st.error(str(exc))
            values = np.array([])
        column_name = "Manual data"
    else:
        df = current_dataset()
        if df is None:
            st.warning("Upload a dataset first using Analyze Dataset.")
            values = np.array([])
            column_name = ""
        else:
            cols = numeric_columns(df)
            if not cols:
                st.warning("No numeric columns were found in the dataset.")
                values = np.array([])
                column_name = ""
            else:
                column_name = st.selectbox("Select numeric column", cols)
                values = pd.to_numeric(
                    df[column_name], errors="coerce"
                ).dropna().to_numpy(dtype=float)

    if len(values) > 0:
        st.subheader("Descriptive statistics")
        clean = values[np.isfinite(values)]
        count = len(clean)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Valid observations", f"{count:,}")
        c2.metric("Mean", f"{np.mean(clean):.4f}")
        c3.metric("Median", f"{np.median(clean):.4f}")
        c4.metric(
            "Sample SD",
            f"{np.std(clean, ddof=1):.4f}" if count > 1 else "Need 2+ values",
        )

        summary = {
            "Count": count,
            "Mean": np.mean(clean),
            "Median": np.median(clean),
            "Minimum": np.min(clean),
            "Maximum": np.max(clean),
            "Range": np.ptp(clean),
            "Sample variance": np.var(clean, ddof=1) if count > 1 else np.nan,
            "Sample standard deviation": (
                np.std(clean, ddof=1) if count > 1 else np.nan
            ),
            "Standard error of mean": (
                stats.sem(clean) if count > 1 else np.nan
            ),
            "Skewness": stats.skew(clean) if count > 2 else np.nan,
        }
        st.dataframe(
            pd.DataFrame(
                {"Statistic": summary.keys(), "Value": summary.values()}
            ),
            use_container_width=True,
        )

        st.subheader("Percentiles")
        percentile_values = [5, 10, 25, 50, 75, 90, 95]
        percentile_table = pd.DataFrame(
            {
                "Percentile": percentile_values,
                "Value": [
                    np.percentile(clean, p) for p in percentile_values
                ],
            }
        )
        st.dataframe(percentile_table, use_container_width=True)

        bins = st.slider("Histogram bins", 3, 50, 10)
        fig, ax = plt.subplots()
        ax.hist(clean, bins=bins, edgecolor="black")
        ax.set_title(f"Distribution of {column_name}")
        ax.set_xlabel(str(column_name))
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)

        st.session_state.analysis_result = {
            "type": "Descriptive statistics",
            "column": str(column_name),
            "summary": {
                key: (None if pd.isna(value) else float(value))
                for key, value in summary.items()
            },
            "percentiles": percentile_table.to_dict(orient="records"),
        }

        with st.form("save_statistics_form"):
            save_title = st.text_input(
                "Analysis title", "Descriptive statistics"
            )
            save_button = st.form_submit_button("Save analysis")
        if save_button:
            save_analysis(
                save_title,
                "Descriptive statistics",
                st.session_state.analysis_result,
            )

# ---------------------------------------------------------
# STATISTICAL TESTS
# ---------------------------------------------------------
elif page == "📈 Statistical Tests":
    st.header("Statistical Tests for Health Research")
    st.write(
        "Upload a dataset to calculate confidence intervals, compare means, "
        "test associations, and fit binary logistic regression models."
    )
    st.info(
        "Choose variables carefully. Statistical tests require appropriate "
        "study designs and assumptions. Results from small or biased samples "
        "may be unreliable."
    )

    df = current_dataset()
    if df is None:
        st.warning("Upload a CSV or Excel dataset first in Analyze Dataset.")
        st.stop()

    st.caption(
        f"Dataset: {st.session_state.dataset_name} · "
        f"{df.shape[0]:,} rows · {df.shape[1]:,} columns"
    )

    test_name = st.selectbox(
        "Choose a statistical procedure",
        [
            "Confidence interval for a mean",
            "Independent samples t-test (Welch)",
            "Paired samples t-test",
            "Chi-square test of independence",
            "Pearson correlation",
            "Binary logistic regression",
        ],
    )
    alpha_label = st.selectbox("Significance level", ["0.05", "0.01", "0.10"])
    alpha = float(alpha_label)
    result = None

    # Confidence interval for one mean
    if test_name == "Confidence interval for a mean":
        nums = numeric_columns(df)
        if nums:
            col = st.selectbox("Numeric variable", nums, key="ci_col")
            x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
            if len(x) >= 2:
                mean = float(np.mean(x))
                se = float(stats.sem(x))
                lower, upper = stats.t.interval(
                    1 - alpha, df=len(x) - 1, loc=mean, scale=se
                )
                st.metric("Sample mean", f"{mean:.4f}")
                st.write(
                    f"{(1-alpha)*100:.0f}% confidence interval: "
                    f"({lower:.4f}, {upper:.4f})"
                )
                result = {
                    "test": test_name, "variable": str(col),
                    "n": len(x), "mean": mean,
                    "confidence_level": 1-alpha,
                    "lower": float(lower), "upper": float(upper),
                }
            else:
                st.warning("At least two non-missing observations are required.")
        else:
            st.warning("No numeric columns were found.")

    # Independent samples Welch t-test
    elif test_name == "Independent samples t-test (Welch)":
        nums = numeric_columns(df)
        cats = df.select_dtypes(exclude=np.number).columns.tolist()
        group_options = list(df.columns)
        if nums and group_options:
            outcome = st.selectbox("Numeric outcome", nums, key="ind_outcome")
            group_col = st.selectbox("Grouping variable", group_options, key="ind_group")
            levels = df[group_col].dropna().unique().tolist()
            if len(levels) >= 2:
                a_level = st.selectbox("Group A", levels, key="ind_a")
                remaining = [v for v in levels if v != a_level]
                if remaining:
                    b_level = st.selectbox("Group B", remaining, key="ind_b")
                    a = pd.to_numeric(
                        df.loc[df[group_col] == a_level, outcome], errors="coerce"
                    ).dropna().to_numpy()
                    b = pd.to_numeric(
                        df.loc[df[group_col] == b_level, outcome], errors="coerce"
                    ).dropna().to_numpy()
                    if len(a) >= 2 and len(b) >= 2:
                        test = stats.ttest_ind(a, b, equal_var=False)
                        st.write(f"Group A n = {len(a)}, mean = {np.mean(a):.4f}")
                        st.write(f"Group B n = {len(b)}, mean = {np.mean(b):.4f}")
                        st.write(f"Welch t statistic = {test.statistic:.4f}")
                        show_p_value(float(test.pvalue), alpha)
                        result = {
                            "test": test_name, "outcome": str(outcome),
                            "group_variable": str(group_col),
                            "group_a": str(a_level), "group_b": str(b_level),
                            "n_a": len(a), "n_b": len(b),
                            "mean_a": float(np.mean(a)),
                            "mean_b": float(np.mean(b)),
                            "statistic": float(test.statistic),
                            "p_value": float(test.pvalue),
                        }
                    else:
                        st.warning("Each group needs at least two numeric observations.")
            else:
                st.warning("The grouping variable must have at least two groups.")
        else:
            st.warning("Select a dataset with a numeric outcome and grouping variable.")

    # Paired samples t-test
    elif test_name == "Paired samples t-test":
        nums = numeric_columns(df)
        if len(nums) >= 2:
            col_a = st.selectbox("First measurement", nums, key="paired_a")
            col_b = st.selectbox(
                "Second measurement", [c for c in nums if c != col_a], key="paired_b"
            )
            paired = df[[col_a, col_b]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(paired) >= 2:
                test = stats.ttest_rel(paired[col_a], paired[col_b])
                st.write(f"Complete pairs: {len(paired)}")
                st.write(f"Mean first measurement: {paired[col_a].mean():.4f}")
                st.write(f"Mean second measurement: {paired[col_b].mean():.4f}")
                st.write(f"Paired t statistic = {test.statistic:.4f}")
                show_p_value(float(test.pvalue), alpha)
                result = {
                    "test": test_name, "first_variable": str(col_a),
                    "second_variable": str(col_b), "n_pairs": len(paired),
                    "statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                }
            else:
                st.warning("At least two complete pairs are required.")
        else:
            st.warning("At least two numeric columns are required.")

    # Chi-square test
    elif test_name == "Chi-square test of independence":
        cols = list(df.columns)
        if len(cols) >= 2:
            col_a = st.selectbox("First categorical variable", cols, key="chi_a")
            col_b = st.selectbox(
                "Second categorical variable", [c for c in cols if c != col_a], key="chi_b"
            )
            subset = df[[col_a, col_b]].dropna()
            if subset.empty:
                st.warning("No complete observations for these columns.")
            else:
                table = pd.crosstab(subset[col_a], subset[col_b])
                st.subheader("Contingency table")
                st.dataframe(table, use_container_width=True)
                if table.shape[0] >= 2 and table.shape[1] >= 2:
                    chi2, p_value, dof, expected = stats.chi2_contingency(table)
                    st.write(f"Chi-square statistic = {chi2:.4f}")
                    st.write(f"Degrees of freedom = {dof}")
                    show_p_value(float(p_value), alpha)
                    if (expected < 5).any():
                        st.warning(
                            "Some expected cell counts are below 5. "
                            "Consider whether an exact test or category grouping "
                            "is appropriate for your study."
                        )
                    result = {
                        "test": test_name, "variables": [str(col_a), str(col_b)],
                        "chi_square": float(chi2), "degrees_of_freedom": int(dof),
                        "p_value": float(p_value),
                        "expected_counts": expected.tolist(),
                    }
                else:
                    st.warning("Both variables must have at least two categories.")

    # Pearson correlation
    elif test_name == "Pearson correlation":
        nums = numeric_columns(df)
        if len(nums) >= 2:
            col_x = st.selectbox("Variable X", nums, key="pearson_x")
            col_y = st.selectbox(
                "Variable Y", [c for c in nums if c != col_x], key="pearson_y"
            )
            pair = df[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) >= 3 and pair[col_x].nunique() > 1 and pair[col_y].nunique() > 1:
                test = stats.pearsonr(pair[col_x], pair[col_y])
                st.metric("Pearson correlation (r)", f"{test.statistic:.4f}")
                show_p_value(float(test.pvalue), alpha)
                fig, ax = plt.subplots()
                ax.scatter(pair[col_x], pair[col_y])
                ax.set_xlabel(str(col_x))
                ax.set_ylabel(str(col_y))
                ax.set_title("Pearson correlation")
                st.pyplot(fig)
                plt.close(fig)
                result = {
                    "test": test_name, "x": str(col_x), "y": str(col_y),
                    "n": len(pair), "r": float(test.statistic),
                    "p_value": float(test.pvalue),
                }
            else:
                st.warning(
                    "At least three complete observations and variation in both "
                    "variables are required."
                )
        else:
            st.warning("At least two numeric columns are required.")

    # Binary logistic regression
    elif test_name == "Binary logistic regression":
        all_cols = list(df.columns)
        if len(all_cols) >= 2:
            outcome = st.selectbox("Binary outcome (dependent variable)", all_cols)
            predictors = st.multiselect(
                "Predictor variables (independent variables)",
                [c for c in all_cols if c != outcome],
            )
            st.caption(
                "The outcome must have exactly two observed categories. "
                "Categorical predictors are encoded as indicator variables."
            )
            if predictors and st.button("Fit logistic regression", type="primary"):
                model_df = df[[outcome] + predictors].dropna().copy()
                levels = model_df[outcome].unique().tolist()
                if len(levels) != 2:
                    st.error("The outcome must have exactly two observed categories.")
                elif len(model_df) <= len(predictors) + 2:
                    st.error("There are too few complete observations for this model.")
                else:
                    try:
                        y = (model_df[outcome] == levels[1]).astype(int)
                        x = pd.get_dummies(
                            model_df[predictors], drop_first=True, dtype=float
                        )
                        x = sm.add_constant(x, has_constant="add")
                        if x.shape[1] >= len(model_df):
                            st.error(
                                "There are too many model parameters for the "
                                "available observations."
                            )
                        elif y.nunique() != 2:
                            st.error("The outcome must contain both categories.")
                        else:
                            model = sm.Logit(y, x).fit(disp=False)
                            params = model.params
                            conf = model.conf_int()
                            odds_table = pd.DataFrame(
                                {
                                    "Coefficient": params,
                                    "Odds ratio": np.exp(params),
                                    "OR CI lower": np.exp(conf[0]),
                                    "OR CI upper": np.exp(conf[1]),
                                    "p-value": model.pvalues,
                                }
                            )
                            st.write(f"Outcome reference category: {levels[0]}")
                            st.write(f"Modeled category: {levels[1]}")
                            st.write(f"Observations used: {int(model.nobs)}")
                            st.write(f"Pseudo R-squared: {model.prsquared:.4f}")
                            st.dataframe(odds_table, use_container_width=True)
                            result = {
                                "test": test_name,
                                "outcome": str(outcome),
                                "modeled_category": str(levels[1]),
                                "predictors": [str(c) for c in predictors],
                                "n": int(model.nobs),
                                "pseudo_r_squared": float(model.prsquared),
                                "coefficients": {
                                    str(k): float(v) for k, v in params.items()
                                },
                                "odds_ratios": {
                                    str(k): float(v)
                                    for k, v in np.exp(params).items()
                                },
                                "p_values": {
                                    str(k): float(v)
                                    for k, v in model.pvalues.items()
                                },
                            }
                    except Exception as exc:
                        st.error(
                            "The model could not be fitted. Check for "
                            "perfect prediction, sparse categories, collinearity, "
                            f"or insufficient data. Details: {exc}"
                        )

    if result is not None:
        st.session_state.analysis_result = result
        st.subheader("Save this analysis")
        with st.form("save_test_form"):
            analysis_title = st.text_input("Analysis title", test_name)
            save_button = st.form_submit_button("Save analysis")
        if save_button:
            save_analysis(analysis_title, test_name, result)

# ---------------------------------------------------------
# SAVED ANALYSES
# ---------------------------------------------------------
elif page == "💾 Saved Analyses":
    st.header("💾 Saved Analyses")
    st.write("Only records belonging to your signed-in account should be returned.")

    try:
        response = (
            supabase.table("saved_analyses")
            .select("id,title,analysis_type,dataset_name,details,created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        records = response.data or []

        if not records:
            st.info("You have no saved analyses yet.")
        else:
            for record in records:
                with st.expander(
                    f"{record.get('title', 'Untitled')} — "
                    f"{record.get('analysis_type', 'Analysis')}"
                ):
                    st.caption(f"Dataset: {record.get('dataset_name') or 'Not recorded'}")
                    st.json(record.get("details") or {})

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load result", key=f"load_{record['id']}"):
                            st.session_state.analysis_result = record.get("details")
                            st.session_state.saved_analysis_id = record["id"]
                            st.success("Loaded saved result into the current session.")
                    with col2:
                        if st.button("Delete", key=f"delete_{record['id']}"):
                            try:
                                (
                                    supabase.table("saved_analyses")
                                    .delete()
                                    .eq("id", record["id"])
                                    .eq("user_id", user_id)
                                    .execute()
                                )
                                st.success("Saved analysis deleted.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not delete analysis: {exc}")
    except Exception as exc:
        st.error(
            "Could not retrieve saved analyses. Create the saved_analyses "
            f"table and configure Row Level Security. Details: {exc}"
        )

st.divider()
st.caption(
    "MU-CHS Faculty of Public Health · Health Informatics Department · "
    "Educational use only. Use fictional or de-identified data."
)
