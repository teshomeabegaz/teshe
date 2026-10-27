
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(
    page_title="Health Informatics Statistics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Health Informatics Statistics")

# Two tabs only
statistics_tab, tests_tab = st.tabs([
    "🧮 Python Statistics",
    "📈 Statistical Tests"
])


def parse_values(text):
    """Read numeric values separated by commas, semicolons, spaces or new lines."""
    text = text.replace(";", ",").replace("\n", ",")
    values = []

    for part in text.split(","):
        part = part.strip()
        if not part:
            continue

        for item in part.split():
            value = float(item)
            if not np.isfinite(value):
                raise ValueError("Values must be finite numbers.")
            values.append(value)

    return values


def show_significance(p_value, alpha):
    st.metric("P-value", f"{p_value:.6g}")

    if p_value < alpha:
        st.success(
            f"Statistically significant at α = {alpha}."
        )
    else:
        st.info(
            f"Not statistically significant at α = {alpha}."
        )


# =====================================================
# TAB 1: PYTHON STATISTICS
# =====================================================

with statistics_tab:
    st.header("Descriptive Statistics")

    raw_data = st.text_area(
        "Enter numeric data",
        value="12, 15, 18, 20, 22, 25, 28, 30",
        help="Separate numbers using commas, spaces or new lines."
    )

    try:
        values = parse_values(raw_data)

        if values:
            data = np.array(values, dtype=float)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Count", len(data))
            col2.metric("Mean", f"{np.mean(data):.4f}")
            col3.metric("Median", f"{np.median(data):.4f}")
            col4.metric(
                "Sample SD",
                f"{np.std(data, ddof=1):.4f}"
                if len(data) >= 2 else "Need 2+ values"
            )

            col5, col6, col7, col8 = st.columns(4)

            col5.metric("Minimum", f"{np.min(data):.4f}")
            col6.metric("Maximum", f"{np.max(data):.4f}")
            col7.metric("Range", f"{np.ptp(data):.4f}")
            col8.metric(
                "Sample Variance",
                f"{np.var(data, ddof=1):.4f}"
                if len(data) >= 2 else "Need 2+ values"
            )

            st.subheader("Percentiles")

            percentile_values = [5, 10, 25, 50, 75, 90, 95]

            percentile_table = pd.DataFrame({
                "Percentile": [
                    f"{p}th" for p in percentile_values
                ],
                "Value": [
                    np.percentile(data, p)
                    for p in percentile_values
                ]
            })

            st.dataframe(
                percentile_table,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Histogram")

            bins = st.slider(
                "Number of bins",
                min_value=3,
                max_value=30,
                value=10
            )

            fig, ax = plt.subplots()
            ax.hist(data, bins=bins, edgecolor="black")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Numeric Data")
            ax.grid(axis="y", alpha=0.25)

            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info("Enter numeric data to begin.")

    except ValueError:
        st.error(
            "Please enter valid numbers separated by commas, "
            "spaces or new lines."
        )


# =====================================================
# TAB 2: STATISTICAL TESTS
# =====================================================

with tests_tab:
    st.header("Statistical Tests")

    test_type = st.selectbox(
        "Select a test",
        [
            "Independent samples t-test",
            "Paired samples t-test",
            "Chi-square test of independence",
            "Pearson correlation"
        ]
    )

    alpha = st.selectbox(
        "Significance level",
        [0.01, 0.05, 0.10],
        index=1
    )

    # -------------------------------------------------
    # INDEPENDENT T-TEST
    # -------------------------------------------------

    if test_type == "Independent samples t-test":
        st.write(
            "Compare the means of two independent groups. "
            "Welch's t-test is used."
        )

        col1, col2 = st.columns(2)

        with col1:
            group_a_text = st.text_area(
                "Group A",
                "12, 14, 15, 17, 18",
                key="group_a"
            )

        with col2:
            group_b_text = st.text_area(
                "Group B",
                "16, 18, 19, 21, 22",
                key="group_b"
            )

        if st.button("Run Independent t-test"):
            try:
                a = parse_values(group_a_text)
                b = parse_values(group_b_text)

                if len(a) < 2 or len(b) < 2:
                    st.error("Enter at least two values per group.")
                else:
                    result = stats.ttest_ind(
                        a, b, equal_var=False
                    )

                    st.write(f"Group A mean: {np.mean(a):.4f}")
                    st.write(f"Group B mean: {np.mean(b):.4f}")
                    st.write(
                        f"t statistic: {result.statistic:.4f}"
                    )

                    show_significance(
                        float(result.pvalue), alpha
                    )

            except ValueError:
                st.error("Please enter valid numeric values.")

    # -------------------------------------------------
    # PAIRED T-TEST
    # -------------------------------------------------

    elif test_type == "Paired samples t-test":
        st.write(
            "Compare before-and-after measurements from "
            "the same participants. Values must be paired "
            "in the same order."
        )

        col1, col2 = st.columns(2)

        with col1:
            before_text = st.text_area(
                "Before measurements",
                "120, 125, 118, 130, 122",
                key="before"
            )

        with col2:
            after_text = st.text_area(
                "After measurements",
                "115, 121, 116, 124, 119",
                key="after"
            )

        if st.button("Run Paired t-test"):
            try:
                before = parse_values(before_text)
                after = parse_values(after_text)

                if len(before) < 2 or len(after) < 2:
                    st.error("Enter at least two paired observations.")

                elif len(before) != len(after):
                    st.error(
                        "Before and After must have equal numbers "
                        "of observations."
                    )

                else:
                    result = stats.ttest_rel(before, after)

                    st.write(
                        f"Mean before: {np.mean(before):.4f}"
                    )
                    st.write(
                        f"Mean after: {np.mean(after):.4f}"
                    )
                    st.write(
                        "Mean difference (Before − After): "
                        f"{np.mean(np.array(before) - np.array(after)):.4f}"
                    )
                    st.write(
                        f"t statistic: {result.statistic:.4f}"
                    )

                    show_significance(
                        float(result.pvalue), alpha
                    )

            except ValueError:
                st.error("Please enter valid numeric values.")

    # -------------------------------------------------
    # CHI-SQUARE TEST
    # -------------------------------------------------

    elif test_type == "Chi-square test of independence":
        st.write(
            "Enter observed counts as a contingency table, "
            "with one row per line."
        )

        table_text = st.text_area(
            "Observed counts",
            "20, 30\n25, 25",
            key="chi_table",
            help="Example: 20, 30 on the first row and 25, 25 on the second."
        )

        if st.button("Run Chi-square Test"):
            try:
                rows = []

                for line in table_text.strip().splitlines():
                    if line.strip():
                        row = [
                            int(item.strip())
                            for item in line.replace(";", ",").split(",")
                            if item.strip()
                        ]
                        rows.append(row)

                if (
                    len(rows) < 2
                    or len(rows[0]) < 2
                    or len({len(row) for row in rows}) != 1
                ):
                    st.error(
                        "Enter a rectangular table with at least "
                        "2 rows and 2 columns."
                    )

                elif any(
                    value < 0
                    for row in rows
                    for value in row
                ):
                    st.error("Counts cannot be negative.")

                elif sum(map(sum, rows)) == 0:
                    st.error("At least one count must be greater than zero.")

                else:
                    result = stats.chi2_contingency(
                        rows, correction=False
                    )

                    st.write(
                        f"Chi-square statistic: {result.statistic:.4f}"
                    )
                    st.write(
                        f"Degrees of freedom: {result.dof}"
                    )

                    show_significance(
                        float(result.pvalue), alpha
                    )

                    st.subheader("Expected Counts")

                    st.dataframe(
                        pd.DataFrame(result.expected_freq),
                        use_container_width=True,
                        hide_index=True
                    )

                    if np.any(result.expected_freq < 5):
                        st.warning(
                            "Some expected counts are below 5. "
                            "Consider whether an exact test or "
                            "combining categories is appropriate."
                        )

            except ValueError:
                st.error(
                    "Enter whole-number counts separated by commas, "
                    "with one row per line."
                )

    # -------------------------------------------------
    # PEARSON CORRELATION
    # -------------------------------------------------

    elif test_type == "Pearson correlation":
        st.write(
            "Assess the linear association between two numeric "
            "variables. Values must be paired in the same order."
        )

        col1, col2 = st.columns(2)

        with col1:
            x_text = st.text_area(
                "Variable X",
                "20, 25, 30, 35, 40",
                key="corr_x"
            )

        with col2:
            y_text = st.text_area(
                "Variable Y",
                "50, 55, 65, 70, 80",
                key="corr_y"
            )

        if st.button("Run Pearson Correlation"):
            try:
                x = parse_values(x_text)
                y = parse_values(y_text)

                if len(x) < 3 or len(y) < 3:
                    st.error(
                        "Enter at least three paired observations."
                    )

                elif len(x) != len(y):
                    st.error(
                        "Variable X and Variable Y must have "
                        "the same number of values."
                    )

                elif np.std(x) == 0 or np.std(y) == 0:
                    st.error(
                        "Both variables must have some variation."
                    )

                else:
                    result = stats.pearsonr(x, y)

                    st.metric(
                        "Pearson correlation (r)",
                        f"{result.statistic:.4f}"
                    )

                    show_significance(
                        float(result.pvalue), alpha
                    )

                    fig, ax = plt.subplots()
                    ax.scatter(x, y)

                    slope, intercept = np.polyfit(x, y, 1)
                    line_x = np.linspace(min(x), max(x), 100)

                    ax.plot(
                        line_x,
                        slope * line_x + intercept
                    )

                    ax.set_xlabel("Variable X")
                    ax.set_ylabel("Variable Y")
                    ax.set_title("Pearson Correlation")
                    ax.grid(alpha=0.25)

                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(
                        "Correlation measures linear association; "
                        "it does not prove causation."
                    )

            except ValueError:
                st.error("Please enter valid numeric values.")


st.divider()
st.caption(
    "For learning and research practice. Check statistical assumptions "
    "and interpret results in context."
)
