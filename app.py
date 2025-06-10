import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Citizen Data Visualizer", layout="wide")

st.title("📊 Citizen Health Data Visualizer")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully!")

    # Define column options
    x_options = [
        "citizen_survey_status.phc_code",
        "citizen_survey_status.age",
        "citizen_survey_status.gender",
        "citizen_survey_status.district_code",
    ]

    y_options = [
        "citizen_survey_status.cervical_cancer",
        "citizen_survey_status.breast_cancer",
        "citizen_survey_status.oral_cancer",
        "citizen_survey_status.diabetis",
    ]

    # Sidebar selections
    st.sidebar.header("Select Axes for Plotting")
    x_axis = st.sidebar.selectbox("Choose X-axis column", x_options)
    y_axis = st.sidebar.selectbox("Choose Y-axis column", y_options)

    # Select chart type
    chart_type = st.sidebar.radio("Select Chart Type", ["Countplot", "Pie Chart", "Forecast", "Comparison Bar"])

    st.write(f"### Showing {chart_type} for `{x_axis}` vs `{y_axis}`")

    # Ensure selected columns exist
    if x_axis in df.columns and y_axis in df.columns:
        plot_df = df[[x_axis, y_axis]].dropna()

        # Convert y-axis column to numeric if needed
        try:
            plot_df[y_axis] = pd.to_numeric(plot_df[y_axis], errors='coerce')
        except:
            pass

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "Countplot":
            sns.countplot(data=plot_df, x=x_axis, hue=y_axis, ax=ax)
            plt.xticks(rotation=45)

        elif chart_type == "Pie Chart":
            pie_data = plot_df[y_axis].value_counts()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            plt.title(f"Distribution of {y_axis}")

        elif chart_type == "Forecast":
            # Forecast-like line chart (not real forecasting)
            trend_data = plot_df.groupby(x_axis)[y_axis].mean().reset_index()
            sns.lineplot(data=trend_data, x=x_axis, y=y_axis, marker='o', ax=ax)
            plt.xticks(rotation=45)

        elif chart_type == "Comparison Bar":
            comp_data = plot_df.groupby(x_axis)[y_axis].sum().reset_index()
            sns.barplot(data=comp_data, x=x_axis, y=y_axis, ax=ax)
            plt.xticks(rotation=45)

        st.pyplot(fig)

    else:
        st.error("Selected columns not found in uploaded Excel file.")
else:
    st.info("Please upload a .xlsx file to begin.")
