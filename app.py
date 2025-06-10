import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Citizen Health Visualizer", layout="wide")

st.title("📊 Citizen Health Data Visualizer")

# File uploader
uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Read and clean data
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove spaces from column names

    st.success("✅ File uploaded successfully.")
    st.write("📄 Columns in file:", df.columns.tolist())  # For debug

    # Dropdown options
    x_options = ["phc_code", "age", "gender", "district_code"]
    y_options = ["cervical_cancer", "breast_cancer", "oral_cancer", "diabetis"]

    # Sidebar UI
    st.sidebar.header("📌 Chart Settings")
    x_axis = st.sidebar.selectbox("Select X-axis", x_options)
    y_axis = st.sidebar.selectbox("Select Y-axis (0 = No, 1 = Yes)", y_options)
    chart_type = st.sidebar.radio("Choose Chart Type", ["Countplot", "Pie Chart", "Trend (Line)", "Comparison Bar"])

    # Drop missing values
    df = df[[x_axis, y_axis]].dropna()

    # Convert gender to label for better plotting
    if x_axis == "gender":
        df["gender_label"] = df["gender"].apply(lambda g: "Male" if g == 0 else "Female")
        x_plot = "gender_label"
    else:
        x_plot = x_axis

    # Ensure y-axis is numeric
    df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce')
    df = df.dropna()

    # Plot
    st.subheader(f"📈 {chart_type}: `{x_axis}` vs `{y_axis}`")
    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "Countplot":
        sns.countplot(data=df, x=x_plot, hue=y_axis, ax=ax)
        ax.set_title(f"Countplot: {x_axis} vs {y_axis}")
        plt.xticks(rotation=45)

    elif chart_type == "Pie Chart":
        pie_data = df[y_axis].value_counts().sort_index()
        labels = ["No", "Yes"]
        ax.pie(pie_data, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title(f"Pie Chart of {y_axis}")

    elif chart_type == "Trend (Line)":
        trend_df = df.groupby(x_plot)[y_axis].mean().reset_index()
        sns.lineplot(data=trend_df, x=x_plot, y=y_axis, marker="o", ax=ax)
        ax.set_title(f"Trend of {y_axis} by {x_axis}")
        plt.xticks(rotation=45)

    elif chart_type == "Comparison Bar":
        bar_df = df.groupby(x_plot)[y_axis].sum().reset_index()
        sns.barplot(data=bar_df, x=x_plot, y=y_axis, ax=ax)
        ax.set_title(f"Total {y_axis} Cases by {x_axis}")
        plt.xticks(rotation=45)

    st.pyplot(fig)

else:
    st.info("📤 Please upload an Excel file to begin.")
