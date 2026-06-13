import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI-Powered Student Success Analytics",
    page_icon="🎓",
    layout="wide"
)

# ==================================================
# LOAD FILES
# ==================================================

model = joblib.load("student_model.pkl")

data = pd.read_csv("Student_Performance.csv")

model_scores = pd.read_csv("model_scores.csv")

feature_importance = pd.read_csv(
    "feature_importance.csv"
)

# Convert Yes/No to 1/0

data["Extracurricular Activities"] = data[
    "Extracurricular Activities"
].map({
    "Yes": 1,
    "No": 0
})

# ==================================================
# TITLE
# ==================================================

st.title("🎓 AI-Powered Student Success Analytics Platform")

# ==================================================
# KPI DASHBOARD
# ==================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Dataset Records",
    len(data)
)

col2.metric(
    "Features Used",
    5
)

best_model = model_scores.loc[
    model_scores["R2 Score"].idxmax(),
    "Model"
]

col3.metric(
    "Best Model",
    best_model
)

# ==================================================
# PROJECT DESCRIPTION
# ==================================================

st.info(
    """
### AI-Powered Student Success Analytics

This platform predicts student performance using Machine Learning
models trained on a real-world Kaggle dataset.

#### Features

✅ Performance Prediction

✅ Model Comparison

✅ Feature Importance Analysis

✅ Interactive Visualizations

✅ Personalized Recommendations

✅ Dataset Analytics
"""
)

# ==================================================
# MODEL COMPARISON
# ==================================================

st.subheader("🤖 Model Comparison")

st.dataframe(
    model_scores,
    use_container_width=True
)

# ==================================================
# INPUT SECTION
# ==================================================

st.subheader("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:

    hours = st.slider(
        "Hours Studied",
        0,
        12,
        6
    )

    previous = st.slider(
        "Previous Scores",
        0,
        100,
        70
    )

    sleep = st.slider(
        "Sleep Hours",
        0,
        12,
        7
    )

with col2:

    activity = st.selectbox(
        "Extracurricular Activities",
        ["No", "Yes"]
    )

    papers = st.slider(
        "Sample Question Papers Practiced",
        0,
        20,
        5
    )

# ==================================================
# PREDICTION
# ==================================================

if st.button("🚀 Predict Performance"):

    activity_value = 1 if activity == "Yes" else 0

    prediction = model.predict([
        [
            hours,
            previous,
            activity_value,
            sleep,
            papers
        ]
    ])

    score = prediction[0]

    st.success(
        f"Predicted Performance Index: {score:.2f}/100"
    )

    if score >= 90:
        category = "🏆 Excellent"

    elif score >= 75:
        category = "✅ Good"

    elif score >= 60:
        category = "⚠️ Average"

    else:
        category = "❌ Needs Improvement"

    st.subheader(category)

    # Prediction Summary

    st.subheader("📜 Prediction Summary")

    st.write(
        f"Predicted Score: {score:.2f}/100"
    )

    st.write(
        f"Performance Category: {category}"
    )

    # Recommendations

    st.subheader("💡 Recommendations")

    recommendations = []

    if hours < 5:
        recommendations.append(
            "Increase daily study hours."
        )

    if sleep < 6:
        recommendations.append(
            "Maintain 6–8 hours of sleep."
        )

    if papers < 5:
        recommendations.append(
            "Practice more sample question papers."
        )

    if activity_value == 0:
        recommendations.append(
            "Participate in extracurricular activities."
        )

    if recommendations:

        for rec in recommendations:
            st.write("•", rec)

    else:

        st.success(
            "Excellent academic habits detected."
        )

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

st.divider()

st.subheader("📈 Feature Importance Analysis")

st.bar_chart(
    feature_importance.set_index("Feature")
)

st.info(
    """
### Key Insights

📌 Previous Scores are the strongest predictor of student performance.

📌 Study Hours are the second most influential factor.

📌 Academic history plays a major role in future outcomes.

📌 Sleep and extracurricular activities have relatively smaller influence in this dataset.
"""
)

# ==================================================
# DATASET ANALYTICS
# ==================================================

st.divider()

st.header("📊 Dataset Analytics")

graph_option = st.selectbox(
    "Select a Graph",
    [
        "Hours Studied vs Performance Index",
        "Previous Scores vs Performance Index",
        "Sleep Hours vs Performance Index",
        "Sample Papers vs Performance Index"
    ]
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

if graph_option == "Hours Studied vs Performance Index":

    ax.scatter(
        data["Hours Studied"],
        data["Performance Index"]
    )

    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Performance Index")
    ax.set_title(
        "Hours Studied vs Performance Index"
    )

elif graph_option == "Previous Scores vs Performance Index":

    ax.scatter(
        data["Previous Scores"],
        data["Performance Index"]
    )

    ax.set_xlabel("Previous Scores")
    ax.set_ylabel("Performance Index")
    ax.set_title(
        "Previous Scores vs Performance Index"
    )

elif graph_option == "Sleep Hours vs Performance Index":

    ax.scatter(
        data["Sleep Hours"],
        data["Performance Index"]
    )

    ax.set_xlabel("Sleep Hours")
    ax.set_ylabel("Performance Index")
    ax.set_title(
        "Sleep Hours vs Performance Index"
    )

elif graph_option == "Sample Papers vs Performance Index":

    ax.scatter(
        data["Sample Question Papers Practiced"],
        data["Performance Index"]
    )

    ax.set_xlabel(
        "Sample Question Papers Practiced"
    )

    ax.set_ylabel(
        "Performance Index"
    )

    ax.set_title(
        "Sample Papers vs Performance Index"
    )

ax.grid(True)

st.pyplot(fig)

# ==================================================
# DATASET STATISTICS
# ==================================================

with st.expander("📊 Dataset Statistics"):

    st.dataframe(
        data.describe(),
        use_container_width=True
    )

# ==================================================
# DATASET PREVIEW
# ==================================================

with st.expander("📄 View Dataset Sample"):

    st.dataframe(
        data.head(20),
        use_container_width=True
    )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown(
    """
### 🛠 Tech Stack

- Python
- Pandas
- Scikit-Learn
- Random Forest
- Matplotlib
- Streamlit

### 👨‍💻 Project

AI-Powered Student Success Analytics Platform

Built using a real-world Kaggle dataset to predict
student performance and provide data-driven insights.
"""
)