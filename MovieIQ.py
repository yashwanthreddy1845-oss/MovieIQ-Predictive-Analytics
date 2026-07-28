import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import ast
from scipy.stats import ttest_ind, chi2_contingency


# Page Configuration
st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 MovieIQ")
st.subheader("Predictive Analytics on Film Success")

st.write(
    "MovieIQ analyses movie data and predicts whether a movie "
    "is likely to be successful."
)

st.info(
    "A movie is considered successful when its revenue is greater "
    "than its budget."
)


# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")

    # Remove invalid financial records
    df = df[
        (df["budget"] > 0) &
        (df["revenue"] > 0)
    ].copy()

    # Create target variable
    df["success"] = (
        df["revenue"] > df["budget"]
    ).astype(int)

    # Process genres
    def extract_genres(value):
        try:
            genres = ast.literal_eval(value)
            return [genre["name"] for genre in genres]
        except:
            return []

    df["genre_list"] = df["genres"].apply(extract_genres)

    return df


df = load_data()

# Load trained model
model = joblib.load("model.pkl")
# -----------------------------
# Dataset Overview
# -----------------------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Movies", len(df))
col2.metric("Successful Movies", int(df["success"].sum()))
col3.metric("Success Rate", f"{df['success'].mean() * 100:.1f}%")


# -----------------------------
# Sidebar Filters
# -----------------------------

st.sidebar.header("Movie Filters")

all_genres = sorted({
    genre
    for genres in df["genre_list"]
    for genre in genres
})

selected_genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + all_genres
)

min_vote = st.sidebar.slider(
    "Minimum Vote Average",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1
)


# -----------------------------
# Apply Filters
# -----------------------------

filtered_df = df[
    df["vote_average"] >= min_vote
].copy()

if selected_genre != "All":
    filtered_df = filtered_df[
        filtered_df["genre_list"].apply(
            lambda genres: selected_genre in genres
        )
    ]


# -----------------------------
# Display Filtered Movies
# -----------------------------

st.header("🎞️ Filtered Movies")

st.write(f"Showing **{len(filtered_df)}** movies")

st.dataframe(
    filtered_df[
        [
            "title",
            "budget",
            "revenue",
            "popularity",
            "runtime",
            "vote_average",
            "success"
        ]
    ],
    use_container_width=True
)
# -----------------------------
# Movie Success Prediction
# -----------------------------

st.header("🎯 Predict Movie Success")

st.write(
    "Enter the movie details below to predict whether "
    "the movie is likely to be successful."
)

budget = st.number_input(
    "Budget",
    min_value=0.0,
    value=50000000.0,
    step=1000000.0
)

popularity = st.number_input(
    "Popularity",
    min_value=0.0,
    value=50.0,
    step=1.0
)

runtime = st.number_input(
    "Runtime (minutes)",
    min_value=1,
    value=120,
    step=1
)

vote_average = st.slider(
    "Vote Average",
    min_value=0.0,
    max_value=10.0,
    value=6.0,
    step=0.1
)

if st.button("Predict Success"):

    input_data = pd.DataFrame(
        [[budget, popularity, runtime, vote_average]],
        columns=[
            "budget",
            "popularity",
            "runtime",
            "vote_average"
        ]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("✅ Movie is predicted to be SUCCESSFUL")
    else:
        st.error("❌ Movie is predicted to be NOT SUCCESSFUL")

    st.write(
        f"Predicted probability of success: **{probability * 100:.2f}%**"
    )