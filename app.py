
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_PATH,
    "tourism_feature_engineered.csv"
)

df = pd.read_csv(DATA_PATH)

# Convert VisitDate
df["VisitDate"] = pd.to_datetime(
    df["VisitDate"],
    errors="coerce"
)

# ============================================================
# LOAD REGRESSION MODEL
# ============================================================

with open(
    os.path.join(BASE_PATH, "random_forest_model.pkl"),
    "rb"
) as file:
    regression_model = pickle.load(file)

with open(
    os.path.join(BASE_PATH, "preprocessor.pkl"),
    "rb"
) as file:
    regression_preprocessor = pickle.load(file)

# ============================================================
# LOAD CLASSIFICATION MODEL
# ============================================================

with open(
    os.path.join(
        BASE_PATH,
        "logistic_regression_classifier.pkl"
    ),
    "rb"
) as file:
    classification_model = pickle.load(file)

with open(
    os.path.join(
        BASE_PATH,
        "classification_preprocessor.pkl"
    ),
    "rb"
) as file:
    classification_preprocessor = pickle.load(file)

# ============================================================
# LOAD RECOMMENDATION SYSTEM
# ============================================================

with open(
    os.path.join(
        BASE_PATH,
        "train_user_item_matrix.pkl"
    ),
    "rb"
) as file:
    train_user_item_matrix = pickle.load(file)

with open(
    os.path.join(
        BASE_PATH,
        "item_similarity_matrix.pkl"
    ),
    "rb"
) as file:
    item_similarity_df = pickle.load(file)

recommendation_evaluation = pd.read_csv(
    os.path.join(
        BASE_PATH,
        "recommendation_evaluation.csv"
    )
)

# ============================================================
# HELPER FUNCTION
# ============================================================

def recommend_attractions(user_id, top_n=5):

    if user_id not in train_user_item_matrix.index:
        return pd.DataFrame(
            columns=[
                "AttractionId",
                "Attraction",
                "RecommendationScore"
            ]
        )

    user_ratings = train_user_item_matrix.loc[user_id]

    interacted_attractions = user_ratings[
        user_ratings > 0
    ].index.tolist()

    recommendation_scores = {}

    for attraction_id in interacted_attractions:

        user_rating = user_ratings[attraction_id]

        similar_attractions = item_similarity_df.loc[
            attraction_id
        ]

        for similar_id, similarity_score in (
            similar_attractions.items()
        ):

            if similar_id in interacted_attractions:
                continue

            score = user_rating * similarity_score

            recommendation_scores[similar_id] = (
                recommendation_scores.get(
                    similar_id,
                    0
                ) + score
            )

    ranked = sorted(
        recommendation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    recommendations = pd.DataFrame(
        ranked,
        columns=[
            "AttractionId",
            "RecommendationScore"
        ]
    )

    attraction_names = (
        df[
            ["AttractionId", "Attraction"]
        ]
        .drop_duplicates("AttractionId")
    )

    recommendations = recommendations.merge(
        attraction_names,
        on="AttractionId",
        how="left"
    )

    return recommendations[
        [
            "AttractionId",
            "Attraction",
            "RecommendationScore"
        ]
    ]


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🌍 Tourism Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Data Analysis",
    "🧮 SQL Insights",
        "⭐ Rating Prediction",
        "🎯 Visit Mode Prediction",
        "🤖 Recommendations",
        "📈 Model Performance",
        "🔍 Feature Importance",
        "🗃️ Dataset Explorer"
    ]
)

# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.title(
        "🌍 Tourism Experience Analytics"
    )

    st.subheader(
        "Classification, Prediction, and Recommendation System"
    )

    st.write(
        """
        This application analyzes tourism experience data
        and provides machine-learning based predictions and
        personalized attraction recommendations.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Users",
            f"{df['UserId'].nunique():,}"
        )

    with col3:
        st.metric(
            "Attractions",
            f"{df['AttractionId'].nunique():,}"
        )

    with col4:
        st.metric(
            "Average Rating",
            f"{df['Rating'].mean():.2f}"
        )

    st.markdown("---")

    st.subheader("Project Objectives")

    st.markdown(
        """
        **⭐ Regression**
        - Predict attraction ratings.

        **🎯 Classification**
        - Predict the user's visit mode.

        **🤖 Recommendation**
        - Recommend personalized attractions.

        **📊 Analytics**
        - Discover tourism trends and visitor behavior.
        """
    )

# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    st.title("📊 Tourism Data Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Visit Mode Distribution")

        mode_counts = (
            df["VisitModeName"]
            .value_counts()
        )

        st.bar_chart(mode_counts)

    with col2:

        st.subheader("Rating Distribution")

        rating_counts = (
            df["Rating"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(rating_counts)

    st.subheader("Top 10 Attractions")

    top_attractions = (
        df["Attraction"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(top_attractions)

    st.subheader("Tourism by Season")

    season_counts = (
        df["VisitSeason"]
        .value_counts()
    )

    st.bar_chart(season_counts)

# ============================================================
# RATING PREDICTION
# ============================================================

elif page == "🧮 SQL Insights":
    st.title("🧮 SQL Insights")
    st.markdown("### Tourism Analytics using SQL")

    sql_folder = os.path.join(BASE_PATH, "sql_results")

    # Load SQL analysis results
    mode_sql = pd.read_csv(
        os.path.join(sql_folder, "sql_rating_by_visit_mode.csv")
    )

    attractions_sql = pd.read_csv(
        os.path.join(sql_folder, "sql_top_attractions.csv")
    )

    city_sql = pd.read_csv(
        os.path.join(sql_folder, "sql_city_analysis.csv")
    )

    type_sql = pd.read_csv(
        os.path.join(sql_folder, "sql_attraction_type_analysis.csv")
    )

    season_sql = pd.read_csv(
        os.path.join(sql_folder, "sql_seasonal_analysis.csv")
    )

    # -----------------------------------------------------
    # Visit Mode Analysis
    # -----------------------------------------------------

    st.subheader("🎯 Rating Analysis by Visit Mode")
    st.dataframe(mode_sql, use_container_width=True)

    # -----------------------------------------------------
    # Top Attractions
    # -----------------------------------------------------

    st.subheader("🏆 Top 10 Most Visited Attractions")
    st.dataframe(attractions_sql, use_container_width=True)

    # -----------------------------------------------------
    # City Analysis
    # -----------------------------------------------------

    st.subheader("🌍 Top 10 Cities by Tourism Visits")
    st.dataframe(city_sql, use_container_width=True)

    # -----------------------------------------------------
    # Attraction Type Analysis
    # -----------------------------------------------------

    st.subheader("🎡 Attraction Type Analysis")
    st.dataframe(type_sql, use_container_width=True)

    # -----------------------------------------------------
    # Seasonal Analysis
    # -----------------------------------------------------

    st.subheader("🌤️ Seasonal Tourism Analysis")
    st.dataframe(season_sql, use_container_width=True)

    st.success("✅ SQL-based tourism analysis loaded successfully.")

elif page == "⭐ Rating Prediction":

    st.title("⭐ Attraction Rating Prediction")

    st.write(
        "Enter tourism details to predict the expected attraction rating."
    )

    col1, col2 = st.columns(2)

    with col1:

        visit_year = st.selectbox(
            "Visit Year",
            sorted(
                df["VisitYear"].dropna().unique()
            )
        )

        visit_month = st.selectbox(
            "Visit Month",
            list(range(1, 13))
        )

        visit_mode = st.selectbox(
            "Visit Mode",
            sorted(
                df["VisitModeName"]
                .dropna()
                .unique()
            )
        )

        city = st.selectbox(
            "City",
            sorted(
                df["CityName"]
                .dropna()
                .unique()
            )
        )

    with col2:

        attraction = st.selectbox(
            "Attraction",
            sorted(
                df["Attraction"]
                .dropna()
                .unique()
            )
        )

        attraction_type = st.selectbox(
            "Attraction Type",
            sorted(
                df["AttractionType"]
                .dropna()
                .unique()
            )
        )

        address = st.selectbox(
            "Attraction Address",
            sorted(
                df["AttractionAddress"]
                .dropna()
                .unique()
            )
        )

        season = st.selectbox(
            "Visit Season",
            sorted(
                df["VisitSeason"]
                .dropna()
                .unique()
            )
        )

    selected_attraction = df[
        df["Attraction"] == attraction
    ]

    if len(selected_attraction) > 0:

        attraction_popularity = (
            selected_attraction[
                "AttractionPopularity"
            ].iloc[0]
        )

    else:

        attraction_popularity = 0

    user_visit_count = st.number_input(
        "User Visit Count",
        min_value=1,
        value=1
    )

    if st.button(
        "⭐ Predict Rating",
        use_container_width=True
    ):

        selected_city = df[
            df["CityName"] == city
        ].iloc[0]

        selected_attraction_row = df[
            df["Attraction"] == attraction
        ].iloc[0]

        input_data = pd.DataFrame({
            "VisitYear": [visit_year],
            "VisitMonth": [visit_month],
            "VisitModeName": [visit_mode],
            "ContinentId": [
                selected_city["ContinentId"]
            ],
            "RegionId": [
                selected_city["RegionId"]
            ],
            "CountryId": [
                selected_city["CountryId"]
            ],
            "CityName": [city],
            "Attraction": [attraction],
            "AttractionType": [attraction_type],
            "AttractionAddress": [address],
            "VisitDate": [
                pd.Timestamp(
                    year=int(visit_year),
                    month=int(visit_month),
                    day=1
                ).timestamp()
            ],
            "VisitQuarter": [
                f"Q{((visit_month - 1) // 3) + 1}"
            ],
            "VisitSeason": [season],
            "UserVisitCount": [
                user_visit_count
            ],
            "AttractionPopularity": [
                attraction_popularity
            ]
        })

        prediction = regression_model.predict(
            regression_preprocessor.transform(
                input_data
            )
        )

        predicted_rating = float(
            np.clip(
                prediction[0],
                1,
                5
            )
        )

        st.success(
            f"⭐ Predicted Attraction Rating: "
            f"{predicted_rating:.2f} / 5"
        )

# ============================================================
# VISIT MODE CLASSIFICATION
# ============================================================

elif page == "🎯 Visit Mode Prediction":

    st.title("🎯 Visit Mode Prediction")

    st.write(
        "Predict whether a tourism visit is likely to be Business, Couples, Family, Friends, or Solo."
    )

    col1, col2 = st.columns(2)

    with col1:

        visit_year = st.selectbox(
            "Visit Year",
            sorted(
                df["VisitYear"].dropna().unique()
            ),
            key="class_year"
        )

        visit_month = st.selectbox(
            "Visit Month",
            list(range(1, 13)),
            key="class_month"
        )

        city = st.selectbox(
            "City",
            sorted(
                df["CityName"].dropna().unique()
            ),
            key="class_city"
        )

        attraction = st.selectbox(
            "Attraction",
            sorted(
                df["Attraction"].dropna().unique()
            ),
            key="class_attraction"
        )

    with col2:

        attraction_type = st.selectbox(
            "Attraction Type",
            sorted(
                df["AttractionType"]
                .dropna()
                .unique()
            ),
            key="class_type"
        )

        address = st.selectbox(
            "Attraction Address",
            sorted(
                df["AttractionAddress"]
                .dropna()
                .unique()
            ),
            key="class_address"
        )

        season = st.selectbox(
            "Visit Season",
            sorted(
                df["VisitSeason"]
                .dropna()
                .unique()
            ),
            key="class_season"
        )

        user_visit_count = st.number_input(
            "User Visit Count",
            min_value=1,
            value=1,
            key="class_count"
        )

    selected_city = df[
        df["CityName"] == city
    ].iloc[0]

    selected_attraction = df[
        df["Attraction"] == attraction
    ].iloc[0]

    attraction_popularity = (
        selected_attraction[
            "AttractionPopularity"
        ]
    )

    if st.button(
        "🎯 Predict Visit Mode",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "VisitYear": [visit_year],
            "VisitMonth": [visit_month],
            "ContinentId": [
                selected_city["ContinentId"]
            ],
            "RegionId": [
                selected_city["RegionId"]
            ],
            "CountryId": [
                selected_city["CountryId"]
            ],
            "CityName": [city],
            "Attraction": [attraction],
            "AttractionType": [attraction_type],
            "AttractionAddress": [address],
            "VisitDate": [
                pd.Timestamp(
                    year=int(visit_year),
                    month=int(visit_month),
                    day=1
                ).timestamp()
            ],
            "VisitQuarter": [
                f"Q{((visit_month - 1) // 3) + 1}"
            ],
            "VisitSeason": [season],
            "UserVisitCount": [
                user_visit_count
            ],
            "AttractionPopularity": [
                attraction_popularity
            ]
        })

        prediction = classification_model.predict(
            classification_preprocessor.transform(
                input_data
            )
        )

        predicted_mode = prediction[0]

        st.success(
            f"🎯 Predicted Visit Mode: **{predicted_mode}**"
        )

# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🤖 Recommendations":

    st.title("🤖 Personalized Attraction Recommendations")

    st.write(
        "Select a user to generate personalized attraction recommendations using collaborative filtering."
    )

    available_users = sorted(
        train_user_item_matrix.index.tolist()
    )

    user_id = st.selectbox(
        "Select User ID",
        available_users
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    if st.button(
        "🤖 Generate Recommendations",
        use_container_width=True
    ):

        recommendations = recommend_attractions(
            user_id,
            top_n=top_n
        )

        if recommendations.empty:

            st.warning(
                "No recommendations available for this user."
            )

        else:

            recommendations["RecommendationScore"] = (
                recommendations[
                    "RecommendationScore"
                ].round(4)
            )

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"Generated {len(recommendations)} personalized recommendations."
            )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.subheader("⭐ Regression Performance")

    regression_results = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "model_comparison.csv"
        )
    )

    st.dataframe(
        regression_results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🎯 Classification Performance")

    classification_results = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "classification_model_comparison.csv"
        )
    )

    st.dataframe(
        classification_results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🤖 Recommendation Performance")

    st.dataframe(
        recommendation_evaluation,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "🔍 Feature Importance":

    st.title("🔍 Feature Importance")

    feature_importance = pd.read_csv(
        os.path.join(
            BASE_PATH,
            "feature_importance.csv"
        )
    )

    top_features = feature_importance.head(15)

    st.bar_chart(
        top_features.set_index(
            "Feature"
        )["Importance"]
    )

    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "🗃️ Dataset Explorer":

    st.title("🗃️ Dataset Explorer")

    st.write(
        f"Dataset contains {len(df):,} records and {len(df.columns)} columns."
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )
