import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. Load Data and Models (with caching) ---
@st.cache_data
def load_data():
    # Change the path to look for the CSV in the same directory as app.py
    df_loaded = pd.read_csv('google_books_dataset.csv')
    # Ensure relevant columns are numeric, fillna for safety before prediction
    df_loaded['description'] = df_loaded['description'].fillna('') # Pre-process description
    df_loaded['page_count'] = pd.to_numeric(df_loaded['page_count'], errors='coerce').fillna(0) # Fill NaN with 0 or a sensible default
    df_loaded['list_price'] = pd.to_numeric(df_loaded['list_price'], errors='coerce').fillna(df_loaded['list_price'].median()) # Fill with median
    return df_loaded

@st.cache_resource
def load_tfidf_vectorizer():
    return joblib.load('tfidf_vectorizer.pkl')

@st.cache_resource
def load_cosine_sim():
    return joblib.load('recommendation_model.pkl')

@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_embeddings():
    return joblib.load('semantic_embeddings.pkl')

@st.cache_resource
def load_overdue_model():
    return joblib.load('overdue_model.pkl')

@st.cache_resource
def load_popularity_predictor_model():
    return joblib.load('popularity_predictor_model.pkl')

df = load_data()
tfidf = load_tfidf_vectorizer()
cosine_sim = load_cosine_sim()
st_model = load_sentence_transformer_model()
embeddings = load_embeddings()
model_rf = load_overdue_model()
model_pop_predictor = load_popularity_predictor_model() # Load the new model

# --- 2. Helper Functions ---

# Recommendation Function (adapted for Streamlit)
def recommend_streamlit(book_title, df_rec, cosine_sim_rec):
    if book_title not in df_rec['title'].values:
        return ["Book title not found for recommendation."]
    idx = df_rec[df_rec['title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim_rec[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6] # Top 5 similar books
    book_indices = [i[0] for i in sim_scores]
    recommended_titles = [df_rec['title'].iloc[i] for i in book_indices]
    return recommended_titles

# Semantic Search Function (adapted for Streamlit)
def search_books_streamlit(query_text, embeddings_search, df_search, st_model_search):
    q_embedding = st_model_search.encode([query_text])
    scores = cosine_similarity(q_embedding, embeddings_search)[0]
    idx = np.argsort(scores)[::-1][:5] # Top 5 most similar
    return df_search.iloc[idx][['title', 'authors', 'description']] # Include description for context

# Chatbot Function (enhanced for Streamlit context)
def chatbot(question):
    question = question.lower()
    if "recommend" in question:
        return "Please go to the 'Recommend Books' section in the sidebar to get recommendations."
    elif "search" in question:
        return "Please use the 'Search Books' section in the sidebar for semantic search."
    elif "overdue" in question:
        return "You can predict if a book is overdue in the 'Overdue Prediction' section in the sidebar."
    elif "popularity" in question or "popular" in question:
        return "Check out the 'Book Popularity Predictor' section to predict a book's popularity score."
    elif "analytics" in question:
        return "Check out the 'Reading Analytics' section for insights into popular authors and categories."
    elif "features" in question or "system" in question:
        return "The 'Library Features (Conceptual)' section outlines the core functionalities of the system."
    else:
        return "Library Assistant Ready. How can I help you with our AI-powered library services today?"

# --- 3. Streamlit UI Layout ---
st.title("AI-Powered Library Management System")

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Home",
        "Search Books",
        "Recommend Books",
        "Overdue Prediction",
        "Book Popularity Predictor", # Added new menu item
        "Chatbot",
        "Reading Analytics",
        "Library Features (Conceptual)"
    ]
)

# --- Main Content Area ---
if menu == "Home":
    st.header("Welcome to your AI-Powered Library Management System!")
    st.write("Explore our intelligent features for book discovery, management, and more.")
    st.image("https://images.unsplash.com/photo-1521587765099-cf5687794511?q=80&w=1770&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
             caption="A Modern Library", use_column_width=True)
    st.markdown("---")
    st.subheader("Key Features:")
    st.markdown("- **Semantic Search:** Find books by describing what you're looking for.")
    st.markdown("- **AI Recommendation Engine:** Discover books similar to your favorites.")
    st.markdown("- **Overdue Prediction:** Predict if a borrowed book will be overdue.")
    st.markdown("- **Book Popularity Predictor:** Estimate a book's popularity score.") # Added new feature description
    st.markdown("- **AI Chatbot:** Get quick answers and guidance.")
    st.markdown("- **Reading Analytics:** Gain insights into book popularity.")

elif menu == "Search Books":
    st.header("Semantic Book Search")
    st.write("Enter a query to find books based on their meaning and content.")
    query_input = st.text_input("Enter your search query (e.g., 'adventure stories about space', 'historical fiction in ancient Rome'):")
    if query_input:
        st.subheader("Top 5 Search Results:")
        results = search_books_streamlit(query_input, embeddings, df, st_model)
        st.dataframe(results, use_container_width=True)

elif menu == "Recommend Books":
    st.header("Content-Based Book Recommendations")
    st.write("Select a book from our catalog to get recommendations for similar titles.")
    book_title_options = df['title'].unique().tolist()
    # Filter out None/NaN if any exist before sorting
    book_title_options = [title for title in book_title_options if pd.notna(title)]
    book_title_options.sort() # Sort alphabetically for easier selection
    book_title_input = st.selectbox("Select a book:", book_title_options)

    if st.button("Get Recommendations"):
        if book_title_input:
            recommendations = recommend_streamlit(book_title_input, df, cosine_sim)
            st.subheader(f"Books similar to '{book_title_input}':")
            if "Book title not found" in recommendations[0]:
                st.warning(recommendations[0])
            else:
                for i, title in enumerate(recommendations):
                    st.write(f"{i+1}. {title}")
        else:
            st.warning("Please select a book to get recommendations.")

elif menu == "Overdue Prediction":
    st.header("Overdue Book Prediction")
    st.write("Enter the number of days a book has been borrowed to predict if it will be overdue (threshold: 30 days).")
    days_input = st.number_input("Number of days borrowed:", min_value=1, max_value=365, value=15, step=1)

    if st.button("Predict Overdue Status"):
        # The model expects a 2D array, even for a single sample
        prediction = model_rf.predict([[days_input]])[0]
        if prediction == 1:
            st.error(f"Prediction: This book (borrowed for {days_input} days) is likely to be **OVERDUE**.")
        else:
            st.success(f"Prediction: This book (borrowed for {days_input} days) is likely to be **ON TIME**.")

elif menu == "Book Popularity Predictor": # New section for popularity prediction
    st.header("Book Popularity Score Predictor")
    st.write("Predict a book's popularity score based on its features. The score is a weighted combination of average rating and normalized ratings count.")

    # Input widgets for features
    page_count_input = st.number_input("Page Count:", min_value=1, value=300, step=10)
    description_length_input = st.number_input("Description Length (number of characters):", min_value=0, value=500, step=50)
    list_price_input = st.number_input("List Price:", min_value=0.0, value=19.00, step=0.01)

    if st.button("Predict Popularity Score"):
        # Create a DataFrame for prediction (model expects features in a specific order/format)
        features_for_prediction = pd.DataFrame([[page_count_input, description_length_input, list_price_input]],
                                               columns=['page_count', 'description_length', 'list_price'])

        # Make prediction
        predicted_popularity = model_pop_predictor.predict(features_for_prediction)[0]
        st.subheader(f"Predicted Popularity Score: {predicted_popularity:.4f}")
        st.info("Note: A higher score indicates higher predicted popularity. The R-squared for this model is low, so consider this an experimental prediction.")

elif menu == "Chatbot":
    st.header("Library Assistant Chatbot")
    st.write("Ask me anything about the library system's features!")
    user_question = st.text_input("Type your question here:")
    if user_question:
        response = chatbot(user_question)
        st.info(f"Chatbot: {response}")

elif menu == "Reading Analytics":
    st.header("Reading Analytics: Insights into Book Data")
    st.write("Explore popular authors and categories based on our dataset.")

    st.subheader("Top 10 Authors by Number of Books")
    top_authors = df['authors'].value_counts().head(10)
    st.bar_chart(top_authors)

    st.subheader("Top 10 Book Categories")
    top_categories = df['categories'].value_counts().head(10)
    st.bar_chart(top_categories)

elif menu == "Library Features (Conceptual)":
    st.header("Core Library Functionalities (Conceptual)")
    st.write("This section outlines other essential library operations that are part of the system design but not fully interactive in this demo.")

    st.subheader("User Management")
    st.markdown("- **User Registration & Login:** Secure access for patrons.")

    st.subheader("Book Operations")
    st.markdown("- **Book Borrowing:** Track books checked out by users.")
    st.markdown("- **Book Return:** Process returned books and update availability.")
    st.markdown("- **Reservation System:** Allow users to reserve books that are currently borrowed.")

    st.subheader("Financials")
    st.markdown("- **Fine Calculation:** Automatically calculate fines for overdue books.")

    st.subheader("Engagement")
    st.markdown("- **Reviews & Ratings:** Enable users to leave reviews and rate books.")
