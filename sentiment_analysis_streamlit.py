
import streamlit as st
import pickle

model = pickle.load(open(r'C:\Users\Sabinaya\PROJECTS\genre_classifier.pkl', 'rb'))
vectorizer = pickle.load(open(r'C:\Users\Sabinaya\PROJECTS\genre_movie_vectorizer.pkl', 'rb'))
mlb = pickle.load(open(r'C:\Users\Sabinaya\PROJECTS\genre_label_binarizer.pkl', 'rb'))

st.title("🎬 Movie Genre Prediction")
overview = st.text_area("Enter the movie overview")

if st.button("Predict Genre"):
    if overview.strip():
        x = vectorizer.transform([overview])
        prediction = model.predict(x)
        predicted_genres = mlb.inverse_transform(prediction)[0]
        
        if predicted_genres:
            st.success("Predicted Genres: " + ", ".join(predicted_genres))
        else:
            st.warning("No genre could be predicted.")
    else:
        st.error("Please enter a movie overview.")
