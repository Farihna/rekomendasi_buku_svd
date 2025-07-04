import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Rekomendasi Buku Goodreads",
    page_icon="📚",
    layout="wide"
)

def load_data():
    try:
        books_df = pd.read_csv('data/books_df.csv')
        ratings_df = pd.read_csv('data/ratings_df.csv')
        
        books_df_cleaned = books_df.drop_duplicates(subset=['book_id'], keep='first')
        return books_df_cleaned, ratings_df
    except FileNotFoundError:
        print("Pastikan file 'books.csv' dan 'ratings.csv' ada di direktori yang sama.")
        st.stop()

def load_model():
    model_path = 'model/svd_gs_model.pkl' 
    try:
        loaded_svd_model = joblib.load(model_path)
        return loaded_svd_model
    except FileNotFoundError:
        print(f"Error: Model SVD tidak ditemukan di '{model_path}'.")
        st.warning("Pastikan file model Anda ('svd_gs_model.pkl') berada di direktori yang sama dengan aplikasi Streamlit ini, atau sesuaikan path-nya.")
        st.stop() 

def get_recommendations(user_id, model, books_df, ratings_df, num_recommendations=10):
    user_rated_books = ratings_df[ratings_df['user_id'] == user_id]['book_id'].tolist()
    all_book_ids = books_df['book_id'].unique()
    books_to_predict = [book_id for book_id in all_book_ids if book_id not in user_rated_books]

    predictions = []
    predict_limit = 2000 
    for book_id in books_to_predict[:predict_limit]: 
        predicted_rating = model.predict(user_id, book_id).est
        predictions.append((book_id, predicted_rating))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = predictions[:num_recommendations * 2] 

    recommended_book_ids = [book_id for book_id, _ in top_recommendations]
    recommended_books_info = books_df[books_df['book_id'].isin(recommended_book_ids)].copy()

    predicted_ratings_map = {book_id: rating for book_id, rating in top_recommendations}
    recommended_books_info['predicted_rating'] = recommended_books_info['book_id'].map(predicted_ratings_map)

    recommended_books_info = recommended_books_info.sort_values(by='predicted_rating', ascending=False)
    recommended_books_info_dedup = recommended_books_info.drop_duplicates(subset=['original_title', 'authors'])
    
    return recommended_books_info_dedup.head(num_recommendations)


st.title("📚 Rekomendasi Buku Goodreads")
st.markdown("Temukan buku menarik berdasarkan preferensi membaca Anda!")

# --- Muat Data dan Model ---
books_df_cleaned, ratings_df = load_data()
loaded_svd_model = load_model() 
# --- Input Pengguna ---
col1, col2 = st.columns(2)
with col1:
    user_id = st.number_input("Masukkan User ID:", min_value=1, value=1, step=1, key="user_id_input")
    try:
        user_id_input = loaded_svd_model.trainset.to_raw_uid(user_id)
    except ValueError:
        st.error("User ID tidak Tersedia")
with col2:
    num_recommendations_input = st.slider("Jumlah Rekomendasi:", min_value=1, max_value=20, value=10, key="num_rec_slider")

if st.button("Dapatkan Rekomendasi", key="get_recommendations_button"):
    if loaded_svd_model:
        with st.spinner("Mencari rekomendasi untuk Anda..."):
            recommendations_df = get_recommendations(
                user_id_input, 
                loaded_svd_model, 
                books_df_cleaned, 
                ratings_df, 
                num_recommendations_input
            )
        
        if not recommendations_df.empty:
            st.subheader(f"Top {len(recommendations_df)} Rekomendasi untuk Pengguna {user_id_input}:")
            num_cols = 4
            rows = [recommendations_df.iloc[i:i+num_cols] for i in range(0, len(recommendations_df), num_cols)]

            for row_chunk in rows:
                cols = st.columns(num_cols)
                for col, (_, row) in zip(cols, row_chunk.iterrows()):
                    with col:
                        if pd.notna(row['image_url']) and row['image_url'].startswith('http'):
                            st.image(row['image_url'], width=300)
                        else:
                            st.image("https://via.placeholder.com/150x200?text=No+Image", width=150)
                        st.markdown(f"**{row['original_title']}**")
                        st.caption(f"Penulis: {row['authors']}")
                        st.caption(f"Rating Rata-rata: {row['average_rating']:.2f}")
                        st.caption(f"Rating Diprediksi: {row['predicted_rating']:.2f}")

        else:
            st.info("Tidak ada rekomendasi yang ditemukan untuk user ID ini atau semua buku sudah dinilai.")
    else:
        print("Model rekomendasi belum siap. Pastikan 'svd_gs_model.pkl' ada.")

st.markdown("---")
st.info("Aplikasi ini menggunakan model SVD untuk merekomendasikan buku.")
model_rmse = 0.901337  
precision = 0.907026   
recall = 0.920575
f1_score = 0.910675

st.markdown("---")
with st.expander("Performa Model"):
    st.write(f"**Root Mean Squared Error (RMSE):** {model_rmse:.3f}")
    st.write(f"**Precision:** {precision:.3f}")
    st.write(f"**Recall:** {recall:.3f}")
    st.write(f"**F1 Score:** {f1_score:.3f}")