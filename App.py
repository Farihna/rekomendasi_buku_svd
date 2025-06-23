# import streamlit as st

# from utils.data_loader import (
#     load_pickle_file,        # Untuk memuat .pkl
#     load_joblib_file,        # Untuk memuat joblib
#     load_csv_data,           # Untuk memuat .csv
#     load_recommender_model,  # Untuk memuat model CF sebagai weight
# )
# from utils.cf_utils import get_cf_recommendations
# from utils.cbf_utils import get_similar_books
# from utils.ui_components import tampilkan_rekomendasi_di_ui

# # Streamlit UI
# st.set_page_config(page_title="Rekomendasi Buku", layout="wide") 
# st.title("Aplikasi Rekomendasi Buku")

# # Aset CF
# user_encoding = load_pickle_file("assets/encoders/user_encoding.pkl")
# isbn_encoding = load_pickle_file("assets/encoders/isbn_encoding.pkl")
# df_ratings = load_csv_data("assets/data/df_ratings.csv")
# book_df = load_csv_data("assets/data/clean_books.csv")

# # Model CF (weight)
# num_users_book_count = load_pickle_file("assets/data/user_book_counts.pkl")
# num_users_book_count = load_pickle_file("assets/data/user_book_counts.pkl")
# # load jumlah user dan buku 
# num_users = num_users_book_count.get('num_user', 0)
# num_books = num_users_book_count.get('num_book', 0)
# # load model
# try:
#     cf_model = load_recommender_model(num_users, num_books, "assets/cf_model/cf.weights.h5")
# except Exception as e:
#     st.error(f"Error loading CF model: {e}")
#     cf_model = None
    
# # Aset CBF
# df_books_cbf = load_csv_data("assets/data/books_cbf.csv")
# cbf_tfidf_matrix = load_joblib_file("assets/cbf_model/cbf_tfidf_matrix.pkl")
# book_titles = load_pickle_file("assets/cbf_model/book_titles.pkl")

# if df_books_cbf is not None and not df_books_cbf.empty:
#     book_titles = df_books_cbf['Book-Title'].tolist()
#     if cbf_tfidf_matrix is not None and book_titles is not None:
#         if cbf_tfidf_matrix.shape[0] != len(book_titles):
#             st.error(f"CRITICAL ERROR: Ketidakcocokan jumlah baris matriks TF-IDF CBF ({cbf_tfidf_matrix.shape[0]}) dan daftar judul CBF ({len(book_titles)}). Fitur CBF mungkin tidak berfungsi.")
#             book_titles = None 
#     elif book_titles is None:
#          st.error("Gagal membuat book_titles karena DF_BOOKS_CBF kosong atau None.")
# else:
#     st.error("DataFrame buku untuk CBF (DF_BOOKS_CBF) gagal dimuat atau kosong.")


# # Simpan status login di session
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
# if "user_id" not in st.session_state:
#     st.session_state.user_id = None

# # user ID
# list_user_id = df_ratings['User-ID']
    
# # Jika belum login, tampilkan form login
# if not st.session_state.logged_in:
#     st.subheader("🔐 Silakan Login Terlebih Dahulu")
#     selected_user_id_cf = st.selectbox("Pilih User ID untuk Login:", options=list_user_id, index=0)

#     if st.button("🔓 Login", key="login_button"):
#         st.session_state.logged_in = True
#         st.session_state.user_id = selected_user_id_cf
#         st.rerun()

# else:
#     user_id = st.session_state.user_id

#     # Tampilkan status login dan tombol logout
#     col1, col2 = st.columns([0.9, 0.1])
#     with col1:
#         st.success(f"Login sebagai User ID: {user_id}")
#     with col2:
#         if st.button("Logout", key="logout_button"):
#             st.session_state.logged_in = False
#             st.session_state.user_id = None
#             st.rerun()

#     # Fitur pencarian CBF
#     if all([df_books_cbf is not None, cbf_tfidf_matrix is not None, book_titles]):
#         book_title_options = df_books_cbf['Book-Title'].drop_duplicates().tolist()
#         with st.container(border=True): 
#             select_title = st.selectbox("Pilih Judul Buku:", options=book_title_options, key="cbf_selectbox")

#             if st.button("Cari Buku", key="cbf_button"):
#                 if select_title:
#                     similar_df = get_similar_books(
#                         select_title,
#                         cbf_tfidf_matrix,
#                         book_titles,
#                         df_books_cbf,
#                         top_n=10,
#                     )
#                     tampilkan_rekomendasi_di_ui(similar_df, f"Buku serupa dengan '{select_title}':")
#     else:
#         st.error("Maaf terjadi kesalahan saat memuat data.")

#     if all([user_encoding, isbn_encoding, df_ratings is not None, cf_model, book_df is not None]):

#         # Tampilkan rekomendasi CF
#         rekomendasi_df = get_cf_recommendations(
#             user_id,
#             df_ratings,
#             book_df,
#             isbn_encoding,
#             user_encoding,
#             cf_model,
#             top_n=20
#         )
#         st.subheader("📌 Rekomendasi Berdasarkan Preferensi Anda")
#         tampilkan_rekomendasi_di_ui(rekomendasi_df, f"Rekomendasi untuk User: {user_id}")
#     else:
#         st.error("Maaf terjadi kesalahan saat memuat data.")

# +===========================================================++
import streamlit as st
import pandas as pd
# from surprise import dump, Reader, Dataset, SVD # surprise.dump tidak lagi diperlukan untuk memuat model
# from surprise.model_selection import train_test_split # Hanya perlu jika melatih model di sini

import joblib # Import joblib untuk memuat model

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Rekomendasi Buku Goodreads",
    page_icon="📚",
    layout="centered"
)

# --- Fungsi untuk Memuat Data (dengan cache untuk performa) ---
def load_data():
    try:
        books_df = pd.read_csv('books_df.csv')
        ratings_df = pd.read_csv('ratings_df.csv')
        
        books_df_cleaned = books_df.drop_duplicates(subset=['book_id'], keep='first')
        return books_df_cleaned, ratings_df
    except FileNotFoundError:
        st.error("Pastikan file 'books.csv' dan 'ratings.csv' ada di direktori yang sama.")
        st.stop()

# --- Fungsi untuk Memuat Model SVD (menggunakan joblib) ---
def load_model():
    # PATH MODEL ANDA
    model_path = 'svd_gs_model.pkl' # Sesuaikan dengan lokasi file Anda
    try:
        # Menggunakan joblib.load untuk memuat model
        loaded_svd_model = joblib.load(model_path)
        st.success("Model SVD berhasil dimuat!")
        return loaded_svd_model
    except FileNotFoundError:
        st.error(f"Error: Model SVD tidak ditemukan di '{model_path}'.")
        st.warning("Pastikan file model Anda ('svd_gs_model.pkl') berada di direktori yang sama dengan aplikasi Streamlit ini, atau sesuaikan path-nya.")
        st.stop() # Hentikan eksekusi jika model tidak ditemukan

# --- Fungsi Rekomendasi (tetap sama) ---
# ... (Sisipkan fungsi get_recommendations yang sudah Anda miliki di sini) ...
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


# --- Judul Aplikasi Streamlit ---
st.title("📚 Rekomendasi Buku Goodreads")
st.markdown("Temukan buku menarik berdasarkan preferensi membaca Anda!")

# --- Muat Data dan Model ---
books_df_cleaned, ratings_df = load_data()
loaded_svd_model = load_model() # Ini akan memanggil joblib.load
# --- Input Pengguna ---
col1, col2 = st.columns(2)
with col1:
    user_id = st.number_input("Masukkan User ID:", min_value=1, value=1, step=1, key="user_id_input")
    user_id_input = loaded_svd_model.trainset.to_raw_uid(user_id)
with col2:
    num_recommendations_input = st.slider("Jumlah Rekomendasi:", min_value=1, max_value=20, value=10, key="num_rec_slider")

# --- Tombol untuk Mendapatkan Rekomendasi ---
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
            cols_display = st.columns(3)
            for index, row in recommendations_df.iterrows():
                with cols_display[index % 3]:
                    st.write(f"**{row['original_title']}**")
                    st.caption(f"Penulis: {row['authors']}")
                    st.caption(f"Rating Rata-rata: {row['average_rating']:.2f}")
                    st.caption(f"Rating Diprediksi: {row['predicted_rating']:.2f}")
                    if pd.notna(row['image_url']) and row['image_url'].startswith('http'):
                        st.image(row['image_url'], width=150, caption=row['original_title'])
                    else:
                        st.image("https://via.placeholder.com/150x200?text=No+Image", width=150, caption="Gambar tidak tersedia")
                    st.markdown("---")
        else:
            st.info("Tidak ada rekomendasi yang ditemukan untuk user ID ini atau semua buku sudah dinilai.")
    else:
        st.error("Model rekomendasi belum siap. Pastikan 'svd_gs_model.pkl' ada.")

st.markdown("---")
st.info("Aplikasi ini menggunakan model SVD untuk merekomendasikan buku.")