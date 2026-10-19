import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

st.set_page_config(
    page_title="Anime Recommendation System by Mega Bunawi",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0B0F19;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #2E1065 40%, #0F172A 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #94A3B8;
        line-height: 1.5;
        margin: 0;
    }
    
    .stat-card {
        background: #151D2E;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stat-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .stat-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #F1F5F9;
    }
    .stat-highlight {
        color: #38BDF8;
    }
    
    .anime-card-box {
        background: linear-gradient(145deg, #131B2E, #0F1626);
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
    }
    .anime-card-box:hover {
        transform: translateY(-3px);
        border-color: #6366F1;
        box-shadow: 0 10px 20px -3px rgba(99, 102, 241, 0.25);
    }
    .card-top-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 10px;
    }
    .rank-pill {
        background: #4F46E5;
        color: #FFFFFF;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        display: inline-block;
    }
    .anime-title {
        color: #F8FAFC;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.35;
    }
    .score-badge {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #FBBF24;
        font-weight: 800;
        font-size: 1.05rem;
        padding: 5px 12px;
        border-radius: 8px;
        white-space: nowrap;
    }
    
    .bar-container {
        background: #1E293B;
        border-radius: 999px;
        height: 6px;
        width: 100%;
        margin: 12px 0 14px 0;
        overflow: hidden;
    }
    .bar-fill {
        background: linear-gradient(90deg, #F59E0B, #10B981);
        height: 100%;
        border-radius: 999px;
    }
    
    .pill-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-type {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-global {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .badge-genre {
        background: #1E293B;
        color: #94A3B8;
        border: 1px solid #334155;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0E1422;
        border-right: 1px solid #1E293B;
    }
    
    .academic-box {
        background: #111827;
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        color: #CBD5E1;
        font-size: 0.92rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, user_indices, item_indices):
        u_emb = self.user_embedding(user_indices)
        i_emb = self.item_embedding(item_indices)
        concat_features = torch.cat([u_emb, i_emb], dim=-1)
        return self.mlp(concat_features)

@st.cache_resource
def load_resources():
    model_path = 'anime_ncf_model.pth'
    data_path = 'recsys_data.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None, None
        
    with open(data_path, 'rb') as f:
        artifacts = pickle.load(f)
        
    model = NeuralCollaborativeFiltering(
        artifacts['num_users'],
        artifacts['num_items'],
        embedding_dim=32
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, artifacts

model, artifacts = load_resources()

if model is None or artifacts is None:
    st.error("File model atau dataset belum siap. Silakan jalankan `python train_eval.py` di terminal terlebih dahulu.")
    st.stop()

user2idx = artifacts['user2idx']
idx2user = artifacts['idx2user']
anime2idx = artifacts['anime2idx']
idx2anime = artifacts['idx2anime']
anime_meta = artifacts['anime_meta']
df_ratings = artifacts['df_ratings']
cv_results = artifacts['cv_results']

active_users_list = sorted(list(user2idx.keys()))

mean_rmse = cv_results.loc[cv_results['Fold'] == 'Rata-Rata (Mean)', 'RMSE'].values[0]
mean_mae = cv_results.loc[cv_results['Fold'] == 'Rata-Rata (Mean)', 'MAE'].values[0]
mean_mse = cv_results.loc[cv_results['Fold'] == 'Rata-Rata (Mean)', 'MSE'].values[0]

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown("---")
    st.markdown("**Pengguna Aktif:**")
    input_mode = st.radio(
        "Pilihan Input ID Pengguna:",
        ["Daftar Pengguna Aktif", "Input Manual User ID"],
        label_visibility="collapsed"
    )
    
    if input_mode == "Daftar Pengguna Aktif":
        selected_user = st.selectbox(
            "Pilih User ID Aktif:",
            active_users_list,
            index=0
        )
    else:
        manual_id = st.number_input(
            "Masukkan User ID:",
            min_value=1,
            max_value=int(max(active_users_list)),
            value=int(active_users_list[0]),
            step=1
        )
        if manual_id in user2idx:
            selected_user = manual_id
            st.caption(f"Status: User ID {selected_user} terdaftar.")
        else:
            selected_user = active_users_list[0]
            st.warning(f"User ID {manual_id} tidak terdaftar pada matriks. Dialihkan ke ID: {selected_user}")
            
    st.markdown("---")
    st.markdown("**Output Rekomendasi:**")
    top_k = st.slider("Jumlah Rekomendasi (Top-K):", min_value=2, max_value=20, value=6, step=2)
    
    all_genres = set()
    for meta in anime_meta.values():
        for g in str(meta['genre']).split(','):
            g_clean = g.strip()
            if g_clean and g_clean != 'Unknown':
                all_genres.add(g_clean)
                
    genre_filter = st.selectbox(
        "Filter Kategori Genre:",
        ["Semua Genre"] + sorted(list(all_genres))
    )
    
    st.markdown("---")
    st.markdown("**Sistem:**")
    st.markdown(f"""
    - **Metode:** Neural CF (MLP)
    - **Jumlah User:** {len(user2idx):,}
    - **Jumlah Item:** {len(anime2idx):,}
    - **Dimensi Embedding:** 32
    """)

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">
        <span>🎬 Anime Recommendation System by Mega Bunawi</span>
    </div>
    <div class="hero-subtitle">
        Implementasi <b>Model-Based Collaborative Filtering</b> menggunakan arsitektur <b>Neural Collaborative Filtering (NCF - Multi-Layer Perceptron)</b> dengan pendekatan non-linear embedding interaction.
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Total Data Interaksi</div>
        <div class="stat-value stat-highlight">{len(df_ratings):,}</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Pengguna Terdaftar</div>
        <div class="stat-value">{len(user2idx):,}</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Katalog Anime</div>
        <div class="stat-value">{len(anime2idx):,}</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Rata-Rata RMSE (CV)</div>
        <div class="stat-value" style="color: #10B981;">{mean_rmse:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Rata-Rata MAE (CV)</div>
        <div class="stat-value" style="color: #38BDF8;">{mean_mae:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab_rec, tab_profile, tab_evaluation = st.tabs([
    "🎯 Hasil Rekomendasi (Top-N)",
    "👤 Profil & Riwayat Pengguna",
    "📈 Evaluasi Model (Cross-Validation)"
])

user_history = df_ratings[df_ratings['user_id'] == selected_user].copy()
watched_anime_ids = set(user_history['anime_id'].values)

with tab_rec:
    st.markdown(f"#### Daftar Rekomendasi Personal untuk Pengguna ID: `{selected_user}`")
    st.markdown("Rekomendasi dihitung berdasarkan estimasi nilai preferensi tertinggi pada item yang **belum pernah berinteraksi** sebelumnya:")
    
    candidate_anime_ids = [a_id for a_id in anime2idx.keys() if a_id not in watched_anime_ids]
    
    if genre_filter != "Semua Genre":
        candidate_anime_ids = [
            a_id for a_id in candidate_anime_ids
            if genre_filter in str(anime_meta.get(a_id, {}).get('genre', ''))
        ]
        
    if not candidate_anime_ids:
        st.info("Tidak ditemukan item rekomendasi untuk kriteria filter genre yang dipilih.")
    else:
        u_idx = user2idx[selected_user]
        u_tensor = torch.tensor([u_idx] * len(candidate_anime_ids), dtype=torch.long)
        i_tensor = torch.tensor([anime2idx[a_id] for a_id in candidate_anime_ids], dtype=torch.long)
        
        with torch.no_grad():
            preds = model(u_tensor, i_tensor).squeeze().cpu().numpy()
            
        rec_df = pd.DataFrame({
            'anime_id': candidate_anime_ids,
            'predicted_score': np.clip(preds, 1.0, 10.0)
        })
        rec_df = rec_df.sort_values(by='predicted_score', ascending=False).head(top_k)
        
        grid_cols = st.columns(2)
        for rank, (_, row) in enumerate(rec_df.iterrows(), start=1):
            col = grid_cols[(rank - 1) % 2]
            a_id = int(row['anime_id'])
            meta = anime_meta.get(a_id, {'name': 'Unknown', 'genre': 'Unknown', 'type': 'Unknown', 'rating': 0.0})
            score = row['predicted_score']
            pct = min(max((score / 10.0) * 100, 10), 100)
            
            genres_list = [g.strip() for g in str(meta['genre']).split(',') if g.strip()]
            genre_badges_html = "".join([f'<span class="pill-badge badge-genre">{g}</span>' for g in genres_list[:4]])
            
            with col:
                st.markdown(f"""<div class="anime-card-box">
<div class="card-top-row">
<div>
<span class="rank-pill">Peringkat #{rank}</span>
<div style="height: 6px;"></div>
<h3 class="anime-title">{meta['name']}</h3>
</div>
<div class="score-badge">⭐ {score:.2f} <span style="font-size: 0.75rem; color: #94A3B8;">/ 10</span></div>
</div>
<div class="bar-container"><div class="bar-fill" style="width: {pct:.1f}%;"></div></div>
<div style="margin-bottom: 8px;">
<span class="pill-badge badge-type">Format: {meta['type']}</span>
<span class="pill-badge badge-global">Benchmark Global: {meta['rating']:.2f}</span>
</div>
<div>{genre_badges_html}</div>
</div>""", unsafe_allow_html=True)

with tab_profile:
    st.markdown(f"#### Profil Preferensi Historis Pengguna ID: `{selected_user}`")
    st.markdown("Informasi riwayat interaksi pengguna yang digunakan sebagai basis representasi preferensi laten:")
    
    if len(user_history) == 0:
        st.info("Pengguna ini belum memiliki riwayat rating pada basis data.")
    else:
        avg_user_rating = user_history['rating'].mean()
        min_user_rating = user_history['rating'].min()
        max_user_rating = user_history['rating'].max()
        
        user_history['title'] = user_history['anime_id'].apply(lambda x: anime_meta.get(x, {}).get('name', 'Unknown'))
        user_history['genre'] = user_history['anime_id'].apply(lambda x: anime_meta.get(x, {}).get('genre', 'Unknown'))
        user_history['type'] = user_history['anime_id'].apply(lambda x: anime_meta.get(x, {}).get('type', 'Unknown'))
        
        up1, up2, up3 = st.columns(3)
        with up1:
            st.metric("Total Judul Ditonton", f"{len(user_history)} judul")
        with up2:
            st.metric("Rata-Rata Rating Diberikan", f"{avg_user_rating:.2f} / 10")
        with up3:
            st.metric("Rentang Rating", f"{min_user_rating:.0f} s/d {max_user_rating:.0f}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Daftar Rating Tertinggi Pengguna (Top Historical Ratings):**")
        
        sorted_history = user_history.sort_values(by='rating', ascending=False)
        display_history = sorted_history[['anime_id', 'title', 'genre', 'type', 'rating']].rename(columns={
            'anime_id': 'ID Anime',
            'title': 'Judul Anime',
            'genre': 'Kategori Genre',
            'type': 'Format Tayangan',
            'rating': 'Rating Diberikan (Skala 1-10)'
        })
        
        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )

with tab_evaluation:
    st.markdown("#### Hasil Pengujian Model: 5-Fold Cross Validation")
    
    st.markdown("""
    <div class="academic-box">
        <b>Prosedur Pengujian:</b><br>
        Evaluasi dilakukan menggunakan metode <b>5-Fold Cross Validation</b> secara acak terdistribusi (random seed = 42). 
        Seluruh data interaksi dibagi menjadi 5 subset pengujian, di mana masing-masing subset berperan sebagai <i>validation set</i> independen 
        secara bergantian untuk menghindari bias pemisahan data (data splitting bias).
    </div>
    """, unsafe_allow_html=True)
    
    em1, em2, em3 = st.columns(3)
    with em1:
        st.markdown(f"""
        <div class="stat-card" style="border-top: 3px solid #10B981;">
            <div class="stat-label">Root Mean Squared Error (RMSE)</div>
            <div class="stat-value" style="color: #10B981;">{mean_rmse:.4f}</div>
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Rata-rata 5 Fold</div>
        </div>
        """, unsafe_allow_html=True)
    with em2:
        st.markdown(f"""
        <div class="stat-card" style="border-top: 3px solid #38BDF8;">
            <div class="stat-label">Mean Absolute Error (MAE)</div>
            <div class="stat-value" style="color: #38BDF8;">{mean_mae:.4f}</div>
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Rata-rata 5 Fold</div>
        </div>
        """, unsafe_allow_html=True)
    with em3:
        st.markdown(f"""
        <div class="stat-card" style="border-top: 3px solid #F59E0B;">
            <div class="stat-label">Mean Squared Error (MSE)</div>
            <div class="stat-value" style="color: #F59E0B;">{mean_mse:.4f}</div>
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Rata-rata 5 Fold</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Tabel Rincian Hasil Pengujian per Fold:**")
    
    st.dataframe(
        cv_results.style.format({
            'RMSE': '{:.4f}',
            'MAE': '{:.4f}',
            'MSE': '{:.4f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Perbandingan Metrik Evaluasi Antar-Fold:**")
    
    folds_chart_data = cv_results[~cv_results['Fold'].str.contains('Rata-Rata|Standar')].copy()
    folds_chart_data = folds_chart_data.set_index('Fold')
    st.bar_chart(folds_chart_data[['RMSE', 'MAE', 'MSE']], height=320)


