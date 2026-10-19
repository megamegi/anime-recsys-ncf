import os
import html
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device yang digunakan: {DEVICE}")

def load_and_preprocess_data(anime_path='anime.csv', rating_path='rating.csv',
                             num_users=500, num_animes=600):
    print("\n[1/4] Memuat dan membersihkan dataset...")
    
    df_anime = pd.read_csv(anime_path)
    df_anime['name'] = df_anime['name'].fillna('Unknown').apply(html.unescape)
    df_anime['genre'] = df_anime['genre'].fillna('Unknown')
    df_anime['type'] = df_anime['type'].fillna('Unknown')
    df_anime['rating'] = pd.to_numeric(df_anime['rating'], errors='coerce').fillna(0.0)
    
    anime_meta = df_anime.set_index('anime_id')[['name', 'genre', 'type', 'rating']].to_dict('index')

    df_rating = pd.read_csv(rating_path)
    
    df_rating = df_rating[df_rating['rating'] > 0]
    
    u_counts = df_rating['user_id'].value_counts()
    a_counts = df_rating['anime_id'].value_counts()
    
    top_users = u_counts.index[:num_users]
    top_animes = a_counts.index[:num_animes]
    
    df_clean = df_rating[df_rating['user_id'].isin(top_users) & df_rating['anime_id'].isin(top_animes)].copy()
    
    unique_users = sorted(df_clean['user_id'].unique())
    unique_animes = sorted(df_clean['anime_id'].unique())
    
    user2idx = {u: i for i, u in enumerate(unique_users)}
    idx2user = {i: u for u, i in user2idx.items()}
    
    anime2idx = {a: i for i, a in enumerate(unique_animes)}
    idx2anime = {i: a for a, i in anime2idx.items()}
    
    df_clean['user_idx'] = df_clean['user_id'].map(user2idx)
    df_clean['anime_idx'] = df_clean['anime_id'].map(anime2idx)
    
    print(f"-> Total Interaksi Rating Bersih : {len(df_clean):,} baris")
    print(f"-> Total Active Users Terpilih   : {len(unique_users):,} user")
    print(f"-> Total Popular Anime Terpilih  : {len(unique_animes):,} anime")
    print(f"-> Rentang Nilai Rating Asli     : Min {df_clean['rating'].min()} s/d Max {df_clean['rating'].max()}")
    
    return df_clean, anime_meta, user2idx, idx2user, anime2idx, idx2anime

class AnimeRatingDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class NeuralCollaborativeFiltering(nn.Module):
    """
    Model-based Collaborative Filtering menggunakan Deep Learning NCF (MLP branch).
    Berbeda dengan Matrix Factorization yang hanya melakukan Dot Product,
    NCF menggabungkan vector embedding User dan Item kemudian memprosesnya
    melalui beberapa lapisan Non-Linear Dense Network (Multi-Layer Perceptron).
    """
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
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
        
        prediction = self.mlp(concat_features)
        return prediction

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for u, i, r in dataloader:
        u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
        optimizer.zero_grad()
        preds = model(u, i)
        loss = criterion(preds, r)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(r)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for u, i, r in dataloader:
            u, i = u.to(DEVICE), i.to(DEVICE)
            preds = model(u, i)
            all_preds.extend(preds.cpu().squeeze().numpy())
            all_targets.extend(r.squeeze().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    
    return rmse, mae, mse

def run_cross_validation(df, num_users, num_items, n_splits=5, epochs=5, batch_size=512, lr=0.003):
    print(f"\n[2/4] Menjalankan {n_splits}-Fold Cross Validation...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_results = []
    
    user_arr = df['user_idx'].values
    item_arr = df['anime_idx'].values
    rating_arr = df['rating'].values
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n--- Fold {fold + 1} / {n_splits} ---")
        
        train_ds = AnimeRatingDataset(user_arr[train_idx], item_arr[train_idx], rating_arr[train_idx])
        val_ds = AnimeRatingDataset(user_arr[val_idx], item_arr[val_idx], rating_arr[val_idx])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        model = NeuralCollaborativeFiltering(num_users, num_items).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion)
            rmse_val, mae_val, mse_val = evaluate_model(model, val_loader)
            print(f"  Epoch {epoch}/{epochs} | Train Loss: {loss:.4f} | Val RMSE: {rmse_val:.4f} | Val MAE: {mae_val:.4f} | Val MSE: {mse_val:.4f}")
            
        fold_results.append({
            'Fold': f'Fold {fold + 1}',
            'RMSE': rmse_val,
            'MAE': mae_val,
            'MSE': mse_val
        })
        
    elapsed = time.time() - start_time
    print(f"\nCross Validation selesai dalam {elapsed:.2f} detik.")
    
    df_results = pd.DataFrame(fold_results)
    mean_row = pd.DataFrame([{
        'Fold': 'Rata-Rata (Mean)',
        'RMSE': df_results['RMSE'].mean(),
        'MAE': df_results['MAE'].mean(),
        'MSE': df_results['MSE'].mean()
    }])
    std_row = pd.DataFrame([{
        'Fold': 'Standar Deviasi (Std)',
        'RMSE': df_results['RMSE'].std(),
        'MAE': df_results['MAE'].std(),
        'MSE': df_results['MSE'].std()
    }])
    
    summary_df = pd.concat([df_results, mean_row, std_row], ignore_index=True)
    print("\n" + "="*50)
    print("HASIL EVALUASI 5-FOLD CROSS VALIDATION")
    print("="*50)
    print(summary_df.to_string(index=False))
    print("="*50)
    
    summary_df.to_csv('cv_evaluation_results.csv', index=False)
    print("Hasil evaluasi tersimpan di: cv_evaluation_results.csv")
    
    return summary_df

def train_and_save_final_model(df, anime_meta, user2idx, idx2user, anime2idx, idx2anime, cv_results,
                               epochs=6, batch_size=512, lr=0.003):
    print("\n[3/4] Melatih Model Final pada Seluruh Data Bersih...")
    
    num_users = len(user2idx)
    num_items = len(anime2idx)
    
    full_ds = AnimeRatingDataset(df['user_idx'].values, df['anime_idx'].values, df['rating'].values)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
    
    final_model = NeuralCollaborativeFiltering(num_users, num_items).to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(final_model, full_loader, optimizer, criterion)
        print(f"  Final Training Epoch {epoch}/{epochs} | Loss: {loss:.4f}")
        
    print("\n[4/4] Menyimpan Model dan Data Pendukung untuk Streamlit UI...")
    
    torch.save(final_model.state_dict(), 'anime_ncf_model.pth')
    print("-> Model disimpan di: anime_ncf_model.pth")
    

    artifacts = {
        'num_users': num_users,
        'num_items': num_items,
        'user2idx': user2idx,
        'idx2user': idx2user,
        'anime2idx': anime2idx,
        'idx2anime': idx2anime,
        'anime_meta': anime_meta,
        'df_ratings': df[['user_id', 'anime_id', 'rating', 'user_idx', 'anime_idx']],
        'cv_results': cv_results
    }
    
    with open('recsys_data.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("-> Data pendukung disimpan di: recsys_data.pkl")
    print("\nSELURUH PROSES SELESAI DENGAN SUKSES! Siap untuk menjalankan Streamlit UI.")

if __name__ == '__main__':
    df_clean, anime_meta, user2idx, idx2user, anime2idx, idx2anime = load_and_preprocess_data(
        num_users=500, num_animes=600
    )
    
    cv_summary = run_cross_validation(
        df_clean,
        num_users=len(user2idx),
        num_items=len(anime2idx),
        n_splits=5,
        epochs=5
    )
    
    train_and_save_final_model(
        df_clean, anime_meta, user2idx, idx2user, anime2idx, idx2anime, cv_summary,
        epochs=6
    )
