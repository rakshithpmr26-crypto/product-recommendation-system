# ======================================================
# IMPORTS
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ======================================================
# LOAD DATASET
# ======================================================

df = pd.read_csv("final_dataset.csv")
df = df.dropna()

# ======================================================
# PRODUCT & USER MAPPINGS
# ======================================================

product_to_index = {pid: idx for idx, pid in enumerate(df['prod_id'].unique())}
index_to_product = {idx: pid for pid, idx in product_to_index.items()}

user_to_index = {uid: idx for idx, uid in enumerate(df['user_id'].unique())}

df['product_index'] = df['prod_id'].map(product_to_index)
df['user_index'] = df['user_id'].map(user_to_index)

# ======================================================
# STATISTICS
# ======================================================

product_avg_rating = df.groupby('prod_id')['ratings'].mean()
user_avg_rating = df.groupby('user_id')['ratings'].mean()

# ======================================================
# SPARSE MATRIX (IMPORTANT)
# ======================================================

sparse_matrix = csr_matrix(
    (df['ratings'], (df['product_index'], df['user_index']))
)

# ======================================================
# MODEL
# ======================================================

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(sparse_matrix)

# ======================================================
# SIMILAR PRODUCTS FUNCTION (FIXED)
# ======================================================

def get_similar_products(product_id, top_n=10):

    if product_id not in product_to_index:
        return pd.DataFrame()

    product_idx = product_to_index[product_id]

    distances, indices = model_knn.kneighbors(
        sparse_matrix[product_idx],
        n_neighbors=min(top_n + 1, sparse_matrix.shape[0])
    )

    results = []

    for i in range(1, len(indices[0])):

        idx = indices[0][i]
        prod_id = index_to_product[idx]
        score = 1 - distances[0][i]

        results.append([prod_id, score])

    return pd.DataFrame(results, columns=['prod_id', 'similarity'])

# ======================================================
# USERS WHO LIKED PRODUCT
# ======================================================

def get_top_users(product_id):

    return df[
        (df['prod_id'] == product_id) &
        (df['ratings'] >= 4)
    ]['user_id'].unique()

# ======================================================
# CANDIDATES
# ======================================================

def generate_candidates(product_id):

    users = get_top_users(product_id)

    candidate_df = df[df['user_id'].isin(users)]
    candidate_df = candidate_df[candidate_df['prod_id'] != product_id]

    return candidate_df

# ======================================================
# HYBRID RECOMMENDATION
# ======================================================

def hybrid_recommendation(product_id, top_n=10):

    candidate_df = generate_candidates(product_id)

    if candidate_df.empty:
        return pd.DataFrame()

    similarity_df = get_similar_products(product_id, top_n=20)

    results = []

    for _, row in candidate_df.iterrows():

        user_id = row['user_id']
        candidate_product = row['prod_id']

        sim_row = similarity_df[
            similarity_df['prod_id'] == candidate_product
        ]

        similarity = sim_row['similarity'].values[0] if len(sim_row) > 0 else 0

        score = (
            similarity * 0.5 +
            product_avg_rating.get(candidate_product, 0) * 0.3 +
            user_avg_rating.get(user_id, 0) * 0.2
        )

        results.append([candidate_product, score])

    rec_df = pd.DataFrame(results, columns=['product', 'score'])

    rec_df = rec_df.groupby('product')['score'].mean().reset_index()

    return rec_df.sort_values(by='score', ascending=False).head(top_n)

# ======================================================
# STREAMLIT UI
# ======================================================

st.title("🛒 Hybrid Product Recommendation System")

product_ids = list(product_to_index.keys())
selected_product = st.selectbox("Select Product ID", product_ids)

top_n = st.slider("Number of recommendations", 5, 20, 10)

# ======================================================
# BUTTON
# ======================================================

if st.button("🚀 Get Recommendations"):

    recs = hybrid_recommendation(selected_product, top_n)

    if recs.empty:
        st.error("No recommendations found")
    else:
        st.success("Top Recommendations")
        st.dataframe(recs, use_container_width=True)