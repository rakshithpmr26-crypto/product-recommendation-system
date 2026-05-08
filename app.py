# ======================================================
# IMPORTS
# ======================================================

import streamlit as st
import pandas as pd
import pickle

# ======================================================
# LOAD MODELS ONLY (NO CSV)
# ======================================================

knn_model = pickle.load(open("knn_model.pkl", "rb"))
sparse_matrix = pickle.load(open("sparse_matrix.pkl", "rb"))

product_to_index = pickle.load(open("product_to_index.pkl", "rb"))
index_to_product = pickle.load(open("index_to_product.pkl", "rb"))

# ======================================================
# STREAMLIT UI
# ======================================================

st.title("🛒 Hybrid Product Recommendation System")

product_ids = list(product_to_index.keys())
selected_product = st.selectbox("Select Product ID", product_ids)

top_n = st.slider("Number of recommendations", 5, 20, 10)

# ======================================================
# RECOMMENDATION FUNCTION
# ======================================================

def recommend(product_id, top_n=10):

    if product_id not in product_to_index:
        return pd.DataFrame()

    product_idx = product_to_index[product_id]

    distances, indices = knn_model.kneighbors(
        sparse_matrix[product_idx],
        n_neighbors=top_n + 1
    )

    results = []

    for i in range(1, len(indices[0])):

        idx = indices[0][i]
        prod_id = index_to_product.get(idx)

        if prod_id is None:
            continue

        similarity = 1 - distances[0][i]

        results.append([prod_id, similarity])

    return pd.DataFrame(results, columns=["Product ID", "Similarity Score"])

# ======================================================
# BUTTON ACTION
# ======================================================

if st.button("🚀 Get Recommendations"):

    recs = recommend(selected_product, top_n)

    if recs.empty:
        st.error("No recommendations found")
    else:
        st.success("Top Recommendations")
        st.dataframe(recs, use_container_width=True)