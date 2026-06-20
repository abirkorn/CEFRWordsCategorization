import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import collections

def main():
    print("Loading data...")
    with open("raw_extracted_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Enrichment
    print("Enriching contexts...")
    contexts = []
    for item in data:
        # Primary variation:Syntactic enrichment
        ctx1 = f"The concept/word: {item['w']}, used as a {item['pos']}."
        # Secondary variation: Semantic/Story enrichment
        ctx2 = f"A story element related to: {item['w']}."
        # Combine them for the embedding
        contexts.append(f"{ctx1} {ctx2}")

    print("Generating embeddings (this may take a few minutes)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(contexts, show_progress_bar=True)

    print("Reducing dimensions with UMAP...")
    # Standard UMAP params for clustering
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=5, # Reduced to 5D for HDBSCAN
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    embeddings_reduced = reducer.fit_transform(embeddings)

    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings_reduced)

    # Map results back to data
    for i, item in enumerate(data):
        item['cluster'] = int(cluster_labels[i])

    # Cluster statistics
    cluster_counts = collections.Counter(cluster_labels)
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)

    # Output temporary results for validation
    with open("clustered_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n--- Cluster Validation Summary ---")

    # Top 5 largest clusters (excluding noise -1 if it's top)
    top_n = 5
    count = 0
    for cid, ccount in sorted_clusters:
        if cid == -1:
            continue
        if count >= top_n:
            break

        print(f"\nCluster ID: {cid}")
        print(f"Total word count: {ccount}")

        cluster_items = [d for d in data if d['cluster'] == cid]
        sample = cluster_items[:10]
        sample_str = ", ".join([f"{d['w']} ({d['pos']})" for d in sample])
        print(f"Sample words: {sample_str}")
        count += 1

    # Noise sample
    noise_count = cluster_counts.get(-1, 0)
    print(f"\nCluster ID: -1 (Noise/Universal)")
    print(f"Total word count: {noise_count}")
    noise_items = [d for d in data if d['cluster'] == -1]
    noise_sample = noise_items[:5]
    noise_sample_str = ", ".join([f"{d['w']} ({d['pos']})" for d in noise_sample])
    print(f"Sample words: {noise_sample_str}")

if __name__ == "__main__":
    main()
