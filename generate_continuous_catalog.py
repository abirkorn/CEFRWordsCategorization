import pandas as pd
import json
import numpy as np
import os
from nltk.stem import WordNetLemmatizer
import nltk
from sentence_transformers import SentenceTransformer, util
import torch

# Setup NLTK
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

def get_lemmatizer_pos(oxford_pos):
    if oxford_pos == 'n.': return 'n'
    if oxford_pos in ['v.', 'modal v.']: return 'v'
    if oxford_pos == 'adj.': return 'a'
    if oxford_pos == 'adv.': return 'r'
    return 'n'

def normalize_pos_oxford_to_cefrj(pos):
    mapping = {
        'n.': 'noun',
        'v.': 'verb',
        'adj.': 'adjective',
        'adv.': 'adverb',
        'pron.': 'pronoun',
        'prep.': 'preposition',
        'det.': 'determiner',
        'conj.': 'conjunction',
        'modal v.': 'modal auxiliary',
        'exclam.': 'interjection',
        'number': 'number',
        'part.': 'adverb' # fallback
    }
    return mapping.get(pos, 'noun')

def main():
    print("Loading datasets...")
    with open('raw_extracted_data.json', 'r') as f:
        oxford_data = json.load(f)
    df_oxford = pd.DataFrame(oxford_data)
    df_oxford.rename(columns={'level': 'orig_cefr'}, inplace=True)

    df_cefrj = pd.read_csv('cefrj-vocabulary-profile-1.5.csv')
    # Normalize CEFR-J POS to simplify matching
    cefrj_pos_norm = {
        'be-verb': 'verb',
        'do-verb': 'verb',
        'have-verb': 'verb',
        'adjective': 'adjective',
        'adverb': 'adverb',
        'conjunction': 'conjunction',
        'determiner': 'determiner',
        'interjection': 'interjection',
        'modal auxiliary': 'modal auxiliary',
        'noun': 'noun',
        'number': 'number',
        'preposition': 'preposition',
        'pronoun': 'pronoun',
        'verb': 'verb'
    }
    df_cefrj['pos_norm'] = df_cefrj['pos'].map(cefrj_pos_norm)

    with open('target_categories.json', 'r') as f:
        target_categories = json.load(f)

    # Feature Enrichment
    print("Enriching features (joining Oxford and CEFR-J)...")
    df_oxford['w_lemma'] = df_oxford.apply(
        lambda row: lemmatizer.lemmatize(row['w'].lower(), get_lemmatizer_pos(row['pos'])), axis=1
    )
    df_oxford['pos_norm'] = df_oxford['pos'].apply(normalize_pos_oxford_to_cefrj)

    # Left join
    df = pd.merge(
        df_oxford,
        df_cefrj[['headword', 'pos_norm', 'CoreInventory 1', 'CoreInventory 2', 'Threshold']],
        left_on=['w_lemma', 'pos_norm'],
        right_on=['headword', 'pos_norm'],
        how='left'
    )

    def build_anchor(row):
        labels = []
        for col in ['CoreInventory 1', 'CoreInventory 2', 'Threshold']:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                labels.append(str(val))

        if labels:
            return ", ".join(labels), False
        else:
            # Fallback
            pos_name = row['pos_norm']
            return f"{row['w']} {pos_name}", True

    anchors = df.apply(build_anchor, axis=1)
    df['semantic_anchor'] = [a[0] for a in anchors]
    df['is_derived_semantically'] = [a[1] for a in anchors]

    # Difficulty Scoring (re-using external sets logic)
    print("Calculating difficulty scores...")
    df_aoa = pd.read_excel('AoA_51715_words.xlsx', usecols=['Word', 'AoA_Kup_lem'])
    df_aoa['Word'] = df_aoa['Word'].astype(str).str.lower()
    df_aoa = df_aoa.groupby('Word')['AoA_Kup_lem'].mean().reset_index()

    df_conc = pd.read_excel('Concreteness_ratings_Brysbaert_et_al_BRM.xlsx', usecols=['Word', 'Conc.M'])
    df_conc['Word'] = df_conc['Word'].astype(str).str.lower()
    df_conc = df_conc.groupby('Word')['Conc.M'].mean().reset_index()

    df = pd.merge(df, df_aoa, left_on='w_lemma', right_on='Word', how='left')
    df = pd.merge(df, df_conc, left_on='w_lemma', right_on='Word', how='left')

    # Imputation
    medians = df.groupby(['orig_cefr', 'pos'], as_index=False)[['AoA_Kup_lem', 'Conc.M']].median()
    medians.columns = ['orig_cefr', 'pos', 'AoA_median', 'Conc_median']
    df = pd.merge(df, medians, on=['orig_cefr', 'pos'], how='left')
    df['AoA_Kup_lem'] = df['AoA_Kup_lem'].fillna(df['AoA_median'])
    df['Conc.M'] = df['Conc.M'].fillna(df['Conc_median'])
    # Global fallback
    df['AoA_Kup_lem'] = df['AoA_Kup_lem'].fillna(df['AoA_Kup_lem'].median())
    df['Conc.M'] = df['Conc.M'].fillna(df['Conc_median'])

    # Normalization
    aoa_min, aoa_max = df['AoA_Kup_lem'].min(), df['AoA_Kup_lem'].max()
    conc_min, conc_max = df['Conc.M'].min(), df['Conc.M'].max()
    cefr_map = {'A1': 0.0, 'A2': 0.2, 'B1': 0.4, 'B2': 0.6, 'C1': 0.8}

    df['AoA_norm'] = (df['AoA_Kup_lem'] - aoa_min) / (aoa_max - aoa_min)
    df['Conc_norm'] = (df['Conc.M'] - conc_min) / (conc_max - conc_min)
    df['CEFR_norm'] = df['orig_cefr'].map(cefr_map)

    df['difficulty_score'] = (0.5 * df['AoA_norm']) + (0.2 * (1 - df['Conc_norm'])) + (0.3 * df['CEFR_norm'])

    # Semantic Mapping
    print("Initialising Sentence-Transformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Computing category centroids...")
    cat_keys = list(target_categories.keys())
    cat_texts = [f"{k}: {', '.join(target_categories[k])}" for k in cat_keys]
    cat_embeddings = model.encode(cat_texts, convert_to_tensor=True)

    print("Computing word embeddings and mapping themes...")
    word_texts = df['semantic_anchor'].tolist()
    word_embeddings = model.encode(word_texts, convert_to_tensor=True, show_progress_bar=True)

    # Compute cosine similarities
    cos_sims = util.cos_sim(word_embeddings, cat_embeddings)
    best_cat_indices = torch.argmax(cos_sims, dim=1).tolist()
    df['theme'] = [cat_keys[idx] for idx in best_cat_indices]

    # Final sort and rank
    df = df.sort_values('difficulty_score').reset_index(drop=True)
    df['rank'] = df.index + 1

    # Distribution analysis
    print("\n--- TAXONOMY DISTRIBUTION ---")
    dist = df['theme'].value_counts()
    for theme, count in dist.items():
        print(f"{theme}: {count} ({count/len(df)*100:.2f}%)")

    # Export
    print("\nExporting to cefr_catalog.json...")
    output = []
    for _, row in df.iterrows():
        output.append({
            "rank": int(row['rank']),
            "w": row['w'],
            "pos": row['pos'],
            "tr": "", # Hebrew placeholder
            "theme": row['theme'],
            "aoa": round(float(row['AoA_Kup_lem']), 2),
            "concreteness": round(float(row['Conc.M']), 2),
            "difficulty_score": round(float(row['difficulty_score']), 4),
            "orig_cefr": row['orig_cefr'],
            "is_derived_semantically": bool(row['is_derived_semantically'])
        })

    with open('cefr_catalog.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
