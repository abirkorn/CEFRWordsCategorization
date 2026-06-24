import pandas as pd
import json
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure NLTK resources are available
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

def get_wn_pos(oxford_pos):
    """Maps Oxford POS tags to NLTK/WordNet POS constants."""
    if oxford_pos == 'n.':
        return wn.NOUN
    elif oxford_pos == 'v.' or oxford_pos == 'modal v.':
        return wn.VERB
    elif oxford_pos == 'adj.':
        return wn.ADJ
    elif oxford_pos == 'adv.':
        return wn.ADV
    return None

def get_lemmatizer_pos(oxford_pos):
    """Maps Oxford POS tags to Lemmatizer POS constants."""
    if oxford_pos == 'n.':
        return 'n'
    elif oxford_pos == 'v.' or oxford_pos == 'modal v.':
        return 'v'
    elif oxford_pos == 'adj.':
        return 'a'
    elif oxford_pos == 'adv.':
        return 'r'
    return 'n'

def get_lexname(word, oxford_pos, concreteness):
    # Functional POS list: pronoun, preposition, determiner, conjunction
    functional_pos = ['pron.', 'prep.', 'det.', 'conj.', 'exclam.', 'number']

    # THE "UNIVERSAL" GATEKEEPER RULE
    if oxford_pos in functional_pos or concreteness < 3.0:
        return "universal"

    # Map Oxford POS to WordNet POS
    wn_pos = get_wn_pos(oxford_pos)

    if wn_pos:
        synsets = wn.synsets(word, pos=wn_pos)
    else:
        synsets = wn.synsets(word)

    if synsets:
        return synsets[0].lexname()
    return "universal"

def main():
    print("Loading Oxford data...")
    with open('raw_extracted_data.json', 'r') as f:
        oxford_data = json.load(f)

    df_oxford = pd.DataFrame(oxford_data)
    df_oxford.rename(columns={'level': 'orig_cefr'}, inplace=True)

    # Lemmatize Oxford words for joining
    print("Lemmatizing Oxford words...")
    df_oxford['w_lemma'] = df_oxford.apply(
        lambda row: lemmatizer.lemmatize(row['w'].lower(), get_lemmatizer_pos(row['pos'])),
        axis=1
    )

    # Load AoA
    print("Loading AoA dataset...")
    df_aoa = pd.read_excel('AoA_51715_words.xlsx', usecols=['Word', 'AoA_Kup_lem'])
    df_aoa['Word'] = df_aoa['Word'].astype(str).str.lower()
    # Handle duplicates by taking mean
    df_aoa = df_aoa.groupby('Word')['AoA_Kup_lem'].mean().reset_index()

    # Load Concreteness
    print("Loading Concreteness dataset...")
    df_conc = pd.read_excel('Concreteness_ratings_Brysbaert_et_al_BRM.xlsx', usecols=['Word', 'Conc.M'])
    df_conc['Word'] = df_conc['Word'].astype(str).str.lower()
    # Handle duplicates by taking mean
    df_conc = df_conc.groupby('Word')['Conc.M'].mean().reset_index()

    # Merge on Lemma
    print("Merging datasets...")
    df = pd.merge(df_oxford, df_aoa, left_on='w_lemma', right_on='Word', how='left')
    df = pd.merge(df, df_conc, left_on='w_lemma', right_on='Word', how='left')

    # Imputation logic: Median by CEFR and POS
    print("Performing median imputation...")
    medians = df.groupby(['orig_cefr', 'pos'], as_index=False)[['AoA_Kup_lem', 'Conc.M']].median()
    medians.columns = ['orig_cefr', 'pos', 'AoA_median', 'Conc_median']

    df = pd.merge(df, medians, on=['orig_cefr', 'pos'], how='left')
    df['AoA_Kup_lem'] = df['AoA_Kup_lem'].fillna(df['AoA_median'])
    df['Conc.M'] = df['Conc.M'].fillna(df['Conc_median'])

    # Global fallbacks
    df['AoA_Kup_lem'] = df['AoA_Kup_lem'].fillna(df['AoA_Kup_lem'].median())
    df['Conc.M'] = df['Conc.M'].fillna(df['Conc.M'].median())

    # Normalization
    print("Normalizing scores...")
    aoa_min, aoa_max = df['AoA_Kup_lem'].min(), df['AoA_Kup_lem'].max()
    conc_min, conc_max = df['Conc.M'].min(), df['Conc.M'].max()

    df['AoA_norm'] = (df['AoA_Kup_lem'] - aoa_min) / (aoa_max - aoa_min)
    df['Conc_norm'] = (df['Conc.M'] - conc_min) / (conc_max - conc_min)

    # Formula: Score = (0.7 * AoA_norm) + (0.3 * (1 - Conc_norm))
    df['difficulty_score'] = (0.7 * df['AoA_norm']) + (0.3 * (1 - df['Conc_norm']))

    # Lexnames with Universal Rule
    print("Mapping Lexnames and applying Universal Rule...")
    df['lexname'] = df.apply(lambda row: get_lexname(row['w'], row['pos'], row['Conc.M']), axis=1)

    # Sort and rank
    df = df.sort_values('difficulty_score').reset_index(drop=True)
    df['rank'] = df.index + 1

    # Prepare final JSON structure
    print("Saving to continuous_catalog.json...")
    final_output = []
    for _, row in df.iterrows():
        entry = {
            "w": row['w'],
            "pos": row['pos'],
            "difficulty_score": round(float(row['difficulty_score']), 4),
            "rank": int(row['rank']),
            "theme": row['lexname'],
            "tr": "", # Placeholder for translation
            "concreteness": round(float(row['Conc.M']), 2),
            "orig_cefr": row['orig_cefr']
        }
        final_output.append(entry)

    with open('continuous_catalog.json', 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"Pipeline complete. {len(final_output)} words processed.")

    # Sanity check for "Universal" rule
    print("\n--- UNIVERSAL RULE CHECK ---")
    abstract_word = df[df['w'] == 'globalization']
    if not abstract_word.empty:
        print(f"Globalization theme: {abstract_word.iloc[0]['lexname']} (Conc: {abstract_word.iloc[0]['Conc.M']})")

    func_word = df[df['w'] == 'although']
    if not func_word.empty:
        print(f"Although theme: {func_word.iloc[0]['lexname']} (POS: {func_word.iloc[0]['pos']})")

if __name__ == "__main__":
    main()
