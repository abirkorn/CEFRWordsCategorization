import json

def main():
    with open("clustered_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    catalog = {}

    for item in data:
        level = item['level']
        cluster_id = item['cluster']

        if level not in catalog:
            catalog[level] = {"themes": {}}

        # Rule 5: Preserve standard POS tags (n., v., adj., adv.).
        # For structural/functional grammar components (prep., conj., det., pron., exclam.),
        # automatically route them via pure Python logic into the "universal" theme bucket.
        functional_pos = ['prep.', 'conj.', 'det.', 'pron.', 'exclam.']

        if item['pos'] in functional_pos or cluster_id == -1:
            theme_name = "universal"
        else:
            theme_name = f"cluster_{cluster_id}"

        if theme_name not in catalog[level]["themes"]:
            catalog[level]["themes"][theme_name] = []

        catalog[level]["themes"][theme_name].append({
            "w": item['w'],
            "pos": item['pos'],
            "tr": "" # Placeholder for Phase 2 translation
        })

    # Sort levels for clean output
    sorted_catalog = {k: catalog[k] for k in sorted(catalog.keys())}

    with open("cefr_catalog_draft.json", "w", encoding="utf-8") as f:
        json.dump(sorted_catalog, f, ensure_ascii=False, indent=2)

    print("Draft catalog generated as 'cefr_catalog_draft.json'.")

if __name__ == "__main__":
    main()
