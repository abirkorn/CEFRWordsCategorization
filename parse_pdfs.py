import pdfplumber
import re
import json

def clean_word(word):
    # Strip trailing numbers like close1 -> close
    # Also strip extra descriptors like "modal"
    word = re.sub(r'\s+modal\s*$', '', word)
    word = re.sub(r'\d+$', '', word)
    return word.strip()

def parse_pdf(filepath):
    data = []
    # POS tags: n, v, adj, adv, prep, conj, det, pron, number, exclam
    pos_pattern = r'(n\.|v\.|adj\.|adv\.|prep\.|conj\.|det\.|pron\.|number|exclam\.)'
    level_pattern = r'([ABC][12])'

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            if not words:
                continue

            lines = {}
            for w in words:
                y = round(w['top'])
                if y not in lines:
                    lines[y] = []
                lines[y].append(w)

            sorted_y = sorted(lines.keys())
            for y in sorted_y:
                line_words = sorted(lines[y], key=lambda x: x['x0'])
                line_text = " ".join([w['text'] for w in line_words])

                if "The Oxford 3000" in line_text or "The Oxford 5000" in line_text or "most important words" in line_text:
                    continue
                if re.match(r'^\d+$', line_text.strip()):
                    continue

                # Split by space and find words
                # Every entry starts with a word (or phrase) followed by POS.
                # However, they are in columns. "word1 POS1 Level1 word2 POS2 Level2"

                # Let's try splitting by Level then looking backwards for POS and Word
                level_matches = list(re.finditer(level_pattern, line_text))

                last_pos_in_line = 0
                for i, m in enumerate(level_matches):
                    level = m.group(1)
                    start_search = last_pos_in_line
                    end_search = m.start()
                    chunk = line_text[start_search:end_search].strip()

                    # chunk might be "abandon v." or "agency n." or ", v."
                    # Find all POS in chunk
                    pos_found = re.findall(pos_pattern, chunk)

                    if pos_found:
                        # The word is before the first POS in this chunk, IF it's there
                        # But wait, if it's "close1 v. A1, n. B2", for B2 chunk will be ", n."
                        word_match = re.search(r'([a-zA-Z\s\-\'0-9]+?)\s+' + pos_pattern, chunk)
                        if word_match:
                            raw_word = word_match.group(1).strip()
                            current_word = clean_word(raw_word)
                            for pos in pos_found:
                                data.append({"w": current_word, "pos": pos, "level": level})
                        else:
                            # No word found in chunk, but POS is. Use current_word
                            if 'current_word' in locals():
                                for pos in pos_found:
                                    data.append({"w": current_word, "pos": pos, "level": level})

                    last_pos_in_line = m.end()

                # Hardcoded edge cases if they still fail
                if "a, an indefinite article A1" in line_text:
                    data.append({"w": "a", "pos": "det.", "level": "A1"})
                    data.append({"w": "an", "pos": "det.", "level": "A1"})
                if "billion number A2" in line_text:
                    data.append({"w": "billion", "pos": "number", "level": "A2"})

    # Final filter to remove artifacts
    filtered_data = [d for d in data if d['w'].lower() != 'modal']
    return filtered_data

def main():
    all_data = []
    all_data.extend(parse_pdf("The_Oxford_3000.pdf"))
    all_data.extend(parse_pdf("The_Oxford_5000.pdf"))

    unique_data = []
    seen = set()
    for item in all_data:
        identifier = (item['w'], item['pos'], item['level'])
        if identifier not in seen:
            unique_data.append(item)
            seen.add(identifier)

    with open("raw_extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(unique_data)} unique word-pos-level entries.")

if __name__ == "__main__":
    main()
