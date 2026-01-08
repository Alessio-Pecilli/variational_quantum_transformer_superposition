#!/usr/bin/env python3
"""
Script per pre-scaricare il dataset PTB in locale.
Esegui questo script sul tuo PC con internet, poi copia il file sul cluster HPC.

Uso: python download_ptb.py
"""

from datasets import load_dataset

OUTPUT_FILE = "ptb_sentences.txt"

def main():
    print("Scaricamento PTB da HuggingFace...")
    dataset_dict = load_dataset("ptb_text_only", trust_remote_code=True)
    
    all_sentences = []
    
    for split in ['train', 'validation', 'test']:
        if split in dataset_dict:
            for entry in dataset_dict[split]:
                sentence = entry["sentence"].strip()
                if sentence:
                    all_sentences.append(sentence)
            print(f"  - {split}: {len(dataset_dict[split])} frasi")
    
    # Salva tutte le frasi in un file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sentence in all_sentences:
            f.write(sentence + '\n')
    
    print(f"\n✅ Salvate {len(all_sentences)} frasi in '{OUTPUT_FILE}'")
    print(f"\nOra copia '{OUTPUT_FILE}' sul cluster HPC nella stessa cartella del progetto.")
    
    # Statistiche per lunghezza
    print("\n📊 Statistiche lunghezza frasi:")
    from collections import Counter
    lengths = Counter(len(s.split()) for s in all_sentences)
    for length in sorted(lengths.keys()):
        print(f"   {length} parole: {lengths[length]} frasi")

if __name__ == "__main__":
    main()
