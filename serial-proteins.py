import csv
import time
import matplotlib.pyplot as plt

def main():
    pattern = input("Enter the pattern : ").upper() #Converts the input pattern to uppercase
    start_time = time.time()
    protein_counts = {}
    protein_hydrofob = {}
    with open('proteins.csv', newline='') as proteins:
        reader = csv.DictReader(proteins)
        for row in reader:
            protid = row['protid']
            sequence = row['sequence'].upper()
            hydrofob = float(row['hydrofob'])
            count = sequence.count(pattern) # Counts occurrences of the pattern
            if count > 0:
                protein_counts[protid] = protein_counts.get(protid, 0) + count # Sum occurrences
                if protid not in protein_hydrofob or hydrofob > protein_hydrofob[protid]:
                    protein_hydrofob[protid] = hydrofob
    end_time = time.time()
    total_time = end_time - start_time

    if not protein_counts:
        print("No occurrences found for pattern:", pattern)
    else:
        sorted_counts = sorted(protein_counts.items(), key=lambda x: x[1], reverse=True) # order by n of occurrences
        max_count = sorted_counts[0][1] # highest occurrence
        max_count_proteins = [id for id, count in sorted_counts if count == max_count] # all proteins with highest occurrence
        max_protein = max(max_count_proteins, key=lambda id: protein_hydrofob[id]) # protein with highest hydrofob among them
        top_10 = sorted_counts[:10]
        if max_protein not in [id for id, count in top_10]:
            top_10.append((max_protein, protein_counts[max_protein]))
        plot_proteins = [id for id, count in top_10]
        plot_counts = [count for id, count in top_10]
        plt.figure(figsize=(12, 6))
        plt.bar(plot_proteins, plot_counts, color='skyblue')
        plt.xlabel('Protein ID')
        plt.ylabel('Number of Occurrences')
        plt.title(f'Top protein matches for pattern "{pattern}"')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('protein_pattern_occurrences.png')
        plt.show()
        print(f"ID of protein with maximum occurrences: {max_protein}, Occurrences: {max_count}")
        print(f"Hydrophobicity of that protein: {protein_hydrofob[max_protein]}")
        print(f"Execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
