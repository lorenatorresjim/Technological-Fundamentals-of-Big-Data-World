from mpi4py import MPI
import pandas as pd
import time
import matplotlib.pyplot as plt


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        pattern = input("Enter the pattern : ").upper()
        start_time = time.time()
        df = pd.read_csv('proteins.csv')
        df['sequence'] = df['sequence'].str.upper()
        chunk_size = len(df) // size
        chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(size-1)]
        chunks.append(df.iloc[(size-1)*chunk_size:])
        chunks = [chunk.reset_index(drop=True) for chunk in chunks]
    else:
        pattern = None
        start_time = None
        chunks = None

    pattern = comm.bcast(pattern, root=0)
    chunk = comm.scatter(chunks, root=0)

    counts = chunk['sequence'].str.count(pattern)
    protein_counts = {}
    for protid, count in zip(chunk['protid'], counts):
        if count > 0:
            protein_counts[protid] = protein_counts.get(protid, 0) + count

    max_hydrofob = chunk.groupby('protid')['hydrofob'].max().to_dict()

    all_counts = comm.gather(protein_counts, root=0)
    all_hydrofob = comm.gather(max_hydrofob, root=0)

    if rank == 0:
        merged_counts = {}
        for d in all_counts:
            for k, v in d.items():
                merged_counts[k] = merged_counts.get(k, 0) + v

        merged_hydrofob = {}
        for d in all_hydrofob:
            for k, v in d.items():
                if k not in merged_hydrofob or v > merged_hydrofob[k]:
                    merged_hydrofob[k] = v

        total_time = time.time() - start_time

        if not merged_counts:
            print("No occurrences found for pattern:", pattern)
        else:
            sorted_counts = sorted(merged_counts.items(), key=lambda x: x[1], reverse=True)
            max_count = sorted_counts[0][1]
            max_proteins = [pid for pid, cnt in sorted_counts if cnt == max_count]
            max_protein = max(max_proteins, key=lambda pid: merged_hydrofob[pid])

            top_10 = sorted_counts[:10]

            protein_ids = [str(pid) for pid, cnt in top_10]
            num_sucesos = [cnt for pid, cnt in top_10]

            plt.figure(figsize=(12, 6))
            plt.bar(protein_ids, num_sucesos, color='skyblue')
            plt.xlabel('Protein ID')
            plt.ylabel('Number of Occurrences')
            plt.title(f'Top protein matches for pattern \"{pattern}\"')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('protein_pattern_counts.png')
            plt.show()

            print(f"Protein with max occurrences: {max_protein}, Occurrences: {max_count}")
            print(f"Hydrophobicity: {merged_hydrofob[max_protein]}")
            print(f"Execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
