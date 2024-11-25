# =============================================================================
#                             Cleaning Collection FILE
# -----------------------------------------------------------------------------
# Cleaning out the TREC-CAR contents and only retrieving the msmarco passges
# Creating a new tsv collection file named msmarco_collection
# -----------------------------------------------------------------------------
# =============================================================================

input_file = '/home/scur2878/ConvDR/datasets/cast-shared/collection.tsv'
output_file = '/home/scur2878/ConvDR/datasets/cast-shared/msmarco_collection.tsv'

malformed_lines = 0

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        if "\t" not in line:
            malformed_lines += 1
            continue  # Skip malformed lines

        try:
            pid, text = line.strip().split("\t", 1)
            if int(pid) < 10000000:  # Keep only MS MARCO passages
                outfile.write(f"{pid}\t{text}\n")
        except ValueError:
            malformed_lines += 1
            continue  # Skip malformed lines
print('Finished generating new collection of MSMARCO only')
print(f"Total malformed lines: {malformed_lines}")



# =============================================================================
#                             MSMARCO Subset
# -----------------------------------------------------------------------------
# From the large MSMARCO Collection (Size: 8,635,155)
# We extract a 10k subset to test out in the other pipline for sanity check 
# before large deployment.
# The 10k subset (msmarco_qrels_subset_10k) stores 10k lines where the passage
# ID matches with the pre-processed cast-19 qrel file containing 21,726 lines.
# -----------------------------------------------------------------------------
# =============================================================================
input_file = "/home/scur2878/ConvDR/datasets/cast-shared/msmarco_collection.tsv"
qrels_path = "/home/scur2878/ConvDR/datasets/cast-19/qrels.tsv"
output_file = "/home/scur2878/ConvDR/datasets/cast-shared/msmarco_qrels_subset_10k.tsv"

qrels_ids = set()
with open(qrels_path, "r") as f:
    for line in f:
        _, _, pid, _ = line.strip().split("\t")
        qrels_ids.add(int(pid))

subset_size = 10000
subset = []

with open(input_file, "r") as f:
    for line in f:
        pid, text = line.strip().split("\t", 1)
        if int(pid) in qrels_ids:
            subset.append(line)
        if len(subset) >= subset_size:
            break

# Save the subset to a new file
with open(output_file, "w") as f:
    f.writelines(subset)

print(f"New subset created with {len(subset)} lines. Saved to {output_file}.")
