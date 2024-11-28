"""IN THIS FILE WE CONVERT THE TSV TO JSON FILE JUST LIKE IN THE PYSERINI GITHUB"""

import json
import os
import argparse

def convert_collection(args):
    print('Converting collection...')
    file_index = 0
    output_jsonl_file = None

    try:
        with open(args.collection_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Skip empty lines
                if not line.strip():
                    print(f"Skipping empty line {i}")
                    continue

                # Parse TSV line
                try:
                    parts = line.rstrip().split('\t', 1)  # Split only on the first tab
                    if len(parts) != 2:
                        raise ValueError(f"Line {i} is malformed: {line.strip()}")
                    doc_id, doc_text = parts
                    doc_json = json.dumps({"id": doc_id.strip(), "contents": doc_text.strip()})
                except ValueError as e:
                    print(f"Skipping line {i} due to error: {e}")
                    continue

                # Handle output file rotation
                if i % args.max_docs_per_file == 0:
                    if output_jsonl_file:
                        output_jsonl_file.close()
                    output_path = os.path.join(args.output_folder, f'docs{file_index:02d}.json')
                    output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
                    file_index += 1

                # Write JSON to file
                output_jsonl_file.write(doc_json + '\n')

                if i % 100000 == 0:
                    print(f'Converted {i:,} docs, writing into file {file_index}')

        # Close last open file
        if output_jsonl_file:
            output_jsonl_file.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert tab-separated collection into JSONL files.')
    parser.add_argument('--collection-path', required=True, help='Path to the collection file.')
    parser.add_argument('--output-folder', required=True, help='Output folder.')
    parser.add_argument('--max-docs-per-file', default=1000000, type=int,
                        help='Maximum number of documents in each JSONL file.')

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    convert_collection(args)
    print('Done!')
