# utils/data_processing.py
import csv
import os

def get_feature_names(csv_filepath):
    """Reads the header row from a CSV file to get feature names."""
    try:
        with open(csv_filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)
        return headers
    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}.")
        return None
    except Exception as e:
        print(f"Error reading CSV headers from {csv_filepath}: {e}")
        return None

def write_causal_edges(output_filepath, causal_edges):
    """Writes the identified causal edges to a CSV file."""
    try:
        with open(output_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            for edge in causal_edges:
                writer.writerow([edge[0], edge[1]])
        print(f"Identified causal edges written to {output_filepath}")
    except Exception as e:
        print(f"Error writing to output file {output_filepath}: {e}")