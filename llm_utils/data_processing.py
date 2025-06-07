import csv


def get_feature_names(csv_filepath):
    """
    Reads and returns the header row from a CSV file.

    This function opens a specified CSV file, reads the first line, and
    interprets it as the column headers or feature names.

    Parameters
    ----------
    csv_filepath : str
        The full path to the input CSV file.

    Returns
    -------
    list of str
        A list containing the column names from the header row of the CSV file.
    """
    with open(csv_filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
    return headers


def write_causal_edges(output_filepath, causal_edges):
    """
    Writes a list of causal edges to a CSV file.

    Each edge is written as a new row in the specified output file, with the
    first element being the cause and the second element being the effect.

    Parameters
    ----------
    output_filepath : str
        The full path for the output CSV file. The file will be created or
        overwritten.
    causal_edges : iterable of (str, str)
        An iterable (e.g., a list of tuples or lists) where each inner
        element represents a directed edge. For example: `[('cause1', 'effect1'),
        ('cause2', 'effect2')]`.

    """
    with open(output_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for edge in causal_edges:
            writer.writerow([edge[0], edge[1]])
