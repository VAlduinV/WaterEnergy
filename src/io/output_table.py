from prettytable import PrettyTable


def display_village_clusters(data, labels):
    """
        Display the village names and their corresponding cluster labels in a table.

        Args:
            data (pd.DataFrame): The data containing the village names.
            labels (np.ndarray): Array of cluster labels corresponding to each row in data.
        """
    table = PrettyTable()  # Initialize PrettyTable
    table.field_names = ["Village Name", "Cluster"]  # Set table field names
    for name, label in zip(
            data["admin4Na_1"].head(20), labels[:20]
    ):  # Iterate over first 20 entries
        table.add_row([name, label])  # Add row to table
    print(table)  # Print table
    return labels
