from pathlib import Path
import pandas as pd


class Olist:

    def get_data(self):
        """
        Load all Olist CSV files into a dictionary of DataFrames.
        """
        csv_path = Path("~/.workintech/olist/data/csv").expanduser()

        file_paths = list(csv_path.glob("*.csv"))

        file_names = [file.name for file in file_paths]

        key_names = [
            name.replace("olist_", "")
                .replace("_dataset.csv", "")
                .replace(".csv", "")
            for name in file_names
        ]

        data = {}

        for key, file_path in zip(key_names, file_paths):
            data[key] = pd.read_csv(file_path)

        return data

    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")
