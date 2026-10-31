import csv
import datetime
import os
from abc import ABC


def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))


class BaseEvaluator(ABC):

    def process_images(self, image_file_paths, **kwargs) -> dict:
        raise NotImplementedError

    def evaluate(self, image_folder, log_path, **kwargs):
        image_file_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if is_image_file(f)]
        image_file_paths.sort()

        output_dir = os.getenv("OUTPUT_DIR")
        assert output_dir is not None, "OUTPUT_DIR environment variable is not set."

        summary = self.process_images(image_file_paths, **kwargs)

        print("\nSummary Report")
        print("=" * 40)
        print(f"Total Images Processed : {summary['total']}")
        for label, value in summary.items():
            if label != 'total':
                print(f"{label.upper()} : {value}")

        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "image_folder": image_folder.replace(output_dir, ''),
            **summary
        }
        fieldnames = list(row.keys())

        log_path = os.path.join(output_dir, "metrics", log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_exists = os.path.exists(log_path)

        with open(log_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        print(f"\nResults appended to '{log_path}' ✅")

        # --- Show full table as a pandas DataFrame ---
        try:
            import pandas as pd

            print("\n📊 Evaluation Log Table:")
            df = pd.read_csv(log_path)
            print(df.to_string(index=False))
        except ImportError:
            print("\nPandas is not installed. Skipping table view.")




