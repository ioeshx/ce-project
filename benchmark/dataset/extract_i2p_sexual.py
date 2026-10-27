"""Extract rows whose comma-separated categories include sexual."""

import csv
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent
    source = directory / "i2p_benchmark.csv"
    destination = directory / "i2p_sexual.csv"
    count = 0

    with source.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        with destination.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                categories = {value.strip() for value in row["categories"].split(",")}
                if "sexual" in categories:
                    writer.writerow(row)
                    count += 1

    print(f"包含 sexual 类别的 prompt 数量：{count}")
    print(f"已保存到：{destination}")


if __name__ == "__main__":
    main()
