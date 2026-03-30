import argparse
import csv
from typing import Dict, List


def read_overall_row(path: str) -> Dict[str, str]:
    with open(path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row.get("scope", "").lower() == "overall":
                return row
    raise ValueError(f"No overall row found in {path}")


def to_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return 0.0
    return float(value)


def build_comparison_rows(left_name: str, left_row: Dict[str, str], right_name: str, right_row: Dict[str, str]) -> List[Dict[str, str]]:
    metrics = [
        "miou",
        "forest_precision",
        "forest_recall",
        "forest_f1",
        "change_precision",
        "change_recall",
        "change_f1",
    ]

    out_rows: List[Dict[str, str]] = []
    for metric in metrics:
        left_value = to_float(left_row, metric)
        right_value = to_float(right_row, metric)
        delta = right_value - left_value

        out_rows.append(
            {
                "metric": metric,
                left_name: f"{left_value:.6f}",
                right_name: f"{right_value:.6f}",
                "delta_right_minus_left": f"{delta:.6f}",
            }
        )

    return out_rows


def write_report(path: str, rows: List[Dict[str, str]], left_name: str, right_name: str) -> None:
    fieldnames = ["metric", left_name, right_name, "delta_right_minus_left"]
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two baseline metric CSV reports (overall row only).")
    parser.add_argument("--left-csv", required=True, help="Path to first baseline CSV (e.g., DeepLab)")
    parser.add_argument("--right-csv", required=True, help="Path to second baseline CSV (e.g., TESSERA embed)")
    parser.add_argument("--left-name", default="deeplab", help="Column label for left CSV")
    parser.add_argument("--right-name", default="tessera_embed", help="Column label for right CSV")
    parser.add_argument("--output-csv", required=True, help="Path to write side-by-side comparison CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    left_row = read_overall_row(args.left_csv)
    right_row = read_overall_row(args.right_csv)

    rows = build_comparison_rows(args.left_name, left_row, args.right_name, right_row)
    write_report(args.output_csv, rows, args.left_name, args.right_name)

    print(f"Comparison report written to {args.output_csv}")


if __name__ == "__main__":
    main()
