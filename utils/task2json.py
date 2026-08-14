import json
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Convert LIBERO tasks parquet to JSON mapping.')
    parser.add_argument('--input', type=str, default='/ssd/yusun/datasets/ori_libero/meta/tasks.parquet',
                        help='Path to the tasks parquet file.')
    parser.add_argument('--output', type=str, default='/ssd/yusun/datasets/tasks.json',
                        help='Output JSON file path.')
    args = parser.parse_args()

    file = pd.read_parquet(args.input, engine='pyarrow')
    out = {}
    for i in range(len(file)):
        out[i] = file.iloc[i].name

    with open(args.output, 'w') as f:
        json.dump(out, f)
    print(f'Saved {len(out)} tasks to {args.output}')


if __name__ == '__main__':
    main()