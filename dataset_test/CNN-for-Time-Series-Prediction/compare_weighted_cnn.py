import argparse
import glob
import os
import pandas as pd


NAMES_5 = ["KO", "AMD", "TSM", "GOOG", "WMT"]
NAMES_25 = ["AAPL", "ABBV", "AMZN", "BABA", "BRK-B", "C", "COST", "CVX", "DIS",
            "GE", "INTC", "MSFT", "nvda", "pypl", "QQQ", "SBUX", "T", "TSLA",
            "WFC", "KO", "AMD", "TSM", "GOOG", "WMT"]
NAMES_50 = ["aal", "AAPL", "ABBV", "AMD", "amgn", "AMZN", "BABA",
            "bhp", "bidu", "biib", "BRK-B", "C", "cat", "cmcsa", "cmg",
            "cop", "COST", "crm", "CVX", "dal", "DIS", "ebay", "GE",
            "gild", "gld", "GOOG", "gsk", "INTC", "KO", "mrk", "MSFT",
            "mu", "nke", "nvda", "orcl", "pep", "pypl", "qcom", "QQQ",
            "SBUX", "T", "tgt", "tm", "TSLA", "TSM", "uso", "v", "WFC",
            "WMT", "xlf"]


def resolve_symbols(n):
    if n == 5:
        return NAMES_5
    if n == 25:
        return NAMES_25
    if n == 50:
        return NAMES_50
    raise ValueError("num-stocks must be 5, 25, or 50")


def newest_file(pattern):
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-stocks", type=int, default=5, choices=[5, 25, 50])
    parser.add_argument("--tags", nargs="*", default=None, help="tags to compare")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    symbols = resolve_symbols(args.num_stocks)

    if args.tags:
        tags = args.tags
    else:
        pattern = os.path.join(base_dir, f"test_result_{args.num_stocks}_*")
        tags = [os.path.basename(p).split(f"test_result_{args.num_stocks}_", 1)[1]
                for p in glob.glob(pattern) if os.path.isdir(p)]
        tags.sort()

    rows = []
    for tag in tags:
        result_dir = os.path.join(base_dir, f"test_result_{args.num_stocks}_{tag}")
        for symbol in symbols:
            pattern = os.path.join(result_dir, "**", f"{symbol}_sentiment_*_eval.csv")
            eval_file = newest_file(pattern)
            if not eval_file:
                rows.append({"Tag": tag, "Stock_symbol": symbol, "MAE": None, "MSE": None, "R2": None})
                continue
            df = pd.read_csv(eval_file)
            rows.append({
                "Tag": tag,
                "Stock_symbol": symbol,
                "MAE": float(df["MAE"].iloc[0]),
                "MSE": float(df["MSE"].iloc[0]),
                "R2": float(df["R2"].iloc[0]),
            })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(base_dir, f"comparison_cnn_weighted_{args.num_stocks}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
