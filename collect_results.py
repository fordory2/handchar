"""汇总 ensemble.py 写出的结果 CSV，渲染成可直接贴进报告的 Markdown 表 + 排序。

配合 ensemble.py 的 --tag / --results_csv 使用：每次评估往同一个 CSV 追加一行，
最后跑一次本脚本即可得到对比表（默认按 accuracy 降序）。

用法:
  python collect_results.py results.csv                 # 打印表 + 写 results_table.md
  python collect_results.py results.csv --sort macro_f1 # 按其他列排序
"""
import argparse
import csv
import os

NICE = {
    "tag": "配置", "members": "成员数", "accuracy": "Acc",
    "balanced_accuracy": "Balanced", "macro_f1": "Macro-F1",
    "macro_auc": "Macro-AUC", "case_insensitive_accuracy": "大小写不敏感", "tta": "TTA",
}
ORDER = ["tag", "members", "accuracy", "balanced_accuracy", "macro_f1",
         "macro_auc", "case_insensitive_accuracy", "tta"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="ensemble.py --results_csv 写出的 CSV")
    ap.add_argument("--sort", default="accuracy",
                    help="排序列 (默认 accuracy; 可选 macro_f1/macro_auc/balanced_accuracy)")
    ap.add_argument("--out", default="results_table.md", help="输出的 Markdown 文件")
    args = ap.parse_args()

    if not os.path.exists(args.csv_path):
        raise SystemExit("找不到 %s (先用 ensemble.py --results_csv %s 生成几行)"
                         % (args.csv_path, args.csv_path))

    with open(args.csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("CSV 为空")

    def keyf(r):
        try:
            return float(r.get(args.sort, "") or -1)
        except ValueError:
            return -1
    rows.sort(key=keyf, reverse=True)

    cols = [c for c in ORDER if c in rows[0]]
    header = "| " + " | ".join(NICE.get(c, c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    table = "\n".join([header, sep] + body)

    md = "# 实验结果汇总 (按 %s 降序)\n\n%s\n" % (NICE.get(args.sort, args.sort), table)
    print(table)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print("\n✔ 已写出 %s (%d 行)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
