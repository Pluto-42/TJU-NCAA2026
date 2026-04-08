#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验三套提交文件与 matchupProbabilities.csv 的差异

比较每个提交与基准表的：
  - 不同行数量
  - 差异样本（前若干条）
  - 最大值、最小值、均值等统计
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 基准表与三个提交路径
BASE_PATH = os.path.join(ROOT, "matchupProbabilities.csv")
SUBMISSIONS = [
    ("A_Duke", "outputs/submission_A_Duke.csv"),
    ("B_Michigan", "outputs/submission_B_Michigan.csv"),
    ("C_Arizona", "outputs/submission_C_Arizona.csv"),
]
# 差异行最多在控制台打印条数
MAX_DIFF_ROWS = 30
# 是否将全部差异行导出为 CSV
EXPORT_DIFF_CSV = True
# 三套提交之间相互比较
COMPARE_SUBMISSIONS = True


def main():
    # 加载基准表（可能含 index 列）
    base = pd.read_csv(BASE_PATH)
    if "ID" not in base.columns and base.shape[1] >= 2:
        base.columns = ["_idx", "ID", "Pred"][: base.shape[1]]
    base = base[["ID", "Pred"]].copy()
    base["ID"] = base["ID"].astype(str)
    base["Pred"] = base["Pred"].astype(float)

    print("=" * 80)
    print("基准表 matchupProbabilities.csv 统计")
    print("=" * 80)
    print(f"  行数: {len(base)}")
    print(f"  Pred 最小值: {base['Pred'].min():.6f}")
    print(f"  Pred 最大值: {base['Pred'].max():.6f}")
    print(f"  Pred 均值:   {base['Pred'].mean():.6f}")
    print()

    # 建立基准 ID -> Pred 映射
    base_map = dict(zip(base["ID"], base["Pred"]))

    for label, rel_path in SUBMISSIONS:
        fp = os.path.join(ROOT, rel_path)
        if not os.path.exists(fp):
            print(f"[{label}] 文件不存在: {fp}\n")
            continue

        sub = pd.read_csv(fp)
        sub["ID"] = sub["ID"].astype(str)
        sub["Pred"] = sub["Pred"].astype(float)

        # 找出与基准不同的行
        diffs = []
        for i, row in sub.iterrows():
            uid = row["ID"]
            pred_sub = row["Pred"]
            pred_base = base_map.get(uid)
            if pred_base is None:
                diffs.append({"ID": uid, "base": None, "sub": pred_sub, "diff": None})
            elif abs(pred_sub - pred_base) > 1e-9:
                diffs.append({
                    "ID": uid,
                    "base": pred_base,
                    "sub": pred_sub,
                    "diff": pred_sub - pred_base,
                })

        df_diff = pd.DataFrame(diffs)

        print("=" * 80)
        print(f"提交 {label} vs 基准")
        print("=" * 80)
        print(f"  总行数:        {len(sub)}")
        print(f"  与基准不同行数: {len(df_diff)}")
        if len(df_diff) > 0:
            df_valid = df_diff[df_diff["base"].notna()].copy()
            if len(df_valid) > 0:
                print(f"  Pred(基准) 最小: {df_valid['base'].min():.6f}")
                print(f"  Pred(基准) 最大: {df_valid['base'].max():.6f}")
                print(f"  Pred(提交) 最小: {df_valid['sub'].min():.6f}")
                print(f"  Pred(提交) 最大: {df_valid['sub'].max():.6f}")
                print(f"  差异(diff) 最小: {df_valid['diff'].min():.6f}")
                print(f"  差异(diff) 最大: {df_valid['diff'].max():.6f}")
            print()
            print("  差异样本（前 {} 条）:".format(min(MAX_DIFF_ROWS, len(df_diff))))
            print("-" * 80)
            sample = df_diff.head(MAX_DIFF_ROWS)
            for _, r in sample.iterrows():
                if r["base"] is not None:
                    print(f"    {r['ID']:<20} base={r['base']:.4f}  sub={r['sub']:.4f}  diff={r['diff']:+.4f}")
                else:
                    print(f"    {r['ID']:<20} base=N/A  sub={r['sub']:.4f}")
        else:
            print("  无差异")

        # 导出全部差异行到 CSV（便于审查）
        if len(df_diff) > 0:
            out_path = os.path.join(ROOT, "outputs", f"diff_{label}_vs_base.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df_diff.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  全部差异行已导出: {out_path}")
        print()

    # 三套提交之间的相互比较
    print("=" * 80)
    print("三套提交之间的差异")
    print("=" * 80)
    subs = {}
    for label, rel_path in SUBMISSIONS:
        fp = os.path.join(ROOT, rel_path)
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df["ID"] = df["ID"].astype(str)
            subs[label] = dict(zip(df["ID"], df["Pred"].astype(float)))

    if len(subs) >= 2:
        labels = list(subs.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                la, lb = labels[i], labels[j]
                diffs = [(uid, subs[la][uid], subs[lb][uid], subs[lb][uid] - subs[la][uid])
                         for uid in subs[la] if uid in subs[lb] and abs(subs[la][uid] - subs[lb][uid]) > 1e-9]
                print(f"\n  {la} vs {lb}: {len(diffs)} 行不同")
                if len(diffs) > 0:
                    df_pair = pd.DataFrame(diffs, columns=["ID", f"Pred_{la}", f"Pred_{lb}", "diff"])
                    print(f"    Pred 范围: {la} [{df_pair[f'Pred_{la}'].min():.4f}, {df_pair[f'Pred_{la}'].max():.4f}], "
                          f"{lb} [{df_pair[f'Pred_{lb}'].min():.4f}, {df_pair[f'Pred_{lb}'].max():.4f}]")
                    sample = diffs[:10]
                    for uid, pa, pb, delta in sample:
                        print(f"      {uid}: {pa:.4f} vs {pb:.4f} (diff={delta:+.4f})")
                    out_path = os.path.join(ROOT, "outputs", f"diff_{la}_vs_{lb}.csv")
                    pd.DataFrame(diffs, columns=["ID", f"Pred_{la}", f"Pred_{lb}", "diff"]).to_csv(
                        out_path, index=False, encoding="utf-8-sig"
                    )
                    print(f"    已导出: {out_path}")
    print()


if __name__ == "__main__":
    main()
