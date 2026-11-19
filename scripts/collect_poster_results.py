"""Collect the result tables used for the poster from the executed notebooks into results/poster/*.csv.

The notebooks show their results as tables but don't all save them; this reads the saved notebook
outputs (no models are retrained). Run from the repo root:

    python scripts/collect_poster_results.py [path/to/notebook/dir ...]

Each table is looked up by notebook name and cell index, in the given notebook directories
(default: notebooks/). Tables are written in long format with the column header levels joined by " | ".
"""
import json
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

TABLES = {
    # file name: (notebook, cell index)
    "sim_prediction": ("09_dsm_baseline", 3),
    "sim_clustering": ("08_ssc_baseline", 4),
    "real_prediction": ("09_dsm_baseline", 5),
    "real_clustering": ("08_ssc_baseline", 6),
    "leakage": ("10_imputation_comparison", 10),
    "imputation": ("10_imputation_comparison", 4),
    "support_missingness": ("06_missingness", 3),
    "missingness_mechanisms": ("06_missingness", 8),
    "missingness_survival": ("06_missingness", 11),
    "missingness_support": ("06_missingness", 13),
}


def html_table(html):
    """pandas-style HTML table -> DataFrame (index levels become columns)."""
    table = BeautifulSoup(html, "html.parser").find("table")
    head = []
    for tr in table.find("thead").find_all("tr"):
        row = []
        for c in tr.find_all(["th", "td"]):
            row += [c.get_text(strip=True)] * int(c.get("colspan", 1))
        head.append(row)
    body, carry = [], {}
    for tr in table.find("tbody").find_all("tr"):
        cells, row, j = tr.find_all(["th", "td"]), [], 0
        it = iter(cells)
        n_cols = len(head[0])
        while len(row) < n_cols:
            if j in carry and carry[j][1] > 0:
                row.append(carry[j][0]); carry[j][1] -= 1
            else:
                c = next(it)
                txt = c.get_text(strip=True)
                span = int(c.get("rowspan", 1))
                if span > 1:
                    carry[j] = [txt, span - 1]
                row.append(txt)
            j += 1
        body.append(row)
    n_idx = sum(1 for c in table.find("tbody").find("tr").find_all("th"))
    # header: column labels from all header rows; index names from the last header row if present
    idx_names = [h for h in head[-1][:n_idx]] if len(head) > 1 and any(head[-1][:n_idx]) else [f"level_{i}" for i in range(n_idx)]
    col_rows = [r for r in head if any(r[n_idx:])]
    cols = [" | ".join(x for x in parts if x) for parts in zip(*[r[n_idx:] for r in col_rows])]
    df = pd.DataFrame(body, columns=idx_names + cols)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main(dirs):
    out = Path("results/poster")
    out.mkdir(parents=True, exist_ok=True)
    for name, (nb_name, cell) in TABLES.items():
        paths = [p for d in dirs for p in Path(d).glob(f"{nb_name}.ipynb")]
        if not paths:
            print(f"skip {name}: {nb_name}.ipynb not found")
            continue
        nb = json.loads(paths[0].read_text(encoding="utf8"))
        html = next(o["data"]["text/html"] for o in nb["cells"][cell]["outputs"] if "text/html" in o.get("data", {}))
        df = html_table("".join(html))
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {df.shape} from {paths[0]}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["notebooks"])
