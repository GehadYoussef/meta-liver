"""Copy the bulk cohort, in-vitro iHeps and WGCNA drug tables from the Meta Liver project
into data/, converting parquet to CSV, and append source, md5 and size to data/MANIFEST.csv.

usage: python -I scripts/09_import_metaliver_data.py [path/to/meta-liver/meta-liver-data]
needs: pandas, pyarrow
"""
import csv, hashlib, shutil, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Documents" / "meta-liver" / "meta-liver-data"
DATA = ROOT / "data"
manifest_rows = []

def record(dst: Path, src: Path):
    md5 = hashlib.md5(dst.read_bytes()).hexdigest()
    manifest_rows.append([dst.relative_to(ROOT).as_posix(), f"meta-liver/{src.relative_to(SRC.parent).as_posix()}",
                          md5, dst.stat().st_size])
    print(f"  {dst.relative_to(ROOT)}  ({dst.stat().st_size/1e6:.1f} MB)")

print("Bulk cohorts")
for f in sorted((SRC / "Bulk_omics").glob("*/*")):
    if f.suffix not in (".tsv", ".txt"):
        continue
    dst = DATA / "bulk_cohorts" / f.parent.name / (f.stem + ".tsv")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(f, dst)
    record(dst, f)

print("In-vitro iHeps model")
for f in sorted((SRC / "stem_cell_model").glob("*.parquet")):
    dst = DATA / "invitro" / (f.stem + ".csv.gz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pd.read_parquet(f).to_csv(dst, index=False, compression={"method": "gzip", "mtime": 0})
    record(dst, f)

print("Drug annotation")
f = SRC / "wgcna" / "active_drugs.parquet"
dst = DATA / "drugs" / "wgcna_active_drugs.csv"
dst.parent.mkdir(parents=True, exist_ok=True)
d = pd.read_parquet(f)
d["Drug Targets"] = d["Drug Targets"].str.replace(r"\s*\n\s*", " ", regex=True)
d.to_csv(dst, index=False)
record(dst, f)

with open(DATA / "MANIFEST.csv", "a", newline="", encoding="utf-8") as fh:
    csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC).writerows(manifest_rows)
print(f"Added {len(manifest_rows)} files to data/MANIFEST.csv")
