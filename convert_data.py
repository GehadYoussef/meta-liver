#!/usr/bin/env python3
"""
Convert all CSV and Excel files in meta-liver-data folder to Parquet format
Run this script in the meta-liver folder to convert your data

Usage:
    python3 convert_data.py
"""

import pandas as pd
from pathlib import Path
import sys

def convert_files():
    """Convert all CSV and Excel files to Parquet"""
    
    data_dir = Path("meta-liver-data")
    
    if not data_dir.exists():
        print("❌ Error: meta-liver-data folder not found!")
        print(f"   Expected location: {data_dir.absolute()}")
        sys.exit(1)
    
    print("=" * 70)
    print("Meta Liver Data Conversion: CSV/Excel → Parquet")
    print("=" * 70)
    print()
    
    # Find all CSV and Excel files
    csv_files = list(data_dir.rglob("*.csv"))
    excel_files = list(data_dir.rglob("*.xlsx")) + list(data_dir.rglob("*.xls"))
    txt_files = list(data_dir.rglob("*.txt"))
    
    total_files = len(csv_files) + len(excel_files)
    
    print(f"Found:")
    print(f"  • {len(csv_files)} CSV files")
    print(f"  • {len(excel_files)} Excel files")
    print(f"  • {len(txt_files)} TXT files (will keep as-is)")
    print()
    
    if total_files == 0:
        print("❌ No CSV or Excel files found!")
        sys.exit(1)
    
    converted = 0
    failed = 0
    
    # Convert CSV files
    print("Converting CSV files...")
    print("-" * 70)
    for csv_file in csv_files:
        try:
            print(f"  {csv_file.relative_to(data_dir)}", end=" ... ")
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Create output path (replace .csv with .parquet)
            parquet_file = csv_file.with_suffix(".parquet")
            
            # Save as Parquet
            df.to_parquet(parquet_file, compression="snappy", index=False)
            
            # Get file sizes
            csv_size = csv_file.stat().st_size / 1024 / 1024  # MB
            parquet_size = parquet_file.stat().st_size / 1024 / 1024  # MB
            compression = (1 - parquet_size / csv_size) * 100
            
            print(f"✓ ({csv_size:.1f}MB → {parquet_size:.1f}MB, {compression:.0f}% smaller)")
            
            # Delete original CSV
            csv_file.unlink()
            converted += 1
            
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
    
    # Convert Excel files
    print()
    print("Converting Excel files...")
    print("-" * 70)
    for excel_file in excel_files:
        try:
            print(f"  {excel_file.relative_to(data_dir)}", end=" ... ")
            
            # Read Excel
            df = pd.read_excel(excel_file)
            
            # Create output path (replace .xlsx/.xls with .parquet)
            parquet_file = excel_file.with_suffix(".parquet")
            
            # Save as Parquet
            df.to_parquet(parquet_file, compression="snappy", index=False)
            
            # Get file sizes
            excel_size = excel_file.stat().st_size / 1024 / 1024  # MB
            parquet_size = parquet_file.stat().st_size / 1024 / 1024  # MB
            compression = (1 - parquet_size / excel_size) * 100
            
            print(f"✓ ({excel_size:.1f}MB → {parquet_size:.1f}MB, {compression:.0f}% smaller)")
            
            # Delete original Excel
            excel_file.unlink()
            converted += 1
            
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
    
    # Summary
    print()
    print("=" * 70)
    print("Conversion Summary")
    print("=" * 70)
    print(f"✓ Successfully converted: {converted} files")
    if failed > 0:
        print(f"✗ Failed: {failed} files")
    print()
    
    if failed == 0:
        print("✅ All files converted successfully!")
        print()
        print("Next steps:")
        print("  1. Commit the changes to Git:")
        print("     git add meta-liver-data/")
        print("     git commit -m 'Add Parquet data files'")
        print("  2. Push to GitHub:")
        print("     git push origin main")
        print("  3. Streamlit Cloud will automatically redeploy")
        print()
    else:
        print(f"⚠️  {failed} files failed to convert. Check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = convert_files()
    sys.exit(0 if success else 1)
