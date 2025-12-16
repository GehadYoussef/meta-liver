# Meta Liver v2 - Deployment Guide

## Overview

Meta Liver v2 is a hypothesis engine for liver genomics research. It combines:
- Evidence-based scoring (hepatocyte + microenvironment + KG prior)
- Gene/drug dossiers with three linked panels
- Mechanistic path explanations
- Query-driven interface

## Files Structure

```
meta-liver/
├── streamlit_app_v2.py          # Main app (RENAME to streamlit_app.py)
├── evidence_scoring.py           # Scoring engine
├── gene_dossier.py              # Gene report template
├── mechanistic_paths.py          # Path explainer
├── data_loaders.py              # Data loading layer
├── requirements.txt              # Python dependencies
├── meta-liver-data/             # Parquet data files
│   ├── wgcna/
│   ├── single_omics/
│   ├── knowledge_graphs/
│   └── ppi_networks/
└── README.md                     # Documentation
```

## Deployment Steps

### Step 1: Prepare Local Files

1. **Rename the app file:**
   ```bash
   cd C:\Users\gonli\Documents\meta-liver
   ren streamlit_app_v2.py streamlit_app.py
   ```

2. **Verify all files are present:**
   ```bash
   dir /s *.py
   ```
   
   You should have:
   - streamlit_app.py (renamed from v2)
   - evidence_scoring.py
   - gene_dossier.py
   - mechanistic_paths.py
   - data_loaders.py

3. **Verify data folder:**
   ```bash
   dir meta-liver-data
   ```
   
   Should show: wgcna, single_omics, knowledge_graphs, ppi_networks

### Step 2: Commit to Git

```bash
cd C:\Users\gonli\Documents\meta-liver

# Stage all changes
git add .

# Commit
git commit -m "Deploy Meta Liver v2 - Hypothesis Engine

- Add evidence scoring engine (hepatocyte + microenv + KG)
- Add gene/drug dossier with three linked panels
- Add mechanistic path explainer
- Add query-driven UI with intent presets
- Include all Parquet data files"

# Push to GitHub
git push origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to https://streamlit.io/cloud

2. Click "New app"

3. Fill in:
   - **Repository:** GehadYoussef/meta-liver
   - **Branch:** main
   - **Main file path:** streamlit_app.py

4. Click "Deploy"

5. Wait 2-5 minutes for deployment

6. Your app will be live at:
   ```
   https://meta-liver-gehadyoussef.streamlit.app
   ```

## Testing Locally (Optional)

Before deploying, you can test locally:

```bash
cd C:\Users\gonli\Documents\meta-liver

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Features

### Evidence Scoring

**Hepatocyte Score** (Cell-Intrinsic)
- Median AUC across studies (dispersion-penalized)
- Direction agreement (sign of logFC)
- logFC magnitude (winsorized)

**Microenvironment Score** (Fibrosis/Inflammation)
- WGCNA module hubness (kME)
- Module-trait correlation
- Module annotation

**Knowledge Graph Prior** (Disease Relevance)
- MASH subgraph centrality
- Shortest path to drugs
- Cluster membership

### Intent-Based Weighting

- **Hepatocyte Targets:** 70% hep, 20% microenv, 10% KG
- **Fibrosis/NASH:** 20% hep, 70% microenv, 10% KG
- **Repurposing:** 25% hep, 25% microenv, 50% KG
- **Custom:** User-defined sliders

### Gene Dossier

Three linked panels:
1. **WGCNA Panel** - Module, kME, co-expressing genes, enrichment
2. **Knowledge Graph Panel** - Cluster neighbors, mechanistic paths
3. **Cross-Study Evidence** - AUC forest plot, heterogeneity badge

### Mechanistic Paths

- Gene → Drug routes with edge-type penalties
- Drug → Gene targets
- Plain-language explanations
- Network neighborhood explorer

## Troubleshooting

### "Module not found" errors

Make sure all Python files are in the same directory:
- streamlit_app.py
- evidence_scoring.py
- gene_dossier.py
- mechanistic_paths.py
- data_loaders.py

### "Parquet file not found" errors

Make sure meta-liver-data folder is in the same directory as streamlit_app.py:
```
meta-liver/
├── streamlit_app.py
├── meta-liver-data/
│   ├── wgcna/
│   ├── single_omics/
│   ├── knowledge_graphs/
│   └── ppi_networks/
```

### Slow loading

First load may take 1-2 minutes as Streamlit caches data. Subsequent loads are instant.

### Missing data

If a gene/drug is not found:
1. Try alternative names (e.g., ENSG ID vs gene symbol)
2. Check if it's in your original data files
3. Verify Parquet conversion was successful

## Next Steps

### Add More Features

1. **Drug Dossier** - Similar to gene dossier but drug-centric
2. **Batch Analysis** - Upload CSV of genes, get ranked reports
3. **Custom Pathways** - Let users define custom pathways
4. **Export Reports** - PDF/HTML export of dossiers
5. **Comparison Tool** - Compare two genes side-by-side

### Improve Scoring

1. **Weighted AUC** - Weight by sample size or expression prevalence
2. **Study-Specific Penalties** - Account for batch effects
3. **Temporal Scoring** - If you have time-series data
4. **Cell-Type Specificity** - If you have cell-type annotations

### Enhance Visualizations

1. **Interactive Network** - PyVis for network exploration
2. **Heatmaps** - Module-trait correlation heatmap
3. **Sankey Diagrams** - Mechanistic paths as Sankey
4. **3D Scatter** - Multi-dimensional evidence space

## Support

For issues or questions:
1. Check the README.md
2. Review error messages in Streamlit Cloud logs
3. Test locally first before deploying

## Version History

- **v2.0** (Current)
  - Evidence-based scoring engine
  - Gene dossier with three panels
  - Mechanistic path explainer
  - Query-driven UI with intent presets
  - Parquet data format

- **v1.0** (Previous)
  - Basic dashboard with tabs
  - Google Drive data loading
  - Static visualizations
