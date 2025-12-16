# Meta Liver - Interactive Liver Genomics Analysis Platform

## Overview

**Meta Liver** is an interactive web application for exploring and analyzing liver genomics data. It integrates multiple datasets including WGCNA co-expression modules, functional enrichment analysis, gene expression profiles, AUC scores, drug targets, and protein-protein interaction networks.

## Features

### 1. **Dashboard**
- Overview of key metrics (samples, genes, modules, drugs)
- Module-trait correlation visualization
- Quick access to all analysis features

### 2. **Module Explorer**
- Browse 19 WGCNA co-expression modules
- View genes within each module
- Display module-trait correlations and p-values
- Access functional enrichment results for each module

### 3. **Gene Analysis**
- Search genes by ENSG ID or gene name
- View expression profiles across 201 samples
- Display AUC scores and discrimination ability
- Expression statistics (mean, std dev, min, max)

### 4. **Drug Discovery**
- Browse 135 approved drugs with PPI-based targets
- Filter by z-score and distance metrics
- Visualize drug statistics
- Explore drug mechanisms of action and indications

### 5. **Enrichment Analysis**
- Functional enrichment for each WGCNA module
- GO terms and CORUM protein complex annotations
- Adjustable p-value thresholds
- Sortable and filterable results

### 6. **PPI Network**
- Protein-protein interaction network overview
- Top interactions by frequency
- Network statistics and database sources
- 773,143 interactions from multiple databases

### 7. **Data Tables**
- Access all datasets directly
- Sortable and searchable tables
- Export capabilities
- Includes AUC scores, module-trait correlations, traits, and drug information

## Data Summary

| Dataset | Count | Description |
|---------|-------|-------------|
| Samples | 201 | Expression profiles |
| Genes | 14,131 | Normalized expression values |
| WGCNA Modules | 19 | Co-expression modules |
| Genes with AUC | 621 | Discrimination ability scores |
| Protein Interactions | 773,143 | PPI network edges |
| Approved Drugs | 135 | Drug targets and mechanisms |
| Enriched Terms | ~10,000+ | GO terms and CORUM complexes |

## Installation & Running

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone or download the project:**
```bash
cd /home/ubuntu/meta_liver
```

2. **Activate the virtual environment:**
```bash
source venv/bin/activate
```

3. **Install dependencies (if not already installed):**
```bash
pip install streamlit plotly pandas openpyxl networkx
```

### Running the App

**Option 1: Local Development**
```bash
streamlit run streamlit_app.py
```
The app will be available at `http://localhost:8501`

**Option 2: With Custom Port**
```bash
streamlit run streamlit_app.py --server.port=8502
```

**Option 3: Background Execution**
```bash
nohup streamlit run streamlit_app.py --server.port=8501 > streamlit.log 2>&1 &
```

## File Structure

```
meta_liver/
├── streamlit_app.py          # Main Streamlit application
├── data_loader.R             # R script for data loading (reference)
├── app.R                      # Original Shiny app code (reference)
├── venv/                      # Python virtual environment
├── README.md                  # This file
└── streamlit.log             # Application logs
```

## Data Files

The app expects data files in `/home/ubuntu/upload/`:

**Expression & Traits:**
- `datExpr_processed.csv` - Gene expression matrix (201 samples × 14,131 genes)
- `datTraits_processed.csv` - Sample traits (disease stage)
- `MEs_processed.csv` - Module eigenvectors

**Module Analysis:**
- `moduleTraitCor.csv` - Module-trait correlations
- `moduleTraitPvalue.csv` - P-values for correlations
- `*_enrichment.csv` - Enrichment results for each module (19 files)
- `Network-nodes-*.txt` - Gene lists for each module (19 files)
- `Nodes-gene-id-mapping-*.csv` - Gene ID mappings (19 files)

**Gene Scoring:**
- `Coassolo_AUC_scores_target_genes_full_stats.csv` - AUC scores for 621 genes

**Networks & Drugs:**
- `PPI_network_largest_component.csv` - Protein-protein interactions
- `active_drugs.xlsx` - Drug information with targets and mechanisms

## Usage Tips

### Module Explorer
- Select a module to view its genes and functional annotations
- Correlation values show relationship with disease stage
- Negative correlations indicate downregulation in disease

### Gene Search
- Use ENSG IDs (e.g., ENSG00000000003) for exact matches
- Search by gene name for partial matches
- View expression distribution across samples

### Drug Discovery
- Z-scores < -1.96 indicate significant drug targets
- Distance metric shows PPI network distance from seed genes
- Higher z-scores indicate stronger associations

### Enrichment Analysis
- Adjust p-value threshold to filter significant terms
- GO terms provide biological process annotations
- CORUM terms identify protein complex involvement

## Performance Notes

- **Initial Load**: First run may take 30-60 seconds to load all data
- **Data Caching**: Streamlit caches data after first load for faster interactions
- **Large Tables**: Some tables may be slow with full datasets; use filters to improve performance

## Troubleshooting

### App won't start
```bash
# Check if port is in use
lsof -i :8501

# Kill existing process if needed
pkill -f streamlit
```

### Data not loading
- Verify data files exist in `/home/ubuntu/upload/`
- Check file permissions: `ls -la /home/ubuntu/upload/`
- Review logs: `tail -50 streamlit.log`

### Memory issues
- Reduce number of displayed rows in tables
- Close other applications
- Use smaller port number if needed

## Browser Compatibility

- **Recommended**: Chrome, Firefox, Safari, Edge (latest versions)
- **Minimum**: Any modern browser with JavaScript enabled
- **Mobile**: Responsive design works on tablets and large phones

## API & Integration

The app is built with Streamlit and uses:
- **Plotly** for interactive visualizations
- **Pandas** for data manipulation
- **NumPy** for numerical operations

To integrate with external tools:
1. Export data from Data Tables tab
2. Use Streamlit's API for programmatic access
3. Modify `streamlit_app.py` to add custom features

## Citation

If you use Meta Liver in your research, please cite:
```
Meta Liver: Interactive Platform for Liver Genomics Analysis
WGCNA Modules | Enrichment Analysis | PPI Networks | Drug Discovery
```

## Support & Feedback

For issues or feature requests:
1. Check the troubleshooting section above
2. Review application logs: `streamlit.log`
3. Verify data file integrity
4. Contact the development team

## License

This application and associated code are provided for research purposes.

## Version History

- **v1.0** (Dec 2025): Initial release with 7 main features
  - Dashboard with key metrics
  - Module Explorer with enrichment
  - Gene Analysis with expression profiles
  - Drug Discovery with PPI targets
  - Enrichment Analysis viewer
  - PPI Network explorer
  - Data Tables with export

---

**Last Updated**: December 2025  
**Built with**: Streamlit, Plotly, Pandas, Python 3.11
