# Google Drive Integration Setup

## Overview

Your Meta Liver app is configured to load data from Google Drive. This guide explains how to set it up properly.

## Your Google Drive Folder

**Folder Link:** https://drive.google.com/drive/folders/1xM71_KnUtTEWpRwXsWN4DKVRSFiP6KEB

**Folder ID:** `1xM71_KnUtTEWpRwXsWN4DKVRSFiP6KEB`

## Folder Structure

```
meta-liver-data/
├── wgcna/
│   ├── datExpr_processed.csv
│   ├── datTraits_processed.csv
│   ├── MEs_processed.csv
│   ├── moduleTraitCor.csv
│   ├── moduleTraitPvalue.csv
│   ├── pathway/
│   │   ├── black_enrichment.csv
│   │   ├── brown_enrichment.csv
│   │   └── ... (19 total)
│   └── modules/
│       ├── Network-nodes-black.txt
│       ├── Network-nodes-brown.txt
│       ├── ... (19 total)
│       ├── Nodes-gene-id-mapping-black.csv
│       ├── Nodes-gene-id-mapping-brown.csv
│       └── ... (19 total)
├── ppi_networks/
│   ├── PPI_network_largest_component.csv
│   └── Early_MAFLD_Network/
│       ├── Centrality_RWR_result_pvalue.csv
│       ├── drug_network_proximity_results.csv
│       ├── key_proteins.txt
│       └── network_edges_key_proteins.txt
├── knowledge_graphs/
│   ├── NASH_shortest_paths.csv
│   ├── Hepatic_steatosis_shortest_paths.csv
│   ├── MASH_subgraph_nodes.csv
│   └── MASH_subgraph_drugs.csv
└── single_omics/
    ├── Coassolo_AUC_scores_target_genes_full_stats.csv
    ├── Wang_HUman_Hepatocyte_gene_AUC_NASH_vs_CTRL.csv
    ├── SU_Hepatocyte_gene_AUC_direction_stats_balanced_new.csv
    ├── active_drugs.xlsx
    ├── GSE210501_Mouse_scRNAseq.csv
    ├── GSE212837_Human_snRNAseq.csv
    └── GSE189600_Human_snRNAseq.csv
```

## Two Methods to Load Data

### Method 1: Direct Folder Access (Recommended for Streamlit Cloud)

The app can access your folder directly if it's shared publicly. No additional setup needed!

**Requirements:**
- Folder must be shared with "Anyone with the link" access
- Files must be in the exact folder structure above

### Method 2: File ID Mapping (More Reliable)

For production, extract individual file IDs:

1. **Open your Google Drive folder**
2. **For each file, get its ID:**
   - Right-click file → "Get link"
   - URL: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
   - Extract `FILE_ID`

3. **Update `google_drive_loader.py`:**
   ```python
   FILE_IDS = {
       "datExpr_processed.csv": "1a2b3c4d5e6f7g8h9i0j",
       "datTraits_processed.csv": "2b3c4d5e6f7g8h9i0j1k",
       # ... add all file IDs
   }
   ```

4. **Uncomment in `streamlit_app.py`:**
   ```python
   from google_drive_loader import load_all_data
   data = load_all_data()
   ```

### Method 3: Google Drive API (Production)

For enterprise deployment:

1. **Create Google Cloud Project:**
   - Go to https://console.cloud.google.com
   - Create new project
   - Enable Google Drive API

2. **Create Service Account:**
   - Create service account
   - Download JSON credentials
   - Share your folder with service account email

3. **Install libraries:**
   ```bash
   pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
   ```

4. **Use in app:**
   ```python
   from google.oauth2 import service_account
   from googleapiclient.discovery import build
   
   credentials = service_account.Credentials.from_service_account_file(
       'credentials.json',
       scopes=['https://www.googleapis.com/auth/drive']
   )
   service = build('drive', 'v3', credentials=credentials)
   ```

## Deployment on Streamlit Cloud

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Add Google Drive integration"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Create new app
3. Select your repository and branch
4. Streamlit Cloud will automatically install dependencies

### Step 3: Share Your Google Drive Folder

Make sure your `meta-liver-data` folder is shared:
1. Right-click folder → "Share"
2. Change to "Anyone with the link"
3. Copy link and keep it safe

## Troubleshooting

### "Permission denied" error
- Ensure folder is shared with "Anyone with the link"
- Check folder link is correct
- Verify files are in correct subfolders

### "File not found" error
- Check file names match exactly (case-sensitive)
- Verify folder structure matches above
- Ensure no extra spaces in file names

### Slow loading
- Large files take time to download
- Use caching in Streamlit (already implemented)
- Consider splitting large files

### Data not updating
- Streamlit caches data for 24 hours
- To refresh: Clear cache in Streamlit Cloud settings
- Or modify file names slightly to force reload

## Security Notes

⚠️ **Important:**
- Never commit credentials or API keys to GitHub
- Use Streamlit Cloud Secrets for sensitive data
- Keep folder link private if data is sensitive
- Use service account for production (not personal account)

## Support

For issues:
1. Check Google Drive folder is accessible
2. Verify file structure matches above
3. Check Streamlit Cloud logs
4. Review `google_drive_loader.py` for file IDs

## Next Steps

1. ✅ Folder structure created
2. ✅ Files uploaded to Google Drive
3. ✅ App configured for Google Drive
4. 📤 Push to GitHub
5. 🚀 Deploy on Streamlit Cloud
6. 🔗 Share permanent URL

Ready to deploy? Follow the deployment steps in DEPLOYMENT.md!
