# Deployment Instructions

## Files to Update in Your Windows Repository

Copy these files to `C:\Users\gonli\Documents\meta-liver\`:

### 1. streamlit_app.py
- **Location:** `/home/ubuntu/meta_liver/streamlit_app.py`
- **Changes:** Updated KG section with three tabs (Genes/Proteins, Drugs, Diseases)
- **Note:** This is the main app file

### 2. kg_analysis.py
- **Location:** `/home/ubuntu/meta_liver/kg_analysis.py`
- **Changes:** Added three new functions:
  - `get_cluster_genes(cluster_id, kg_data)` - Returns genes/proteins in cluster
  - `get_cluster_drugs(cluster_id, kg_data)` - Returns drugs in cluster
  - `get_cluster_diseases(cluster_id, kg_data)` - Returns diseases in cluster

### 3. robust_data_loader.py
- **Location:** `/home/ubuntu/meta_liver/robust_data_loader.py`
- **Status:** No changes needed (already in repo)

### 4. single_omics_analysis.py
- **Location:** `/home/ubuntu/meta_liver/single_omics_analysis.py`
- **Status:** No changes needed (already in repo)

## Deployment Steps

1. Copy the updated files to your Windows repo
2. In PowerShell/Git Bash:
   ```bash
   cd C:\Users\gonli\Documents\meta-liver
   git add streamlit_app.py kg_analysis.py
   git commit -m "Update KG section with three tabs (genes, drugs, diseases)"
   git push origin main
   ```
3. Streamlit Cloud will auto-deploy within 1-2 minutes
4. Verify at: https://meta-liver-gehadyoussef.streamlit.app

## Testing Checklist

- [ ] Search for a gene (e.g., "APOB")
- [ ] Verify three tabs appear in "Nodes in Cluster" section
- [ ] Check that each tab shows the correct node types
- [ ] Verify tables are sorted by PageRank (descending)
- [ ] Test with a gene not in MASH subgraph (should show warning)

## Next Phase

Once edge list is provided, upgrade to show:
- True shortest paths between gene and drugs
- Ego networks (1-hop and 2-hop neighborhoods)
- Edge weights and interaction types
