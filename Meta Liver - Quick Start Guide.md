# Meta Liver - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup (One-time)
```bash
cd /home/ubuntu/meta_liver
bash setup.sh
```

### Step 2: Activate Environment
```bash
source venv/bin/activate
```

### Step 3: Run the App
```bash
streamlit run streamlit_app.py
```

The app will open at: **http://localhost:8501**

---

## 📊 What You Can Do

### 🏠 Dashboard
- View key statistics (201 samples, 14,131 genes, 19 modules, 135 drugs)
- See module-trait correlations
- Quick overview of all features

### 📁 Module Explorer
- Browse 19 WGCNA co-expression modules
- View genes in each module
- See functional enrichment (GO terms, CORUM complexes)

### 🔍 Gene Analysis
- Search genes by ID or name
- View expression profiles
- Check AUC discrimination scores

### 💊 Drug Discovery
- Browse 135 approved drugs
- Filter by z-score and distance
- Explore drug targets and mechanisms

### 🏷️ Enrichment Analysis
- Functional annotations for each module
- GO biological processes
- CORUM protein complexes
- Adjustable p-value filters

### 🕸️ PPI Network
- Protein-protein interactions (773,143 edges)
- Top interactions by frequency
- Network statistics

### 📋 Data Tables
- Access all raw data
- Sort and filter
- Export results

---

## 🎯 Common Tasks

### Search for a Gene
1. Go to **Gene Analysis** tab
2. Enter ENSG ID (e.g., `ENSG00000000003`)
3. Click **Search**
4. View expression profile and AUC score

### Find Drugs for a Module
1. Go to **Module Explorer**
2. Select a module (e.g., `brown`)
3. View enriched pathways
4. Go to **Drug Discovery** to find targeting drugs

### Explore Protein Interactions
1. Go to **PPI Network** tab
2. View top interactions
3. Check interaction sources (databases)
4. See network statistics

### Download Data
1. Go to **Data Tables** tab
2. Select dataset (AUC Scores, Drugs, etc.)
3. Use browser's download option

---

## 💻 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8 | 3.10+ |
| RAM | 2 GB | 4+ GB |
| Disk Space | 500 MB | 1+ GB |
| Browser | Modern | Chrome/Firefox |

---

## 🔧 Troubleshooting

### "Port 8501 already in use"
```bash
# Use different port
streamlit run streamlit_app.py --server.port=8502
```

### "Module not found" error
```bash
# Reinstall packages
pip install -r requirements.txt
```

### App runs slowly
- Close other applications
- Reduce number of displayed rows
- Use filters to limit data

### Data not loading
- Check data files in `/home/ubuntu/upload/`
- Verify file permissions
- Check logs: `tail -50 streamlit.log`

---

## 📱 Access Remotely

If running on a remote server:

```bash
# On remote server
streamlit run streamlit_app.py --server.address=0.0.0.0

# Then access from local machine
# http://remote_server_ip:8501
```

---

## 📚 Learn More

- Full documentation: See `README.md`
- Data format: Check data files in `/home/ubuntu/upload/`
- Code: Review `streamlit_app.py`

---

## ✨ Tips & Tricks

1. **Bookmark favorite modules** - Use browser bookmarks for quick access
2. **Export data** - Use Data Tables tab to export CSV files
3. **Filter enrichment** - Adjust p-value threshold to focus on significant terms
4. **Search genes** - Use partial names or full ENSG IDs
5. **Compare modules** - Open multiple browser tabs for side-by-side comparison

---

**Happy exploring! 🔬**
