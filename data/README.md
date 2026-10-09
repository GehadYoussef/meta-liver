# Data

All files were copied from `Shiny_app_data_dec_2025/`. `MANIFEST.csv` lists every file
with its original path and MD5 checksum. Files were renamed only to remove spaces, and
exact duplicates were dropped.

| Folder | Contents | Source |
|---|---|---|
| `raw/` | Raw single-cell data, **git-ignored** (~2.5 GB). `GSE189600_Xiao/` was copied from the original folder. The rest is fetched by `scripts/00_download_geo.R`. | see below |
| `single_cell/legacy/` | Per-gene AUC tables from the original scripts (pre-fix). Read by the app, which fixes the AUC/direction coding and hides invalid p-values. | `single_omics/`, `new_scripts/` |
| `single_cell/results/` | Per-gene tables from the fixed pipeline (`scripts/01–05`). These replace the legacy tables in the app. | generated |
| `single_cell/sensitivity/` | Sensitivity analyses: Wang without FACS-sorted captures (`scripts/04b`), and Xiao and Wang with DecontX ambient RNA correction (`scripts/04c`). Summarised in `consistency/sensitivity_agreement.csv`. | generated |
| `single_cell/consistency/` | Direction consistency by gene, pairwise dataset agreement, agreement with bulk, direction QC, original-vs-fixed comparison (`scripts/11`). | generated |
| `single_cell/legacy_vs_v2_comparison.csv` | Per-dataset AUC correlation / significance counts, original vs fixed. | generated |
| `bulk/` | DESeq2 results for the human bulk liver RNA-seq cohort GSE135251 (Govaere et al. 2020): disease stage 1–4 vs 0, and NAFL / NASH F0–F4 vs control. | `BULK_datasets/` (identity confirmed against Meta Liver) |
| `bulk_cohorts/` | Per-cohort DEG tables for 4 contrasts (MASLD, early MASLD and MASH vs control, and MASH vs early MASLD): GSE126848 and GSE135251 (RNA-seq), GSE151158 (618-gene panel, MASLD vs control only). Input to the random-effects meta-analysis. | Meta Liver `meta-liver-data/Bulk_omics/`, via `scripts/09_import_metaliver_data.py` |
| `invitro/` | iHeps (iPSC-derived hepatocytes, lines 1b and 5a): DESeq2 vs untreated for OA+PA, + resistin/myostatin, + PBMC co-culture. Ensembl IDs, mapped to symbols by `gene_mapping.csv.gz`. Converted from parquet. | Meta Liver `meta-liver-data/stem_cell_model/`, same script |
| `drugs/` | `wgcna_active_drugs.csv`: drugs targeting genes in stage-associated WGCNA modules, with mechanism and indication. | Meta Liver, same script |
| `knowledge_graph/` | MASH knowledge-graph subgraph nodes with PageRank/betweenness/eigenvector scores, plus drugs on shortest paths to NASH and hepatic steatosis. `MASH_subgraph_drugs.csv` contains ~1M blank rows from the export, which the code drops. | Fatima, Supp Table 6 |
| `ppi/early_mafld_network/` | PPI centrality/RWR results, 158 key proteins, and drug–network proximity (z, d). Drug IDs are mostly DrugBank (1,814) plus some ChEMBL (96). | `PPI_networks/Early MAFLD Network/` |
| `external/metaliver/wgcna/` | WGCNA modules on the same bulk cohort (trait = stage), module enrichment. The 1.6 GB TOM and other raw `.rds` files were not copied. | `MetaLiver_source_code/inst/extdata/` |
| `external/metaliver/degs/` | In-vitro model DEG tables. **The numeric columns are corrupted** (decimal separators lost, e.g. baseMean = 112911823570072), so these files are not used. `invitro/` has intact copies. | same |
| `reference/` | Ensembl to symbol map, target gene list, and (optionally) MGI mouse–human orthologs. Built by `scripts/10_build_reference.R`. | generated |

## Target gene list

`reference/target_genes_mouse.txt` (621 genes) was recovered from the Coassolo
target-gene result table, because the original `my_genes_mus.txt` was not in the
project folder. It therefore lists only targets that were present in the Coassolo data.
Replace it with the original list if you have it.

## Raw single-cell data (verified against GEO)

| Folder | GEO | Study | Design (sample = donor/animal) |
|---|---|---|---|
| `raw/GSE189600_Xiao/` | GSE189600 | Xiao et al., human snRNA-seq | 3 healthy vs 3 NASH livers, 1 capture each |
| `raw/GSE166504_Su/` | GSE166504 | Su et al. 2021 (PMID 34755088), mouse scRNA-seq | Hepatocyte libraries: 3 mice at 15 weeks HFHFD (2 captures each) vs 6 chow mice (1 capture each). Chow ages are not recorded on GEO. |
| `raw/GSE210501_Coassolo/` | GSE210501 | Jung, Coassolo et al., mouse scRNA-seq | 1 chow vs 1 NASH-diet liver (6.5 weeks): no biological replication |
| `raw/GSE212837_Wang/` | GSE212837 | Wang et al. 2023, Friedman lab (PMID 36599008), snRNA-seq | Human: 3 control vs 9 NASH livers. NASH livers 1-5 have 2–3 captures each (18 captures). Mouse: 2 NASH livers (30 weeks), no controls. |

Run `Rscript scripts/00_download_geo.R` to re-fetch Su, Coassolo and Wang (about 2 GB).
