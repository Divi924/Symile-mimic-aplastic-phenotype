# Symile-MIMIC Aplastic-like Phenotype Clustering Pipeline

A reproducible Python pipeline for admission-level clustering of haematological indices in a Symile-MIMIC/MIMIC-derived cohort, with optional ICD-coded cytopenia validation and SHAP-based explainability.

> **Research-use and interpretation notice:** This pipeline identifies an *aplastic-like / multi-lineage cytopenia phenotype*. The blood-count proxy implemented here is **not a clinical diagnosis of aplastic anaemia and is not the complete Camitta severe aplastic anaemia definition**. The available dataset does not contain the marrow-cellularity and reticulocyte data needed to apply the complete criteria. Results require clinical and methodological review before publication or use.

## Repository contents

```text
symile-mimic-aplastic-phenotype/
├── README.md
├── CITATION.cff
├── requirements.txt
├── LICENSE
├── .gitignore
└── scripts/
    └── symile_mimic_pipeline.py
```

## What the pipeline does

The script includes the following workflow:

1. Loads the input CSV and maps MIMIC item IDs to laboratory feature names.
2. Checks the unit of analysis and handles repeated admission rows using the documented admission-level aggregation fallback.
3. Reports missingness for the core haematological variables and related cell-count fields.
4. Combines direct and derived absolute neutrophil count (ANC) where available and reports their agreement.
5. Creates explicitly named `Modified_SAA_Phenotype` and `Modified_vSAA_Phenotype` peripheral-blood-only proxy labels.
6. Standardises nine core CBC features and performs K-means clustering, with a combined K-selection summary.
7. Produces cluster profiles, correlation and PCA summaries, bootstrap stability, gap-statistic results, and a missing-data sensitivity analysis.
8. Runs Kruskal–Wallis and Dunn post-hoc comparisons.
9. Evaluates cluster concordance with the modified rule-based label.
10. Optionally validates clusters against admission-level ICD-9/ICD-10 cytopenia categories.
11. Optionally trains Random Forest models for independent ICD-coded targets and generates SHAP explanations and sensitivity analyses.

The chosen cluster count is set to `K=3` in the script. The code reports several K-selection metrics; the chosen value should be described transparently rather than implying that all metrics independently select the same K.

## Requirements

- Python 3.10 or newer is recommended.
- Input data must be available to you under the applicable data-use agreement.
- Dependencies are listed in `requirements.txt`.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Create and activate a virtual environment (recommended).

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Input files

The script does **not** include clinical data. Obtain and prepare data independently and only use it in accordance with the applicable access agreement.

### Required

- `symile_mimic_data.csv` — admission-level extraction containing `hadm_id` and the laboratory columns expected by the script. The pipeline maps item-ID column names using the `ITEMID_MAP` dictionary in the script. It expects one row per hospital admission; if multiple rows are present, it applies the documented median aggregation fallback.

### Optional ICD validation inputs

- `diagnoses_icd.csv` — per-admission diagnosis table containing at least `subject_id`, `hadm_id`, `icd_code`, and `icd_version` (or compatible column variants described in the script).
- `d_icd_diagnoses.csv` — ICD code dictionary used for a traceability check.

Do not commit the input CSVs, patient-level data, or derived patient-level output files to this repository. The `.gitignore` excludes common CSV data files and output folders by default.

## Run the pipeline

Place the input file in the project root, then run:

```bash
python scripts/symile_mimic_pipeline.py \
  --input symile_mimic_data.csv \
  --outdir outputs
```

To include ICD validation, provide both optional files:

```bash
python scripts/symile_mimic_pipeline.py \
  --input symile_mimic_data.csv \
  --diagnoses_icd diagnoses_icd.csv \
  --d_icd_diagnoses d_icd_diagnoses.csv \
  --outdir outputs
```

On Google Colab, upload the script and run it in a notebook. If the expected input file is absent, the script can prompt for file upload. ICD validation can be skipped by not supplying the diagnosis file. The script can install `scikit-posthocs` and `shap` at runtime if they are missing; installing dependencies in advance is recommended.

## Outputs

The pipeline writes reports and figures under `outputs/`, including:

- unit-of-analysis and missingness reports;
- ANC agreement and modified-label comparison reports;
- K-selection summary and justification;
- complete-case and median-imputation sensitivity results;
- clustered admission-level data and cluster profiles;
- Spearman correlation and PCA explained-variance tables;
- K-means stability, gap statistic, Kruskal–Wallis, and Dunn post-hoc results;
- cluster-vs-modified-label screening validity;
- if ICD inputs are supplied, ICD matching/validation tables and figures;
- if ICD inputs are supplied, SHAP performance, feature-importance, and explanatory figures.

Exact filenames are defined in the script and may change between releases. Treat all outputs as research analyses, not clinical decision-support results.

## Reproducibility and limitations

- Random seeds are defined in the script (`RANDOM_STATE = 42`).
- The analysis is unsupervised for phenotype discovery; optional Random Forest/SHAP analysis predicts independent ICD-coded targets.
- The modified SAA/vSAA labels are peripheral-blood-only proxies because marrow cellularity and reticulocyte information are unavailable in this dataset.
- ICD codes depend on coding practices and the selected code list; inspect `CYTOPENIA_ICD_CODES` and the generated traceability report.
- The script uses a single stratified train/test split for its optional SHAP-associated classification analysis; performance estimates may vary across splits and should not be interpreted as definitive external validation.
- K-means cluster labels are arbitrary identifiers. Interpret clusters from their feature profiles rather than the numeric cluster ID alone.
- Validate the code, assumptions, feature units, ICD code list, and generated results against the specific data extract before relying on results.

## Citation

Please cite the versioned Zenodo record for the exact release used in your work. Complete `CITATION.cff` with the correct author names and metadata before publishing a release. Once Zenodo creates a DOI, add it to the citation metadata and this README.

## License

The included MIT license is a template. Before publishing, confirm that you have the authority to license this code and replace the copyright-holder placeholder with the correct author or institution. If institutional or co-author approval is required, obtain it before making the repository public.
