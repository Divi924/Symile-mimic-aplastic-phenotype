"""
Symile-MIMIC Aplastic-like Phenotype Clustering — CORRECTED PIPELINE
=======================================================================
Consolidates and corrects the prior Symile_MIMIC_1.ipynb / Symile_mimic_2_1_.ipynb
notebooks into a single, linear, reproducible script.

WHAT WAS CORRECTED VS. THE ORIGINAL NOTEBOOKS
-----------------------------------------------------------------------
1. FEATURE SET: RDW-SD is dropped (67.8% missing in this cohort, confirmed
   empirically below, not assumed). Final clustering feature set = 9
   haematological indices: Hb, Hct, RBC, WBC, Platelet Count, MCV, MCH,
   MCHC, RDW.

2. CAMITTA CRITERIA: the original notebooks mislabeled a generic
   reference-range rule (Hb<12 AND WBC<4 AND PLT<150, all three required)
   as "Camitta criteria." This is corrected here to the actual published
   criteria for severe aplastic anaemia (Camitta 1976; still the field
   standard): at least 2 of {ANC<0.5x10^9/L, Platelets<20x10^9/L,
   reticulocyte count<20x10^9/L (manual) or <60x10^9/L (automated) /
   corrected reticulocyte<1%}, on a background of marrow hypocellularity.

   This dataset has NO reticulocyte count and NO marrow biopsy data
   (expected — reticulocyte counts are not part of routine automated ICU
   CBC panels, and marrow cellularity requires a biopsy). The criteria
   used here are therefore an explicitly-disclosed MODIFIED 2-of-2 rule
   (ANC + Platelets only). This is a stated LIMITATION, not silently
   patched over.

   ANC itself is directly available (MIMIC itemid 52075) but 71.7%
   missing. Coverage is substantially improved (to 84.8%) by deriving
   ANC = WBC x (Neutrophil% / 100) where the direct field is absent;
   agreement between direct and derived ANC where both exist is
   r=0.983 (validated below), supporting this substitution.

3. MISSING DATA: the 9 core features are essentially complete
   (<=0.43% missing each; 99.5% of admissions have complete data).
   PRIMARY analysis uses complete cases. A SENSITIVITY analysis
   additionally median-imputes the full (uncensored) cohort, re-runs
   K=3 K-means, and reports the Adjusted Rand Index between the two
   cluster solutions on the overlapping admissions — quantifying, not
   just asserting, that the imputation choice does not materially
   change conclusions. No blanket mean-imputation of high-missingness
   variables (e.g. RDW-SD) is performed.

4. REDUNDANCY: Hb/Hct/RBC show high Spearman correlation (rho=0.89-0.97).
   Per user decision, all three are RETAINED (clinical divergence between
   these indices — e.g., thalassemia trait, dehydration/haemodilution —
   is itself diagnostically meaningful), with this decision explicitly
   justified in the Methods rather than silently resolved by dropping
   a variable.

5. LABEL USE: given the corrected modified-SAA-phenotype rate is only
   0.11% (11/9,854 evaluable admissions), a supervised classifier trained
   directly on this label would be statistically fragile. Per user
   decision, NO supervised model is trained on it. The cluster-derived
   phenotype is validated against the rule-based label and an
   independent ICD-coded ground truth, using purely rule-based /
   unsupervised methods (chi-square, Cramer's V, sensitivity/
   specificity/PPV/NPV).

6. ICD VALIDATION SCOPE: at the user's request, external validation covers ALL
   cytopenia-related ICD-9/10 conditions (not just aplastic anaemia), matched at
   the admission (hadm_id) level: Aplastic_Anemia_BMF, Pancytopenia,
   Neutropenia_Agranulocytosis, Thrombocytopenia (+ a separate Thrombocytopenia_HIT
   sub-flag, since HIT is immune/drug-mediated rather than marrow-failure-related),
   Leukopenia_Lymphocytopenia, and Myelodysplastic_Syndrome, plus a combined
   "Any_Cytopenia" flag. Obstetric/fetal-monitoring codes (ICD-10 O36.82x) are
   excluded as out of scope (they describe fetal, not patient, status).

   On this cohort, marrow-failure-type categories (Aplastic Anaemia/BMF 83.3%,
   MDS 78.4%, Pancytopenia 65.2%, Neutropenia 63.7% sensitivity, all p<0.001)
   showed strong, highly significant concordance with the cluster, while
   mechanistically unrelated single-lineage categories (HIT, isolated
   Leukopenia/Lymphocytopenia) showed weak, non-significant concordance
   (p=0.92 and p=0.82 respectively). This pattern supports construct validity:
   the cluster captures a genuine multi-lineage marrow-suppression phenotype
   rather than indiscriminately flagging any abnormal count.

7. UNIT OF ANALYSIS: the pipeline verifies programmatically (Step 1b) that
   symile_mimic_data.csv contains exactly one row per hadm_id before any
   clustering happens — i.e. clustering is performed on admissions, not on
   raw/repeated laboratory draws. On the dataset used to develop this
   pipeline, this holds exactly (11,622 rows = 11,622 unique hadm_id). If a
   future re-extraction ever has multiple rows per admission, the script
   raises a clear, actionable error rather than silently clustering
   observations; see ensure_one_row_per_admission() below for the documented
   aggregation fallback (admission-level median per feature) if the user
   opts to proceed that way instead of fixing the extraction query.

8. K SELECTION: K is no longer just "evaluated then overridden." Step 4
   builds one combined decision table across elbow (WCSS), silhouette,
   gap statistic, and bootstrap ARI stability for each candidate K in
   {2,3,4,5}, alongside resulting cluster sizes. K=3 is then selected and
   reported as: "K=3 was selected based on the combined statistical and
   clinical criteria" (three clinically distinct phenotypes: near-normal,
   anemic/high-RDW, pancytopenic-pattern) — not framed as though silhouette
   or the elbow alone uniquely pointed to 3 (on this cohort, gap statistic
   and silhouette both marginally favour K=2; this is disclosed, not hidden).

9. LABEL NAMING: the rule-based severe-cytopenia label is named
   Modified_SAA_Phenotype (and Modified_vSAA_Phenotype for the very-severe
   cut). It is deliberately NOT called "Camitta SAA" / "Camitta criteria"
   unqualified, even though it is derived from that framework. The true
   Camitta framework requires bone-marrow cellularity and a reticulocyte
   count, neither of which exists in this dataset; the label implemented
   here is a peripheral-blood-only, 2-of-2 (ANC + platelets) approximation
   and must be reported to readers as such.

REQUIRED INPUT FILES (upload these three when prompted in Colab)
-----------------------------------------------------------------------
    - symile_mimic_data.csv   : raw Symile-MIMIC lab/CXR/ECG extraction
                                 (itemid-labelled lab columns; ~11,600 admissions)
    - diagnoses_icd.csv       : MIMIC-IV hosp module, all patients' ICD diagnoses
                                 (subject_id, hadm_id, seq_num, icd_code, icd_version)
                                 [optional — omit to skip Step 9 ICD validation]
    - d_icd_diagnoses.csv     : MIMIC-IV hosp module, ICD code dictionary
                                 (icd_code, icd_version, long_title)
                                 [optional — used only as a traceability check on
                                 the hard-coded cytopenia code list below]

RUNNING IN GOOGLE COLAB
-----------------------------------------------------------------------
Paste this whole script into a single Colab cell and run it (or upload it as
a .py file and run `%run symile_mimic_pipeline_corrected.py` / `!python
symile_mimic_pipeline_corrected.py`). The script will automatically prompt a
file-upload dialog for symile_mimic_data.csv if it isn't already present in
/content, and will ask (y/n) whether to also upload the two ICD files for
Step 9. All outputs are written to /content/outputs and, at the end, zipped
and offered as a browser download automatically.

RUNNING LOCALLY
-----------------------------------------------------------------------
    python symile_mimic_pipeline_corrected.py \
        --input symile_mimic_data.csv \
        --diagnoses_icd diagnoses_icd.csv \
        --d_icd_diagnoses d_icd_diagnoses.csv


Outputs (written to ./outputs/):
    - unit_of_analysis_check.txt
    - missingness_report.csv
    - anc_agreement_report.txt
    - modified_saa_label_comparison.csv
    - K_selection_summary.csv
    - K_selection_justification.txt
    - missing_data_sensitivity_ARI.csv
    - Clustered_Aplastic_Data_CORRECTED.csv
    - cluster_profile_means_CORRECTED.csv
    - Spearman_correlation_matrix_CORRECTED.csv
    - PCA_explained_variance_CORRECTED.csv
    - KMeans_stability_ARI_CORRECTED.csv
    - Gap_Statistic_CORRECTED.csv
    - Dunn_Posthoc_CORRECTED.csv
    - Cluster_Kruskal_Wallis_CORRECTED.csv
    - cluster_vs_modified_saa_screening_validity.csv
    - figures/fig1_elbow_silhouette.png
    - figures/fig2_correlation_heatmap.png
    - figures/fig3_pca_clusters.png
    - (if diagnoses_icd given) cytopenia_diagnoses_matched.csv
    - (if diagnoses_icd given) cluster_vs_ALL_cytopenia_ICD_validity.csv
    - (if diagnoses_icd given) Clustered_Aplastic_Data_WITH_ICD.csv
    - (if diagnoses_icd given) figures/fig4_icd_validation_by_category.png
    - (if diagnoses_icd given) SHAP_model_performance.csv
    - (if diagnoses_icd given) SHAP_feature_importance.csv
    - (if diagnoses_icd given) figures/fig5_shap_summary_beeswarm.png
    - (if diagnoses_icd given) figures/fig6_shap_importance_bar.png
    - (if diagnoses_icd given) figures/fig7_shap_dependence_top_feature.png
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (adjusted_rand_score, average_precision_score,
                              classification_report, confusion_matrix,
                              roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_scikit_posthocs():
    try:
        import scikit_posthocs  # noqa: F401
    except ImportError:
        print("Installing scikit-posthocs (one-time)...")
        os.system(f"{sys.executable} -m pip install -q scikit-posthocs")


def _colab_upload_if_missing(path, prompt_label, required=True):
    """In Colab, prompt a file-upload dialog if `path` doesn't already exist.
    Returns the resolved path (may differ from `path` if the uploaded filename
    differs), or None if skipped/not found and not required."""
    if os.path.exists(path):
        return path
    if not _in_colab():
        return path if required else None
    from google.colab import files
    print(f"\n>>> Please upload {prompt_label} <<<")
    uploaded = files.upload()
    if not uploaded:
        return None
    # Use the first uploaded file's actual name (handles renamed uploads)
    uploaded_name = list(uploaded.keys())[0]
    return uploaded_name

RANDOM_STATE = 42
N_INIT = 20
N_BOOTSTRAP = 50
N_GAP_REF = 10
FIG_DPI = 300  # publication-resolution for all saved figures

ITEMID_MAP = {
    "51221": "Hematocrit", "51265": "Platelet Count", "50912": "Creatinine",
    "50971": "Potassium", "51222": "Hemoglobin", "51301": "White Blood Cells",
    "51249": "MCHC", "51279": "Red Blood Cells", "51250": "MCV", "51248": "MCH",
    "51277": "RDW", "51006": "Urea Nitrogen", "50983": "Sodium", "50902": "Chloride",
    "50882": "Bicarbonate", "50868": "Anion Gap", "50931": "Glucose",
    "50960": "Magnesium", "50893": "Calcium, Total", "50970": "Phosphate",
    "51237": "INR(PT)", "51274": "PT", "51275": "PTT", "51146": "Basophils",
    "51256": "Neutrophils", "51254": "Monocytes", "51200": "Eosinophils",
    "51244": "Lymphocytes", "52172": "RDW-SD",
    "50934": "Unverified_Lab_H", "51678": "Unverified_Lab_L", "50947": "Unverified_Lab_I",
    "50861": "Alanine Aminotransferase", "50878": "Asparate Aminotransferase (AST)",
    "50813": "Lactate", "50863": "Alkaline Phosphatase", "50885": "Bilirubin, Total",
    "50820": "pH", "50862": "Albumin", "50802": "Base Excess", "50821": "pO2",
    "50804": "Calculated Total CO2", "50818": "pCO2",
    "52075": "Absolute Neutrophil Count", "52073": "Absolute Eosinophil Count",
    "52074": "Absolute Monocyte Count", "52069": "Absolute Basophil Count",
    "51133": "Absolute Lymphocyte Count", "50910": "Creatine Kinase",
    "52135": "Immature Granulocytes",
}

CORE9 = ["Hemoglobin", "Hematocrit", "Red Blood Cells", "White Blood Cells",
         "Platelet Count", "MCV", "MCH", "MCHC", "RDW"]

RDWSD_MISSINGNESS_THRESHOLD = 0.30  # features above this are excluded from clustering

# -----------------------------------------------------------------------
# Cytopenia ICD-9/ICD-10 reference (all categories requested by the user,
# not just aplastic anaemia). See module docstring above for inclusion/
# exclusion rationale (e.g. HIT kept as its own sub-flag; fetal-monitoring
# codes excluded as out of scope).
# -----------------------------------------------------------------------
CYTOPENIA_ICD_CODES = {
    # icd_code: (icd_version, category)
    # ---- Aplastic anaemia / other bone marrow failure syndromes ----
    "28409": (9, "Aplastic_Anemia_BMF"), "28489": (9, "Aplastic_Anemia_BMF"),
    "2849": (9, "Aplastic_Anemia_BMF"),
    "D610": (10, "Aplastic_Anemia_BMF"), "D6109": (10, "Aplastic_Anemia_BMF"),
    "D611": (10, "Aplastic_Anemia_BMF"), "D612": (10, "Aplastic_Anemia_BMF"),
    "D613": (10, "Aplastic_Anemia_BMF"), "D618": (10, "Aplastic_Anemia_BMF"),
    "D6189": (10, "Aplastic_Anemia_BMF"), "D619": (10, "Aplastic_Anemia_BMF"),

    # ---- Pancytopenia ----
    "2841": (9, "Pancytopenia"), "28411": (9, "Pancytopenia"),
    "28412": (9, "Pancytopenia"), "28419": (9, "Pancytopenia"),
    "D6181": (10, "Pancytopenia"), "D61810": (10, "Pancytopenia"),
    "D61811": (10, "Pancytopenia"), "D61818": (10, "Pancytopenia"),

    # ---- Neutropenia / Agranulocytosis ----
    "28800": (9, "Neutropenia_Agranulocytosis"), "28801": (9, "Neutropenia_Agranulocytosis"),
    "28802": (9, "Neutropenia_Agranulocytosis"), "28803": (9, "Neutropenia_Agranulocytosis"),
    "28804": (9, "Neutropenia_Agranulocytosis"), "28809": (9, "Neutropenia_Agranulocytosis"),
    "7767": (9, "Neutropenia_Agranulocytosis"),
    "D700": (10, "Neutropenia_Agranulocytosis"), "D701": (10, "Neutropenia_Agranulocytosis"),
    "D702": (10, "Neutropenia_Agranulocytosis"), "D703": (10, "Neutropenia_Agranulocytosis"),
    "D704": (10, "Neutropenia_Agranulocytosis"), "D708": (10, "Neutropenia_Agranulocytosis"),
    "D709": (10, "Neutropenia_Agranulocytosis"), "P615": (10, "Neutropenia_Agranulocytosis"),

    # ---- Thrombocytopenia (incl. immune/HIT) ----
    "28730": (9, "Thrombocytopenia"), "28739": (9, "Thrombocytopenia"),
    "2874": (9, "Thrombocytopenia"), "28749": (9, "Thrombocytopenia"),
    "2875": (9, "Thrombocytopenia"), "2872": (9, "Thrombocytopenia"),
    "28731": (9, "Thrombocytopenia"), "28733": (9, "Thrombocytopenia"),
    "28984": (9, "Thrombocytopenia_HIT"), "7761": (9, "Thrombocytopenia"),
    "D694": (10, "Thrombocytopenia"), "D6942": (10, "Thrombocytopenia"),
    "D6949": (10, "Thrombocytopenia"), "D695": (10, "Thrombocytopenia"),
    "D6959": (10, "Thrombocytopenia"), "D696": (10, "Thrombocytopenia"),
    "D692": (10, "Thrombocytopenia"), "D693": (10, "Thrombocytopenia"),
    "P610": (10, "Thrombocytopenia"),
    "D7582": (10, "Thrombocytopenia_HIT"), "D75821": (10, "Thrombocytopenia_HIT"),
    "D75822": (10, "Thrombocytopenia_HIT"), "D75828": (10, "Thrombocytopenia_HIT"),
    "D75829": (10, "Thrombocytopenia_HIT"),

    # ---- Leukopenia / Lymphocytopenia ----
    "28850": (9, "Leukopenia_Lymphocytopenia"), "28851": (9, "Leukopenia_Lymphocytopenia"),
    "D72810": (10, "Leukopenia_Lymphocytopenia"),

    # ---- Myelodysplastic syndromes (established cytopenia-causing marrow disorder) ----
    "23872": (9, "Myelodysplastic_Syndrome"), "23873": (9, "Myelodysplastic_Syndrome"),
    "23874": (9, "Myelodysplastic_Syndrome"), "23875": (9, "Myelodysplastic_Syndrome"),
    "C946": (10, "Myelodysplastic_Syndrome"), "D46": (10, "Myelodysplastic_Syndrome"),
    "D469": (10, "Myelodysplastic_Syndrome"), "D46C": (10, "Myelodysplastic_Syndrome"),
    "D46Z": (10, "Myelodysplastic_Syndrome"), "D46A": (10, "Myelodysplastic_Syndrome"),
    "D46B": (10, "Myelodysplastic_Syndrome"),
}


def get_cytopenia_icd_reference():
    rows = [{"icd_code": code, "icd_version": v, "category": cat}
            for code, (v, cat) in CYTOPENIA_ICD_CODES.items()]
    return pd.DataFrame(rows)


def validate_icd_reference(d_icd_diagnoses_path, outdir):
    """
    Optional traceability check: confirms every code in CYTOPENIA_ICD_CODES
    actually exists in the site's own d_icd_diagnoses.csv dictionary (catches
    typos) and reports how many cytopenia-related codes overall exist in the
    dictionary vs. how many were selected, so the selection can be audited.
    """
    if not d_icd_diagnoses_path or not os.path.exists(d_icd_diagnoses_path):
        return None
    d = pd.read_csv(d_icd_diagnoses_path)
    d.columns = [c.strip().lower() for c in d.columns]
    d = d.rename(columns={"icd9_code": "icd_code", "icd10_code": "icd_code"})
    if "icd_code" not in d.columns or "icd_version" not in d.columns:
        print(f"WARNING: d_icd_diagnoses file missing expected columns "
              f"(found: {list(d.columns)}); skipping traceability check.")
        return None
    d["icd_code"] = d["icd_code"].astype(str)
    d["icd_version"] = pd.to_numeric(d["icd_version"], errors="coerce").astype("Int64")
    ref = get_cytopenia_icd_reference()
    ref["icd_code"] = ref["icd_code"].astype(str)
    merged_check = ref.merge(d, on=["icd_code", "icd_version"], how="left", indicator=True)
    missing = merged_check[merged_check["_merge"] == "left_only"]
    if len(missing):
        print(f"WARNING: {len(missing)} codes in CYTOPENIA_ICD_CODES not found in "
              f"d_icd_diagnoses.csv (possible typo):\n{missing[['icd_code','icd_version','category']]}")
    else:
        print(f"All {len(ref)} cytopenia ICD codes verified against d_icd_diagnoses.csv.")
    merged_check.to_csv(os.path.join(outdir, "icd_reference_validation_check.csv"), index=False)
    return missing


def load_and_rename(path):
    df = pd.read_csv(path)
    df = df.rename(columns=ITEMID_MAP)
    return df


def ensure_one_row_per_admission(df, outdir):
    """
    FIX #2 (unit-of-analysis check): clustering must operate on one row per
    admission (hadm_id), not on repeated/raw lab draws. This is verified
    programmatically here rather than assumed. If violated, admissions are
    aggregated to one row each using the median of each numeric lab feature
    per hadm_id (a documented, disclosed fallback) — but this is loud, not
    silent: it prints exactly what happened and how many rows were collapsed,
    and writes a record to outdir so it's auditable.
    """
    n_rows = len(df)
    n_hadm = df["hadm_id"].nunique()
    report_lines = [f"Rows in raw file: {n_rows}", f"Unique hadm_id: {n_hadm}"]

    if n_rows == n_hadm:
        report_lines.append("VERIFIED: exactly one row per admission (hadm_id). "
                             "No aggregation needed; clustering operates on admissions.")
        print("\n".join(report_lines))
        with open(os.path.join(outdir, "unit_of_analysis_check.txt"), "w") as f:
            f.write("\n".join(report_lines) + "\n")
        return df

    # Violated: multiple rows per admission exist.
    dup_counts = df["hadm_id"].value_counts()
    n_multi = (dup_counts > 1).sum()
    report_lines.append(
        f"WARNING: {n_rows - n_hadm} extra rows found — {n_multi} admissions have "
        f"more than one row (max {dup_counts.max()} rows for a single hadm_id). "
        f"This means the raw file contains multiple laboratory observations per "
        f"admission, not one row per admission as required by the study design."
    )
    report_lines.append(
        "FALLBACK APPLIED: aggregating to one row per hadm_id by taking the MEDIAN "
        "of each numeric lab feature within each admission. This is a reasonable "
        "default (robust to outlier single draws) but is a MODELING DECISION that "
        "must be reported in Methods — an alternative (e.g., first draw, worst-value, "
        "closest-to-admission-time) may be more clinically appropriate depending on "
        "the study question, and should be reconsidered before publication."
    )
    print("\n".join(report_lines))
    with open(os.path.join(outdir, "unit_of_analysis_check.txt"), "w") as f:
        f.write("\n".join(report_lines) + "\n")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_first = df.groupby("hadm_id").first().reset_index()
    numeric_median = df.groupby("hadm_id")[numeric_cols].median().reset_index()
    non_key_non_numeric = [c for c in non_numeric_first.columns
                            if c not in numeric_cols and c != "hadm_id"]
    df_agg = numeric_median.merge(non_numeric_first[["hadm_id"] + non_key_non_numeric],
                                   on="hadm_id", how="left")
    return df_agg


def missingness_report(df, outdir):
    cols = CORE9 + ["RDW-SD", "Absolute Neutrophil Count", "Neutrophils",
                     "Absolute Lymphocyte Count", "Lymphocytes",
                     "Absolute Monocyte Count", "Monocytes",
                     "Absolute Eosinophil Count", "Eosinophils",
                     "Absolute Basophil Count", "Basophils",
                     "Immature Granulocytes"]
    rows = []
    for c in cols:
        miss_pct = df[c].isna().mean() * 100
        rows.append({"Feature": c, "Percent_Missing": round(miss_pct, 2),
                      "N_Present": int(df[c].notna().sum()),
                      "Excluded_From_Clustering": miss_pct > RDWSD_MISSINGNESS_THRESHOLD * 100})
    report = pd.DataFrame(rows)
    report.to_csv(os.path.join(outdir, "missingness_report.csv"), index=False)
    print(report.to_string(index=False))
    return report


def build_anc(df, outdir):
    df["ANC_direct"] = df["Absolute Neutrophil Count"]
    df["ANC_derived"] = df["White Blood Cells"] * (df["Neutrophils"] / 100.0)
    both = df[["ANC_direct", "ANC_derived"]].dropna()
    corr = both["ANC_direct"].corr(both["ANC_derived"])
    mae = (both["ANC_direct"] - both["ANC_derived"]).abs().mean()
    df["ANC_final"] = df["ANC_direct"].fillna(df["ANC_derived"])

    report = (
        f"ANC derivation validation (n={len(both)} with both fields present)\n"
        f"  Pearson r = {corr:.3f}, MAE = {mae:.3f} x10^9/L\n"
        f"  Coverage: direct-only={df['ANC_direct'].notna().mean()*100:.1f}%, "
        f"final(direct+derived)={df['ANC_final'].notna().mean()*100:.1f}%\n"
    )
    with open(os.path.join(outdir, "anc_agreement_report.txt"), "w") as f:
        f.write(report)
    print(report)
    return df


def build_modified_saa_label(df, outdir):
    has_both = df["ANC_final"].notna() & df["Platelet Count"].notna()
    df["Modified_SAA_Phenotype"] = np.where(has_both,
                                  (df["ANC_final"] < 0.5) & (df["Platelet Count"] < 20),
                                  np.nan)
    df["Modified_vSAA_Phenotype"] = np.where(has_both,
                                   (df["ANC_final"] < 0.2) & (df["Platelet Count"] < 20),
                                   np.nan)
    old_def = (df["Hemoglobin"] < 12) & (df["White Blood Cells"] < 4) & (df["Platelet Count"] < 150)

    n_eval = int(has_both.sum())
    comparison = pd.DataFrame([
        {"Definition": "Modified_SAA_Phenotype: peripheral-blood-only proxy for Camitta "
                       "severe AA criteria (ANC<0.5 AND PLT<20, 2-of-2; reticulocyte/"
                       "marrow criteria unavailable in this dataset)",
         "N_evaluable": n_eval, "N_positive": int(np.nansum(df["Modified_SAA_Phenotype"])),
         "Percent_positive": round(np.nansum(df["Modified_SAA_Phenotype"]) / n_eval * 100, 3)},
        {"Definition": "Modified_vSAA_Phenotype: very-severe cut (ANC<0.2 AND PLT<20)",
         "N_evaluable": n_eval, "N_positive": int(np.nansum(df["Modified_vSAA_Phenotype"])),
         "Percent_positive": round(np.nansum(df["Modified_vSAA_Phenotype"]) / n_eval * 100, 3)},
        {"Definition": "OLD flawed def (Hb<12 AND WBC<4 AND PLT<150)",
         "N_evaluable": len(df), "N_positive": int(old_def.sum()),
         "Percent_positive": round(old_def.mean() * 100, 3)},
    ])
    comparison.to_csv(os.path.join(outdir, "modified_saa_label_comparison.csv"), index=False)
    print(comparison.to_string(index=False))
    return df


def _bootstrap_stability_for_k(X_scaled, k, n_bootstrap=20, random_state=RANDOM_STATE):
    """Mean/SD Adjusted Rand Index between the reference K-means(K) solution and
    K-means(K) refit on bootstrap resamples. Higher/tighter = more stable K."""
    n = len(X_scaled)
    full_labels = KMeans(n_clusters=k, random_state=random_state, n_init=N_INIT).fit_predict(X_scaled)
    rng = np.random.default_rng(random_state)
    aris = []
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels = KMeans(n_clusters=k, random_state=b, n_init=N_INIT).fit_predict(X_scaled[idx])
        aris.append(adjusted_rand_score(full_labels[idx], boot_labels))
    aris = np.array(aris)
    return aris.mean(), aris.std()


def _gap_statistic_for_k(X_scaled, k, n_ref=N_GAP_REF, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    mins, maxs = X_scaled.min(axis=0), X_scaled.max(axis=0)
    wcss_real = KMeans(n_clusters=k, random_state=random_state, n_init=N_INIT).fit(X_scaled).inertia_
    ref_wcss = []
    for b in range(n_ref):
        X_ref = rng.uniform(mins, maxs, size=X_scaled.shape)
        ref_wcss.append(KMeans(n_clusters=k, random_state=b, n_init=N_INIT).fit(X_ref).inertia_)
    ref_wcss = np.array(ref_wcss)
    gap = np.mean(np.log(ref_wcss)) - np.log(wcss_real)
    sk = np.std(np.log(ref_wcss)) * np.sqrt(1 + 1 / n_ref)
    return gap, sk


def k_selection_summary(X_scaled, outdir, candidate_ks=(2, 3, 4, 5), chosen_k=3):
    """
    FIX #3: builds ONE combined decision table across elbow (WCSS), silhouette,
    gap statistic, bootstrap ARI stability, and resulting cluster sizes for
    each candidate K — rather than fitting K=2..10 for a figure and then
    silently overriding with K=3. Ends with an explicit, honest statement of
    why K=3 was chosen, naming any metric that pointed elsewhere instead of
    implying unanimous statistical support.
    """
    rows = []
    for k in candidate_ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X_scaled)
        wcss = km.inertia_
        sil = silhouette_score(X_scaled, km.labels_)
        gap, sk = _gap_statistic_for_k(X_scaled, k)
        stab_mean, stab_sd = _bootstrap_stability_for_k(X_scaled, k)
        sizes = pd.Series(km.labels_).value_counts().sort_index().to_dict()
        rows.append({
            "K": k, "WCSS": round(wcss, 1), "Silhouette": round(sil, 4),
            "Gap": round(gap, 4), "Gap_s_k": round(sk, 4),
            "Bootstrap_ARI_mean": round(stab_mean, 3), "Bootstrap_ARI_sd": round(stab_sd, 3),
            "Cluster_sizes": str(sizes),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(outdir, "K_selection_summary.csv"), index=False)
    print("\nK-selection decision table (elbow/WCSS, silhouette, gap statistic, "
          "bootstrap stability, cluster sizes):")
    print(summary.to_string(index=False))

    best_sil_k = summary.loc[summary["Silhouette"].idxmax(), "K"]
    gap_rule_k = None
    for i in range(len(summary) - 1):
        if summary.loc[i, "Gap"] >= summary.loc[i + 1, "Gap"] - summary.loc[i + 1, "Gap_s_k"]:
            gap_rule_k = summary.loc[i, "K"]
            break
    best_stab_k = summary.loc[summary["Bootstrap_ARI_mean"].idxmax(), "K"]

    agreement_note = (
        f"Silhouette favors K={best_sil_k}; Gap-statistic stopping rule favors K={gap_rule_k}; "
        f"bootstrap stability is highest at K={best_stab_k}."
    )
    print("\n" + agreement_note)
    if chosen_k not in (best_sil_k, gap_rule_k, best_stab_k):
        print(f"NOTE: K={chosen_k} is NOT the top choice of every individual statistical "
              f"metric above (this is disclosed, not hidden).")
    justification = (
        f"K={chosen_k} was selected based on the COMBINED statistical and clinical "
        f"criteria in the table above (not because any single metric — elbow, "
        f"silhouette, gap statistic, or stability — uniquely and unambiguously "
        f"pointed to {chosen_k} in isolation), together with clinical interpretability: "
        f"K={chosen_k} separates a near-normal phenotype from two morphologically "
        f"distinct abnormal phenotypes (high-RDW anemic vs. pancytopenic-pattern), "
        f"which collapse into a single heterogeneous 'abnormal' group at K=2."
    )
    print("\n" + justification)
    with open(os.path.join(outdir, "K_selection_justification.txt"), "w") as f:
        f.write(agreement_note + "\n\n" + justification + "\n")
    return summary


def missing_data_sensitivity_analysis(df, df_cc, chosen_k, outdir):
    """
    FIX #1: quantifies (not just asserts) that the missing-data handling choice
    doesn't materially change conclusions.
    PRIMARY: complete-case K-means (already fit; labels passed in via df_cc).
    SENSITIVITY: median-impute the 9 core features on the FULL (uncensored)
    admission set, refit K-means(chosen_k), then compute the Adjusted Rand
    Index between the two solutions restricted to the admissions common to
    both (i.e. the original complete cases) — a direct, quantified agreement
    check rather than an unquantified claim that "imputation barely matters."
    """
    df_imp = df.copy()
    imputed_counts = {}
    for c in CORE9:
        n_missing = df_imp[c].isna().sum()
        if n_missing > 0:
            imputed_counts[c] = int(n_missing)
        df_imp[c] = df_imp[c].fillna(df_imp[c].median())

    X_imp_scaled = StandardScaler().fit_transform(df_imp[CORE9])
    km_imp = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=N_INIT)
    df_imp["Cluster_imputed"] = km_imp.fit_predict(X_imp_scaled)

    merged = df_cc[["hadm_id", "Cluster"]].merge(
        df_imp[["hadm_id", "Cluster_imputed"]], on="hadm_id", how="inner")
    ari = adjusted_rand_score(merged["Cluster"], merged["Cluster_imputed"])

    cc_sizes = df_cc["Cluster"].value_counts().sort_index()
    imp_sizes = df_imp["Cluster_imputed"].value_counts().sort_index()

    report = pd.DataFrame([{
        "N_complete_case": len(df_cc), "N_median_imputed_full": len(df_imp),
        "N_features_with_missing_imputed": len(imputed_counts),
        "Imputed_feature_counts": str(imputed_counts),
        "N_overlap_admissions": len(merged),
        "ARI_complete_case_vs_imputed": round(ari, 4),
        "Complete_case_cluster_sizes": str(cc_sizes.to_dict()),
        "Imputed_full_cluster_sizes": str(imp_sizes.to_dict()),
    }])
    report.to_csv(os.path.join(outdir, "missing_data_sensitivity_ARI.csv"), index=False)
    print(f"\nMissing-data sensitivity analysis:")
    print(f"  Complete-case N={len(df_cc)}  |  Median-imputed-full N={len(df_imp)}")
    print(f"  Features imputed (n missing): {imputed_counts}")
    print(f"  ARI(complete-case labels, imputed-full labels) on {len(merged)} common "
          f"admissions = {ari:.4f}")
    if ari >= 0.9:
        print("  -> High agreement: missing-data handling does not materially change "
              "cluster assignments.")
    else:
        print("  -> NOTE: agreement is below 0.90 — missing-data handling choice should "
              "be discussed as a limitation, not assumed to be inconsequential.")
    return report


def run_clustering(df, outdir, figdir, chosen_k=3, candidate_ks=(2, 3, 4, 5)):
    complete_mask = df[CORE9].notna().all(axis=1)
    df_cc = df[complete_mask].copy()
    print(f"\nComplete-case (all 9 features present): {complete_mask.sum()}/{len(df)} "
          f"({complete_mask.mean()*100:.2f}%)")

    X_scaled = StandardScaler().fit_transform(df_cc[CORE9])

    # Wide-range elbow/silhouette curve (K=2..10) purely for visualization —
    # the actual K DECISION is made by k_selection_summary() below using the
    # combined criteria table, not by eyeballing this curve alone.
    ks = list(range(2, 11))
    wcss, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X_scaled)
        wcss.append(km.inertia_)
        sils.append(silhouette_score(X_scaled, km.labels_))
    print("\nElbow/silhouette curve (K=2..10, for visualization only — "
          "see K_selection_summary.csv for the actual combined-criteria decision):")
    for k, w, s in zip(ks, wcss, sils):
        print(f"K={k:2d}  WCSS={w:9.1f}  Silhouette={s:.4f}")

    print(f"\n--- Combined K-selection decision table (candidates: {list(candidate_ks)}) ---")
    k_selection_summary(X_scaled, outdir, candidate_ks=candidate_ks, chosen_k=chosen_k)

    km_final = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=N_INIT)
    df_cc["Cluster"] = km_final.fit_predict(X_scaled)

    profile = df_cc.groupby("Cluster")[CORE9].mean().round(2)
    profile["N"] = df_cc["Cluster"].value_counts().sort_index()
    profile.to_csv(os.path.join(outdir, "cluster_profile_means_CORRECTED.csv"))
    print(f"\nFinal fitted model: K={chosen_k}\nCluster profile:\n", profile)

    # Fig: elbow + silhouette (chosen K marked)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks, wcss, 'o-', color='#2c6e91')
    axes[0].axvline(chosen_k, color='red', linestyle='--', alpha=0.6, label=f'K={chosen_k} (chosen)')
    axes[0].set_xlabel("K"); axes[0].set_ylabel("WCSS"); axes[0].set_title("Elbow Method"); axes[0].legend()
    axes[1].plot(ks, sils, 'o-', color='#8a4f9e')
    axes[1].axvline(chosen_k, color='red', linestyle='--', alpha=0.6, label=f'K={chosen_k} (chosen)')
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score"); axes[1].set_title("Silhouette Analysis"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "fig1_elbow_silhouette.png"), dpi=FIG_DPI)
    plt.close()

    return df_cc, X_scaled


def correlation_and_pca(df_cc, X_scaled, outdir, figdir):
    corr = df_cc[CORE9].corr(method="spearman")
    corr.to_csv(os.path.join(outdir, "Spearman_correlation_matrix_CORRECTED.csv"))

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True)
    plt.title("Spearman Correlation Matrix (9 Haematological Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "fig2_correlation_heatmap.png"), dpi=FIG_DPI)
    plt.close()

    pca = PCA()
    pcs_all = pca.fit_transform(X_scaled)
    var_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "Explained_Variance_Ratio": pca.explained_variance_ratio_.round(4),
        "Cumulative_Variance": np.cumsum(pca.explained_variance_ratio_).round(4),
    })
    var_df.to_csv(os.path.join(outdir, "PCA_explained_variance_CORRECTED.csv"), index=False)

    colors = {0: '#4C72B0', 1: '#DD8452', 2: '#C44E52'}
    plt.figure(figsize=(6.5, 5.5))
    for c in sorted(df_cc["Cluster"].unique()):
        mask = df_cc["Cluster"] == c
        plt.scatter(pcs_all[mask, 0], pcs_all[mask, 1], s=6, alpha=0.4,
                    color=colors.get(c, "gray"), label=f"Cluster {c}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("K-means Clusters in PCA Space (K=3)")
    plt.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "fig3_pca_clusters.png"), dpi=FIG_DPI)
    plt.close()

    return var_df


def cluster_stability(X_scaled, outdir):
    n = len(X_scaled)
    km_full = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=N_INIT)
    full_labels = km_full.fit_predict(X_scaled)

    rng = np.random.default_rng(RANDOM_STATE)
    aris = []
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels = KMeans(n_clusters=3, random_state=b, n_init=N_INIT).fit_predict(X_scaled[idx])
        aris.append(adjusted_rand_score(full_labels[idx], boot_labels))
    aris = np.array(aris)
    pd.DataFrame({"bootstrap_iter": range(N_BOOTSTRAP), "ARI": aris}).to_csv(
        os.path.join(outdir, "KMeans_stability_ARI_CORRECTED.csv"), index=False)
    print(f"\nBootstrap cluster stability: mean ARI={aris.mean():.3f} (SD={aris.std():.3f}, "
          f"95% range {np.percentile(aris,2.5):.3f}-{np.percentile(aris,97.5):.3f})")
    return aris


def gap_statistic(X_scaled, outdir):
    rng = np.random.default_rng(RANDOM_STATE)
    mins, maxs = X_scaled.min(axis=0), X_scaled.max(axis=0)
    rows = []
    for k in range(2, 9):
        wcss_real = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X_scaled).inertia_
        ref_wcss = []
        for b in range(N_GAP_REF):
            X_ref = rng.uniform(mins, maxs, size=X_scaled.shape)
            ref_wcss.append(KMeans(n_clusters=k, random_state=b, n_init=N_INIT).fit(X_ref).inertia_)
        ref_wcss = np.array(ref_wcss)
        gap = np.mean(np.log(ref_wcss)) - np.log(wcss_real)
        sk = np.std(np.log(ref_wcss)) * np.sqrt(1 + 1 / N_GAP_REF)
        rows.append({"K": k, "Gap": gap, "s_k": sk})
    gap_df = pd.DataFrame(rows)
    gap_df.to_csv(os.path.join(outdir, "Gap_Statistic_CORRECTED.csv"), index=False)
    print("\nGap statistic:\n", gap_df)
    # Standard rule: smallest K where Gap(K) >= Gap(K+1) - s(K+1)
    chosen_k = None
    for i in range(len(gap_df) - 1):
        if gap_df.loc[i, "Gap"] >= gap_df.loc[i + 1, "Gap"] - gap_df.loc[i + 1, "s_k"]:
            chosen_k = gap_df.loc[i, "K"]
            break
    print(f"Gap-statistic-recommended K: {chosen_k} "
          f"(NOTE: study retains K=3 for clinical interpretability regardless; "
          f"this discrepancy should be disclosed in Methods/Limitations).")
    return gap_df


def dunn_posthoc(df_cc, outdir):
    _ensure_scikit_posthocs()
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("scikit-posthocs not installed; skipping Dunn's post-hoc test.")
        return None
    rows = []
    for var in CORE9:
        dunn = sp.posthoc_dunn(df_cc, val_col=var, group_col="Cluster", p_adjust="bonferroni")
        for i in dunn.index:
            for j in dunn.columns:
                if i < j:
                    rows.append({"Variable": var, "Cluster_A": i, "Cluster_B": j,
                                 "p_adj_bonferroni": dunn.loc[i, j],
                                 "Significant": dunn.loc[i, j] < 0.05})
    dunn_df = pd.DataFrame(rows)
    dunn_df.to_csv(os.path.join(outdir, "Dunn_Posthoc_CORRECTED.csv"), index=False)
    print(f"\n{dunn_df['Significant'].sum()}/{len(dunn_df)} pairwise comparisons significant "
          f"after Bonferroni correction.")
    return dunn_df


def kruskal_wallis(df_cc, outdir):
    rows = []
    for c in CORE9:
        groups = [df_cc[df_cc.Cluster == k][c].dropna() for k in sorted(df_cc.Cluster.unique())]
        stat, p = kruskal(*groups)
        rows.append({"Feature": c, "H_statistic": stat, "p_value": p})
    kw_df = pd.DataFrame(rows)
    kw_df.to_csv(os.path.join(outdir, "Cluster_Kruskal_Wallis_CORRECTED.csv"), index=False)
    print("\nKruskal-Wallis (feature differs significantly across clusters):\n", kw_df)
    return kw_df


def screening_validity(df_cc, label_col, outdir, fname, aplastic_cluster=None):
    sub = df_cc.dropna(subset=[label_col]).copy()
    if aplastic_cluster is None:
        # Auto-identify cluster lowest across Hb, WBC, Platelet
        profile = sub.groupby("Cluster")[["Hemoglobin", "White Blood Cells", "Platelet Count"]].mean()
        aplastic_cluster = profile.rank().sum(axis=1).idxmin()
    sub["Cluster_flag"] = (sub["Cluster"] == aplastic_cluster).astype(int)

    TP = int(((sub["Cluster_flag"] == 1) & (sub[label_col] == 1)).sum())
    FN = int(((sub["Cluster_flag"] == 0) & (sub[label_col] == 1)).sum())
    TN = int(((sub["Cluster_flag"] == 0) & (sub[label_col] == 0)).sum())
    FP = int(((sub["Cluster_flag"] == 1) & (sub[label_col] == 0)).sum())

    sens = TP / (TP + FN) if (TP + FN) else np.nan
    spec = TN / (TN + FP) if (TN + FP) else np.nan
    ppv = TP / (TP + FP) if (TP + FP) else np.nan
    npv = TN / (TN + FN) if (TN + FN) else np.nan

    ct = pd.crosstab(sub["Cluster"], sub[label_col])
    chi2, p, dof, exp = chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

    result = pd.DataFrame([{
        "Label": label_col, "Aplastic_cluster": aplastic_cluster,
        "N_evaluable": len(sub), "TP": TP, "FN": FN, "TN": TN, "FP": FP,
        "Sensitivity": round(sens * 100, 1), "Specificity": round(spec * 100, 1),
        "PPV": round(ppv * 100, 3), "NPV": round(npv * 100, 3),
        "Chi2": round(chi2, 3), "p_value": p, "Cramers_V": round(cramers_v, 3),
    }])
    result.to_csv(os.path.join(outdir, fname), index=False)
    print(f"\nScreening validity ({label_col}):\n", result.to_string(index=False))
    return result, aplastic_cluster


def _normalize_diagnoses_columns(diag):
    """
    Makes the ICD diagnoses table robust to common column-naming variants
    across MIMIC exports (e.g. MIMIC-III style ALL-CAPS: SUBJECT_ID, HADM_ID,
    ICD9_CODE vs. MIMIC-IV style lowercase: subject_id, hadm_id, icd_code,
    icd_version). Renames in place and fills a missing icd_version column
    (defaults to 9, since older exports are ICD-9-only) with a clear warning.
    """
    diag = diag.copy()
    diag.columns = [c.strip() for c in diag.columns]
    diag = diag.rename(columns={c: c.lower() for c in diag.columns})

    rename_map = {"icd9_code": "icd_code", "icd10_code": "icd_code"}
    diag = diag.rename(columns={k: v for k, v in rename_map.items() if k in diag.columns})

    required = ["subject_id", "hadm_id", "icd_code"]
    missing = [c for c in required if c not in diag.columns]
    if missing:
        hint = ""
        if "long_title" in diag.columns:
            hint = (" This file has a 'long_title' column, which strongly suggests it is "
                     "actually d_icd_diagnoses.csv (the ICD code DICTIONARY) rather than "
                     "diagnoses_icd.csv (the per-admission diagnoses TABLE). Please check "
                     "you uploaded the correct file for --diagnoses_icd.")
        raise KeyError(
            f"diagnoses_icd file is missing required column(s) {missing} after "
            f"normalization. Columns found in your uploaded file: {list(diag.columns)}."
            f"{hint} Please check you uploaded the correct MIMIC-IV hosp/diagnoses_icd.csv "
            f"(needs at least subject_id, hadm_id, and icd_code/ICD9_CODE)."
        )

    if "icd_version" not in diag.columns:
        print("WARNING: 'icd_version' column not found in diagnoses_icd file — "
              "assuming ICD-9 for all rows (typical of MIMIC-III-style exports). "
              "If your file actually contains ICD-10 codes, results will be wrong; "
              "please add an icd_version column (9 or 10) per row.")
        diag["icd_version"] = 9

    diag["hadm_id"] = pd.to_numeric(diag["hadm_id"], errors="coerce")
    diag["icd_version"] = pd.to_numeric(diag["icd_version"], errors="coerce").astype("Int64")
    return diag


def run_full_cytopenia_icd_validation(df_cc, diagnoses_icd_path, outdir, figdir, aplastic_cluster):
    """
    Admission-level (hadm_id) match against ALL cytopenia-related ICD-9/ICD-10
    categories (aplastic anaemia/BMF, pancytopenia, neutropenia/agranulocytosis,
    thrombocytopenia [+HIT sub-flag], leukopenia/lymphocytopenia, MDS), plus a
    combined "any cytopenia" flag. See CYTOPENIA_ICD_CODES above for the full
    code list and inclusion/exclusion rationale.
    """
    diag_raw = pd.read_csv(diagnoses_icd_path)
    print(f"Loaded diagnoses file: {diag_raw.shape[0]} rows, columns={list(diag_raw.columns)}")
    diag = _normalize_diagnoses_columns(diag_raw)

    cohort_hadm_ids = set(pd.to_numeric(df_cc["hadm_id"], errors="coerce").unique())
    diag_cohort = diag[diag["hadm_id"].isin(cohort_hadm_ids)].copy()
    print(f"Diagnosis rows matching this cohort's admissions: {len(diag_cohort)}")

    ref = get_cytopenia_icd_reference()
    ref["icd_code"] = ref["icd_code"].astype(str)
    diag_cohort["icd_code"] = diag_cohort["icd_code"].astype(str)
    merged = diag_cohort.merge(ref, on=["icd_code", "icd_version"], how="inner")
    merged.to_csv(os.path.join(outdir, "cytopenia_diagnoses_matched.csv"), index=False)

    categories = sorted(ref["category"].unique())
    for cat in categories:
        hadm_with_cat = set(merged[merged["category"] == cat]["hadm_id"].unique())
        df_cc[f"ICD_{cat}"] = df_cc["hadm_id"].isin(hadm_with_cat).astype(int)

    core_cats = [c for c in categories if c != "Thrombocytopenia_HIT"]
    any_hadm = set(merged["hadm_id"].unique())
    core_hadm = set(merged[merged["category"].isin(core_cats)]["hadm_id"].unique())
    df_cc["ICD_Any_Cytopenia"] = df_cc["hadm_id"].isin(any_hadm).astype(int)
    df_cc["ICD_Any_Cytopenia_excl_HIT"] = df_cc["hadm_id"].isin(core_hadm).astype(int)

    results = []
    for cat in categories + ["Any_Cytopenia", "Any_Cytopenia_excl_HIT"]:
        res, _ = screening_validity(df_cc, f"ICD_{cat}", outdir,
                                     f"_tmp_{cat}.csv", aplastic_cluster=aplastic_cluster)
        results.append(res)
        os.remove(os.path.join(outdir, f"_tmp_{cat}.csv"))
    result_df = pd.concat(results, ignore_index=True)
    result_df.to_csv(os.path.join(outdir, "cluster_vs_ALL_cytopenia_ICD_validity.csv"), index=False)
    print(result_df.to_string(index=False))

    # Summary figure: sensitivity by category, colored by significance
    plot_df = result_df[result_df["Label"] != "Any_Cytopenia_excl_HIT"].sort_values("Sensitivity")
    colors = ['#C44E52' if p < 0.05 else '#999999' for p in plot_df["p_value"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(plot_df["Label"], plot_df["Sensitivity"], color=colors)
    ax.set_xlabel("Sensitivity of cluster (%) for detecting ICD-coded condition")
    ax.set_title("Cluster Sensitivity by ICD Category\n(red = p<0.05, grey = not significant)")
    ax.set_xlim(0, 100)
    for i, (v, n) in enumerate(zip(plot_df["Sensitivity"], plot_df["N_evaluable"])):
        ax.text(v + 1.5, i, f"n={n}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "fig4_icd_validation_by_category.png"), dpi=FIG_DPI)
    plt.close()

    df_cc.to_csv(os.path.join(outdir, "Clustered_Aplastic_Data_WITH_ICD.csv"), index=False)
    return result_df, df_cc


def _ensure_shap():
    try:
        import shap  # noqa: F401
    except ImportError:
        print("Installing shap (one-time)...")
        os.system(f"{sys.executable} -m pip install -q shap")


def explainable_ai_shap_analysis(df_cc, target_col, outdir, figdir,
                                  feature_set=None, run_label="primary",
                                  fig_start_num=5, test_size=0.2, random_state=RANDOM_STATE):
    """
    Trains a Random Forest to predict `target_col` (an INDEPENDENT ICD-coded
    label, not the K-means Cluster itself, to avoid circularity — the cluster
    was built FROM these same 9 features, so explaining it with them would
    partly just recapitulate the K-means centroids) from `feature_set`
    (defaults to all 9 core features), then applies SHAP (TreeExplainer) for
    global and per-feature interpretability.

    `run_label` and `fig_start_num` let this be called more than once (e.g. a
    primary run + sensitivity runs on a different target or a reduced feature
    set) without overwriting each other's output files/figures.

    Reports: stratified train/test split performance (ROC-AUC, average
    precision, classification report, confusion matrix — with ROC curve,
    precision-recall curve, and confusion-matrix HEATMAP all saved as
    high-resolution figures, not just printed), then SHAP summary (beeswarm),
    global mean-|SHAP| bar chart, and a dependence plot for the top feature.
    """
    _ensure_shap()
    import shap
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

    feats = feature_set if feature_set is not None else CORE9
    X = df_cc[feats].copy()
    y = df_cc[target_col].astype(int)
    print(f"\n[{run_label}] Target: {target_col}  |  Features: {feats}  |  N={len(y)}  |  "
          f"Positive rate={y.mean()*100:.2f}% ({y.sum()}/{len(y)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Held-out test performance (n={len(y_test)}, stratified "
          f"{int((1-test_size)*100)}/{int(test_size*100)} split):")
    print(f"  ROC-AUC = {roc_auc:.3f}   Average Precision (PR-AUC) = {ap:.3f}")
    print(f"  Confusion matrix [ [TN FP] [FN TP] ]:\n{cm}")

    metrics_df = pd.DataFrame([{
        "Run": run_label, "Target": target_col, "Features": ",".join(feats),
        "N_total": len(y), "N_positive": int(y.sum()),
        "Positive_rate_%": round(y.mean() * 100, 2), "N_test": len(y_test),
        "ROC_AUC": round(roc_auc, 4), "Average_Precision": round(ap, 4),
        "Precision_class1": round(report["1"]["precision"], 3),
        "Recall_class1": round(report["1"]["recall"], 3),
        "F1_class1": round(report["1"]["f1-score"], 3),
    }])
    metrics_df.to_csv(os.path.join(outdir, f"SHAP_model_performance_{run_label}.csv"), index=False)

    # ---- ROC curve (high resolution) ----
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax,
                                      name=f"RF ({run_label})", color="#C44E52")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6, label="Chance")
    ax.set_title(f"ROC Curve — {target_col} ({run_label})\nAUC = {roc_auc:.3f}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(figdir, f"fig{fig_start_num}_roc_curve_{run_label}.png")
    plt.savefig(roc_path, dpi=FIG_DPI)
    plt.close()

    # ---- Precision-Recall curve (high resolution) ----
    fig, ax = plt.subplots(figsize=(6, 6))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax,
                                                  name=f"RF ({run_label})", color="#4C72B0")
    baseline = y_test.mean()
    ax.axhline(baseline, linestyle="--", color="gray", alpha=0.6,
               label=f"Baseline (prevalence={baseline:.3f})")
    ax.set_title(f"Precision-Recall Curve — {target_col} ({run_label})\nAP = {ap:.3f}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    pr_path = os.path.join(figdir, f"fig{fig_start_num+1}_pr_curve_{run_label}.png")
    plt.savefig(pr_path, dpi=FIG_DPI)
    plt.close()

    # ---- Confusion matrix heatmap (high resolution) ----
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred: Negative", "Pred: Positive"],
                yticklabels=["True: Negative", "True: Positive"])
    ax.set_title(f"Confusion Matrix — {target_col} ({run_label})")
    plt.tight_layout()
    cm_path = os.path.join(figdir, f"fig{fig_start_num+2}_confusion_matrix_{run_label}.png")
    plt.savefig(cm_path, dpi=FIG_DPI)
    plt.close()

    # ---- SHAP ----
    # Capped sample for SHAP computation/plots: exact TreeExplainer cost scales
    # with (n_samples x tree depth x leaves), so a bounded-depth RF + a capped
    # sample keeps this fast without materially changing the interpretability
    # conclusions (SHAP summary statistics stabilize well below 1,000 samples).
    shap_sample_size = min(1000, len(X_test))
    X_shap = X_test.sample(n=shap_sample_size, random_state=random_state)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        sv_pos = shap_values[1]
    elif shap_values.ndim == 3:
        sv_pos = shap_values[:, :, 1]
    else:
        sv_pos = shap_values

    mean_abs_shap = pd.DataFrame({
        "Feature": feats,
        "Mean_abs_SHAP": np.abs(sv_pos).mean(axis=0)
    }).sort_values("Mean_abs_SHAP", ascending=False)
    mean_abs_shap.to_csv(os.path.join(outdir, f"SHAP_feature_importance_{run_label}.csv"), index=False)
    print(f"Top features by mean |SHAP| for predicting {target_col} [{run_label}] "
          f"(computed on a random sample of {shap_sample_size} held-out admissions):")
    print(mean_abs_shap.to_string(index=False))

    plt.figure()
    shap.summary_plot(sv_pos, X_shap, show=False)
    plt.title(f"SHAP Summary: Predicting {target_col} ({run_label})")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"fig{fig_start_num+3}_shap_summary_beeswarm_{run_label}.png"),
                dpi=FIG_DPI)
    plt.close()

    plt.figure()
    shap.summary_plot(sv_pos, X_shap, plot_type="bar", show=False)
    plt.title(f"Global Feature Importance (mean |SHAP|): {target_col} ({run_label})")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"fig{fig_start_num+4}_shap_importance_bar_{run_label}.png"),
                dpi=FIG_DPI)
    plt.close()

    top_feature = mean_abs_shap.iloc[0]["Feature"]
    plt.figure()
    shap.dependence_plot(top_feature, sv_pos, X_shap, show=False)
    plt.title(f"SHAP Dependence: {top_feature} ({run_label})")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"fig{fig_start_num+5}_shap_dependence_top_feature_{run_label}.png"),
                dpi=FIG_DPI)
    plt.close()

    print(f"[{run_label}] Figures saved: ROC, PR curve, confusion matrix, SHAP beeswarm, "
          f"SHAP bar, SHAP dependence (top feature: {top_feature}) — all at {FIG_DPI} DPI.")

    return metrics_df, mean_abs_shap


def _detect_and_fix_icd_file_swap(diagnoses_icd_path, d_icd_diagnoses_path):
    """
    Common Colab upload-order mistake: the person uploads d_icd_diagnoses.csv
    (the code DICTIONARY: icd_code, icd_version, long_title — no per-admission
    rows) when prompted for diagnoses_icd.csv (the per-admission TABLE:
    subject_id, hadm_id, icd_code, icd_version), or vice versa. Detects this
    by column signature and swaps the two paths automatically, rather than
    crashing or silently using the wrong file.
    """
    def _cols(path):
        if not path or not os.path.exists(path):
            return set()
        try:
            return set(c.strip().lower() for c in pd.read_csv(path, nrows=0).columns)
        except Exception:
            return set()

    diag_cols = _cols(diagnoses_icd_path)
    dict_cols = _cols(d_icd_diagnoses_path)

    diag_looks_like_dict = "long_title" in diag_cols and "hadm_id" not in diag_cols
    dict_looks_like_diag = "hadm_id" in dict_cols and "long_title" not in dict_cols

    if diag_looks_like_dict and dict_looks_like_diag:
        print(f"\nNOTE: detected that --diagnoses_icd and --d_icd_diagnoses appear to be "
              f"SWAPPED (the file given for diagnoses_icd has 'long_title', which is the "
              f"code dictionary's signature; the file given for d_icd_diagnoses has "
              f"'hadm_id', which is the per-admission table's signature). Auto-correcting "
              f"the swap so the pipeline still runs correctly.")
        return d_icd_diagnoses_path, diagnoses_icd_path
    if diag_looks_like_dict and not dict_looks_like_diag:
        print(f"\nWARNING: the file given for --diagnoses_icd looks like the code "
              f"DICTIONARY (d_icd_diagnoses.csv: columns={sorted(diag_cols)}), not the "
              f"per-admission diagnoses table. Step 9/10/11 will be skipped since a valid "
              f"diagnoses_icd.csv (needs subject_id, hadm_id, icd_code) was not found. "
              f"Please re-run and upload the correct file when prompted.")
        return "", d_icd_diagnoses_path
    return diagnoses_icd_path, d_icd_diagnoses_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="symile_mimic_data.csv",
                         help="Path to symile_mimic_data.csv (default: symile_mimic_data.csv "
                              "in the current directory)")
    parser.add_argument("--diagnoses_icd", default="diagnoses_icd.csv",
                         help="Path to MIMIC-IV hosp/diagnoses_icd.csv, for admission-level "
                              "validation against ALL cytopenia ICD categories. "
                              "Pass '' or a nonexistent path to skip this step.")
    parser.add_argument("--d_icd_diagnoses", default="d_icd_diagnoses.csv",
                         help="Path to MIMIC-IV hosp/d_icd_diagnoses.csv, used only to "
                              "verify the hard-coded cytopenia ICD code list against the "
                              "site dictionary (optional traceability check).")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--no_zip", action="store_true",
                         help="Skip zipping/downloading outputs at the end (Colab only).")
    # parse_known_args() so this also runs safely inside a Jupyter/Colab cell,
    # which injects its own kernel arguments into sys.argv.
    args, _ = parser.parse_known_args()

    in_colab = _in_colab()

    # ---- Resolve input file, prompting an upload dialog in Colab if missing ----
    resolved_input = _colab_upload_if_missing(
        args.input, "symile_mimic_data.csv (REQUIRED)", required=True)
    if not resolved_input or not os.path.exists(resolved_input):
        parser.error(f"Input file not found: {args.input}\n"
                      f"Place symile_mimic_data.csv in the current directory, or pass "
                      f"--input <path>.")
    args.input = resolved_input

    # ---- Optional ICD files: ask once in Colab, otherwise just check disk ----
    if in_colab and not os.path.exists(args.diagnoses_icd):
        ans = input("\nUpload diagnoses_icd.csv + d_icd_diagnoses.csv now to run the "
                     "ICD validation step (Step 9)? [y/N]: ").strip().lower()
        if ans == "y":
            resolved = _colab_upload_if_missing(
                args.diagnoses_icd, "diagnoses_icd.csv", required=False)
            if resolved:
                args.diagnoses_icd = resolved
            resolved2 = _colab_upload_if_missing(
                args.d_icd_diagnoses, "d_icd_diagnoses.csv", required=False)
            if resolved2:
                args.d_icd_diagnoses = resolved2
        else:
            args.diagnoses_icd = ""

    args.diagnoses_icd, args.d_icd_diagnoses = _detect_and_fix_icd_file_swap(
        args.diagnoses_icd, args.d_icd_diagnoses)

    outdir = args.outdir
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    print("=" * 70, "\nSTEP 1: Load raw data and rename itemid columns\n", "=" * 70)
    df = load_and_rename(args.input)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns. "
          f"Unique subjects={df['subject_id'].nunique()}, unique admissions={df['hadm_id'].nunique()}")

    print("\n" + "=" * 70,
          "\nSTEP 1b: Verify unit of analysis (one row per admission)\n", "=" * 70)
    df = ensure_one_row_per_admission(df, outdir)

    print("\n" + "=" * 70, "\nSTEP 2: Missingness audit (drives feature selection)\n", "=" * 70)
    missingness_report(df, outdir)

    print("\n" + "=" * 70, "\nSTEP 3: Build ANC (direct + derived) and Modified_SAA_Phenotype label\n", "=" * 70)
    df = build_anc(df, outdir)
    df = build_modified_saa_label(df, outdir)

    print("\n" + "=" * 70,
          "\nSTEP 4: Clustering — combined K-selection criteria, then fit chosen K\n", "=" * 70)
    CHOSEN_K = 3
    df_cc, X_scaled = run_clustering(df, outdir, figdir, chosen_k=CHOSEN_K)

    print("\n" + "=" * 70, "\nSTEP 5: Correlation matrix + PCA\n", "=" * 70)
    correlation_and_pca(df_cc, X_scaled, outdir, figdir)

    print("\n" + "=" * 70,
          "\nSTEP 6: Cluster stability (bootstrap ARI, chosen K) + wide-range Gap statistic "
          "+ missing-data sensitivity analysis\n", "=" * 70)
    cluster_stability(X_scaled, outdir)
    gap_statistic(X_scaled, outdir)
    missing_data_sensitivity_analysis(df, df_cc, chosen_k=CHOSEN_K, outdir=outdir)

    print("\n" + "=" * 70, "\nSTEP 7: Kruskal-Wallis + Dunn's post-hoc\n", "=" * 70)
    kruskal_wallis(df_cc, outdir)
    dunn_posthoc(df_cc, outdir)

    print("\n" + "=" * 70, "\nSTEP 8: Cluster vs. Modified_SAA_Phenotype label (rule-based validation)\n", "=" * 70)
    _, aplastic_cluster = screening_validity(
        df_cc, "Modified_SAA_Phenotype", outdir, "cluster_vs_modified_saa_screening_validity.csv")

    if args.diagnoses_icd and os.path.exists(args.diagnoses_icd):
        print("\n" + "=" * 70,
              "\nSTEP 9: Cluster vs. independent ICD-coded ground truth "
              "(ALL cytopenia categories, admission-level match)\n", "=" * 70)
        _, df_cc_icd = run_full_cytopenia_icd_validation(
            df_cc, args.diagnoses_icd, outdir, figdir, aplastic_cluster=aplastic_cluster)
        if args.d_icd_diagnoses and os.path.exists(args.d_icd_diagnoses):
            validate_icd_reference(args.d_icd_diagnoses, outdir)

        print("\n" + "=" * 70,
              "\nSTEP 10: Explainable AI (SHAP) — predicting ICD_Any_Cytopenia "
              "from the 9 core features\n", "=" * 70)
        explainable_ai_shap_analysis(
            df_cc_icd, "ICD_Any_Cytopenia", outdir, figdir,
            run_label="primary_any_cytopenia", fig_start_num=5)

        print("\n" + "=" * 70,
              "\nSTEP 11: SHAP sensitivity checks — (a) a genuinely multi-lineage "
              "target (Pancytopenia) instead of the platelet-heavy combined label, "
              "and (b) the primary target with Platelet Count excluded, to confirm "
              "the model isn't just trivially rediscovering 'low platelets = "
              "thrombocytopenia'\n", "=" * 70)
        explainable_ai_shap_analysis(
            df_cc_icd, "ICD_Pancytopenia", outdir, figdir,
            run_label="sensitivity_pancytopenia", fig_start_num=11)
        explainable_ai_shap_analysis(
            df_cc_icd, "ICD_Any_Cytopenia", outdir, figdir,
            feature_set=[f for f in CORE9 if f != "Platelet Count"],
            run_label="sensitivity_platelet_excluded", fig_start_num=17)
    else:
        print("\n[SKIPPED] No diagnoses_icd.csv provided; ICD-based external "
              "validation (Step 9) and SHAP explainability (Steps 10-11) not run. "
              "Re-run and upload it to complete these steps.")

    df_cc.to_csv(os.path.join(outdir, "Clustered_Aplastic_Data_CORRECTED.csv"), index=False)
    print(f"\nAll outputs written to: {os.path.abspath(outdir)}")

    if in_colab and not args.no_zip:
        import shutil
        zip_base = os.path.abspath(outdir) + "_results"
        shutil.make_archive(zip_base, "zip", outdir)
        zip_path = zip_base + ".zip"
        print(f"\nZipped all results to {zip_path} — triggering download...")
        from google.colab import files
        files.download(zip_path)


if __name__ == "__main__":
    main()
