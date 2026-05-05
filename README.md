<img src="assets/header.png" height=450 width=1280 alt=""/>
<!--
<p align="center">
  <a href="#">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  </a>
  <a href="https://earthengine.google.com/">
    <img src="https://img.shields.io/badge/Platform-Google%20Earth%20Engine-4285F4?logo=google" alt="Google Earth Engine">
  </a>
  <a href="https://www.tandfonline.com/journals/tjde20">
    <img src="https://img.shields.io/badge/Journal-IJDE-orange" alt="Paper">
  </a>
</p>
-->

**Authors:** Elif Sertel¹², Doğu İlmak³, Samet Aksoy²⁴, Beyza Ustaoğlu⁵

¹ University of California, Los Angeles — The B. John Garrick Institute for the Risk Sciences  
² Istanbul Technical University — Dept. of Geomatics Engineering  
³ Mersin University — Department of Remote Sensing and GIS  
⁴ Linnaeus University — Dept. of Forestry and Wood Technology  
⁵ Sakarya University — Department of Geography

**Corresponding author:** [esertel@ucla.edu](mailto:esertel@ucla.edu)

<br>

<details>
<summary><b>Table of Contents</b></summary>

<br>

- [Overview](#overview)
- [Study Area](#study-area)
- [Dataset](#dataset)
- [Feature Configurations](#feature-configurations)
- [Methods](#methods)
- [Performance Metrics](#performance-metrics)
- [Estimated LULC Distribution (Hectares)](#estimated-lulc-distribution-hectares)
- [SHAP Explainability](#shap-explainability)
- [Data Availability](#data-availability)
- [Citation](#citation)

</details>

<br>

## Overview

This repository provides the complete classification pipelines, feature extraction scripts, hyperparameter configurations, and accuracy assessment code for a systematic comparative evaluation of 64-dimensional **AlphaEarth Foundation (AEF) embeddings** against conventional multitemporal **Sentinel-2** spectral feature sets for **11-class LULC mapping** across Sakarya Province, northwestern Türkiye — one of the world's most extensive hazelnut (*Corylus avellana* L.) production regions. Twenty classification experiments were conducted by pairing four input feature configurations with five machine learning algorithms (Random Forest, XGBoost, LightGBM, LinearSVC, and Decision Tree), demonstrating that AEF embeddings consistently outperform all Sentinel-2-based configurations across every algorithm and evaluation metric, with the best-performing AEF + LightGBM model achieving an overall accuracy of **95.75%** and a weighted F1 score of **0.9561**.

<br>

## Study Area

**Sakarya Province**, located in the Marmara–Black Sea transition zone of northwestern Türkiye, encompasses a heterogeneous landscape of dense forest, fragmented agricultural lands, urban fabric, wetlands, and water bodies, making it one of Turkey's most complex and thematically rich LULC mapping environments. The province is also among Turkey's leading hazelnut production regions, situated within the broader Black Sea coastal belt where Turkey accounts for more than 70% of global hazelnut output — approximately 747,000 ha under cultivation and 650,000 tonnes produced as of 2023. The study area covers ~48.9 million valid pixels at 10 m spatial resolution and spans 11 spectrally heterogeneous LULC classes, seven of which correspond to distinct forms of vegetated cover, posing a particularly demanding discrimination problem for medium-resolution satellite-based classification.

<br>

## Dataset

### Reference Samples

Training and validation samples were collected as polygon-based reference data for all 11 LULC classes. Spatial splitting was enforced at the **polygon level** to prevent data leakage from spatial autocorrelation.

**Table 1.** Reference sample distribution (pixels and polygons) used for model training, validation, and testing.

| Partition | AEF Pixels | AEF Polygons | S2 Pixels | S2 Polygons |
|-----------|:----------:|:------------:|:---------:|:-----------:|
| Training | 549,816 (74%) | 4,387 (72%) | 549,816 (74%) | 4,387 (72%) |
| Validation | 132,004 (18%) | 1,097 (18%) | 132,002 (18%) | 1,097 (18%) |
| Test | 64,856 (9%) | 610 (10%) | 64,856 (9%) | 610 (10%) |
| **Total** | **746,676** | **6,094** | **746,674** | **6,094** |


### LULC Classes (11-class scheme)

`Hazelnut` · `Forest` · `Permanent Cropland` · `Arable Land` · `Grassland` · `Sparsely Vegetated Areas` · `Discontinuous Urban Fabric` · `Road and Rail Networks` · `Water Courses` · `Water Bodies` · `Wetland`

<br>

## Feature Configurations

### 1. AlphaEarth Foundation (AEF) Embeddings — 64 dimensions

Google DeepMind's **AlphaEarth Foundations** model fuses Sentinel-1/2, Landsat 8/9, PALSAR-2, GEDI LiDAR, ERA5-Land, and GRACE gravity fields into temporally continuous **64-dimensional pixel embeddings** at 10 m resolution. The Space-Time-Precision (STP) encoder enables temporal interpolation and extrapolation without retraining. Embeddings are available on Google Earth Engine as the **Satellite Embedding V1** dataset (2017–2025).

### 2. Sentinel-2 Multitemporal Bands — 20 features

**Table 2.** Multitemporal Sentinel-2 acquisition dates and phenological purposes.

| Date | Season | Purpose |
|------|--------|---------|
| June 19 & 27, 2025 | Peak vegetation growth | Full canopy development |
| October 30, 2025 | Post-harvest / senescence | Phenological separation |

10 spectral bands per date (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12) resampled to 10 m → **20 features total**.

### 3. Sentinel-2 Spectral Indices — 18 features

Nine indices computed per acquisition date:

**Table 3.** Spectral indices and corresponding biophysical targets utilized for feature engineering.

| Index | Formula | Biophysical Target |
|-------|---------|-------------------|
| NDVIRedEdge | (B8−B5)/(B8+B5) | Vegetation density |
| NDREI | (B6−B5)/(B6+B5) | Canopy chlorophyll |
| RECI | (B7/B5)−1 | Chlorophyll content |
| GCI | (B7/B3)−1 | Green chlorophyll index |
| GNDVI | (B8−B3)/(B8+B3) | Chlorophyll variation |
| GI | B3/B2 | Greenness index |
| NDMI | (B8−B11)/(B8+B11) | Canopy moisture |
| mNDWI | (B3−B11)/(B3+B11) | Surface water |
| NDTI | (B11−B12)/(B11+B12) | Tillage / bare soil |

### 4. Combined Sentinel-2 (Bands + Indices) — 38 features

Full multitemporal stack integrating all spectral bands and indices across both dates.

<br>

## Methods

### Machine Learning Classifiers

**Table 4.** Machine learning classifiers and their key characteristics in the context of the study.

| Classifier | Key Characteristics |
|-----------|-------------------|
| **Random Forest (RF)** | Bootstrap aggregation + feature bagging; robust to multicollinearity |
| **XGBoost** | L1/L2-regularized gradient boosting; sparsity-aware; most consistent across feature sets |
| **LightGBM** | GOSS + EFB; fastest training; highest accuracy with AEF; most sensitive to feature quality |
| **LinearSVC** | LIBLINEAR coordinate descent; linear baseline; competitive in high-dimensional AEF space |
| **Decision Tree** | Non-parametric; interpretable baseline; susceptible to overfitting |

### Hyperparameter Optimization

Stratified 5-fold cross-validated grid search on 11,000 balanced samples (1,000 per class) per feature configuration. **Best configuration hyperparameters for top six models can be found [here](./tables/best_params_top_6.txt).**

<br>

## Performance Metrics

<br>

**Table 5.** Summary of the top 6 performing model configurations based on Overall Accuracy.

| Best Model | Overall Accuracy | Weighted F1 | Cohen's κ | MCC |
|:----------:|:----------------:|:-----------:|:---------:|:---:|
| **AEF + LightGBM** | **95.75%** | **0.9561** | **0.9264** | **0.9266** |
| AEF + XGBoost | 95.57% | 0.9540 | 0.9233 | 0.9234 |
| AEF + Random Forest | 95.45% | 0.9519 | 0.9210 | 0.9212 |
| AEF + LinearSVC | 94.34% | 0.9422 | 0.9022 | 0.9024 |
| S2-All + XGBoost | 93.41% | 0.9313 | 0.8857 | 0.8860 |
| S2-All + Random Forest | 93.17% | 0.9278 | 0.8809 | 0.8814 |

> AEF + LightGBM surpasses the best conventional Sentinel-2 configuration by **+2.34 percentage points** in overall accuracy and outperforms widely used global products (ESA WorldCover ~74.4%, Dynamic World ~72%) by a substantial margin.

### Full Results — All 20 Experiments

**Table 6.** Performance metrics for all 20 classification experiments.

| ID | Feature Set | Model | OA | BA | F1 (W) | F1 (M) | κ | MCC |
|:--:|:------------|:------|:--:|:--:|:------:|:------:|:-:|:---:|
| **1** | **AEF** | **LightGBM** | **0.9575** | **0.8320** | **0.9561** | **0.8553** | **0.9264** | **0.9266** |
| 2 | AEF | XGBoost | 0.9557 | 0.8088 | 0.9540 | 0.8375 | 0.9233 | 0.9234 |
| 3 | AEF | Random Forest | 0.9545 | 0.7961 | 0.9519 | 0.8337 | 0.9210 | 0.9212 |
| 4 | AEF | LinearSVC | 0.9434 | 0.8124 | 0.9422 | 0.8130 | 0.9022 | 0.9024 |
| 5 | AEF | Decision Tree | 0.9210 | 0.7469 | 0.9199 | 0.7468 | 0.8640 | 0.8640 |
| 6 | S2-Indices | Random Forest | 0.9157 | 0.7285 | 0.9110 | 0.7599 | 0.8532 | 0.8537 |
| 7 | S2-Indices | XGBoost | 0.9139 | 0.7459 | 0.9108 | 0.7628 | 0.8513 | 0.8515 |
| 8 | S2-Indices | Decision Tree | 0.8375 | 0.7298 | 0.8535 | 0.6503 | 0.7385 | 0.7432 |
| 9 | S2-Indices | LinearSVC | 0.8149 | 0.7143 | 0.8322 | 0.6253 | 0.7038 | 0.7099 |
| 10 | S2-Indices | LightGBM | 0.8000 | 0.6743 | 0.8261 | 0.5597 | 0.6844 | 0.6911 |
| 11 | S2-Bands | XGBoost | 0.9303 | 0.7801 | 0.9269 | 0.8083 | 0.8790 | 0.8793 |
| 12 | S2-Bands | Random Forest | 0.9248 | 0.7279 | 0.9202 | 0.7663 | 0.8688 | 0.8694 |
| 13 | S2-Bands | Decision Tree | 0.8935 | 0.7269 | 0.8936 | 0.6997 | 0.8173 | 0.8174 |
| 14 | S2-Bands | LinearSVC | 0.8438 | 0.6950 | 0.8580 | 0.6169 | 0.7442 | 0.7471 |
| 15 | S2-Bands | LightGBM | 0.7053 | 0.5354 | 0.7476 | 0.4429 | 0.5483 | 0.5591 |
| **16** | **S2-All** | **XGBoost** | **0.9341** | **0.7816** | **0.9313** | **0.8086** | **0.8857** | **0.8860** |
| 17 | S2-All | Random Forest | 0.9317 | 0.7571 | 0.9278 | 0.7946 | 0.8809 | 0.8814 |
| 18 | S2-All | Decision Tree | 0.8986 | 0.7409 | 0.9013 | 0.7044 | 0.8269 | 0.8270 |
| 19 | S2-All | LinearSVC | 0.8974 | 0.7711 | 0.8997 | 0.7098 | 0.8260 | 0.8267 |
| 20 | S2-All | LightGBM | 0.8288 | 0.7203 | 0.8527 | 0.6271 | 0.7251 | 0.7302 |

> **OA** = Overall Accuracy · **BA** = Balanced Accuracy · **F1 (W)** = Weighted F1 · **F1 (M)** = Macro F1 · **κ** = Cohen's Kappa · **MCC** = Matthews Correlation Coefficient

### Key Algorithmic Observations

- **LightGBM** is the most feature-sensitive classifier: AEF → best performer (95.75%), raw S2-Bands → worst performer (70.53%). A >25 percentage point swing driven solely by input feature quality.
- **XGBoost** is the most robust algorithm, consistently ranking within the top 2 configurations for every dataset group. Recommended as a default when feature quality is uncertain.
- **Balanced Accuracy** is markedly lower than Overall Accuracy across all experiments, reflecting inherent class imbalance in real-world LULC distributions — an expected and ecologically meaningful outcome.
- **Cohen's κ and MCC** converge closely across all 20 experiments, confirming metric consistency and ruling out evaluation artifacts.

### 🌰 Focus: Hazelnut Orchard Classification

The best Hazelnut model (**ID 2, AEF + XGBoost**) correctly identified **>92%** of all actual hazelnut pixels in the unseen test set — without any crop-specific model adaptation or fine-tuning.

**Table 7.** Comparative performance of the top 2 models specifically for the Hazelnut orchard class.

| Model ID | Feature Set | Classifier | Precision | Recall | **F1-Score** |
|:--------:|:------------|:-----------|:---------:|:------:|:------------:|
| **2** | **AEF** | **XGBoost** | **0.8702** | **0.9246** | **0.8966** |
| 1 | AEF | LightGBM | 0.8567 | 0.9298 | 0.8917 |
| 16 | S2-All | XGBoost | 0.8898 | 0.8741 | 0.8819 |

Substituting AEF embeddings for conventional S2-All features yields a **+1.47 percentage point F1 improvement** for hazelnut. This is attributable to the richer multi-sensor, multitemporal phenological signatures encoded within AEF's latent space that two-date optical imagery cannot fully resolve.

> **Detailed Metrics:** For a comprehensive breakdown of all 20 experiments—including per-class precision, recall, and F1-scores—please refer to the **[tables/](./tables/)** directory.

> **Notebooks:** Sentinel-2 All and AlphaEarth training notebooks can be found in **[notebooks/](./notebooks/)**.

<br>

## Estimated LULC Distribution (Hectares)

Pixel-count area estimates across Sakarya Province (~48.9 million valid pixels at 10 m resolution). Values are rounded to the nearest integer.

**Table 8.** Estimated LULC area distribution (hectares) across Sakarya Province derived from top-performing configurations.

| LULC Class | ID 1 (AEF·LGBM) | ID 2 (AEF·XGB) | ID 3 (AEF·RF) | ID 4 (AEF·SVC) | ID 16 (S2·XGB) | ID 17 (S2·RF) |
|:-----------|----------------:|---------------:|--------------:|---------------:|---------------:|--------------:|
| **Forest** | 187,761 | 193,932 | 213,027 | 180,527 | 208,362 | 219,476 |
| **Hazelnut** | 92,195 | 86,900 | 83,542 | 92,249 | 66,689 | 64,313 |
| **Arable Land** | 69,911 | 75,221 | 85,101 | 65,699 | 100,024 | 108,089 |
| **Permanent Cropland** | 38,836 | 34,325 | 26,957 | 47,143 | 26,512 | 19,345 |
| **Grassland** | 28,667 | 27,130 | 18,324 | 25,813 | 22,701 | 17,617 |
| **Sparsely Vegetated** | 22,392 | 23,171 | 13,885 | 24,685 | 17,660 | 14,386 |
| **Urban** | 21,279 | 22,022 | 26,299 | 22,013 | 27,185 | 29,225 |
| **Road and Rail** | 16,371 | 14,981 | 11,240 | 16,499 | 9,145 | 6,670 |
| **Water Bodies** | 9,044 | 9,138 | 9,392 | 9,922 | 9,048 | 9,054 |
| **Wetland** | 1,643 | 1,334 | 815 | 3,326 | 1,135 | 439 |
| **Water Course** | 1,342 | 1,289 | 860 | 1,565 | 979 | 827 |

<div align="center">
  <img src="/assets/Inference.png" width="960" alt="LULC Maps">
  <p>
    <em><b>Figure 1</b> — Predicted LULC maps across Sakarya Province for the six best-performing model configurations.</em>
  </p>
</div>

### Notable Spatial Patterns

- **Forest** is the dominant land cover in all configurations (180,500–219,500 ha), reflecting dense forestation in the northern and eastern portions of the province.
- **Hazelnut** mapped extents are systematically larger in AEF-based models (83,500–92,200 ha) compared to Sentinel-2 models (64,300–66,700 ha). This divergence likely reflects AEF embeddings capturing broader phenological and structural hazelnut signatures that are otherwise misclassified as Arable Land under conventional spectral features.
- **Arable Land** is correspondingly larger in S2 models (100,000–108,100 ha) vs. AEF models (65,700–85,100 ha), consistent with the above hypothesis.
- **Water Bodies** show the highest inter-model consistency (~9,000–9,900 ha), expected given the strong spectral contrast of open water.
- **Wetland and Water Course** exhibit the highest relative variance, consistent with their small spatial extent, spectral ambiguity, and limited training data.

<br>

## SHAP Explainability

SHAP (SHapley Additive exPlanations) analysis was conducted using `TreeExplainer` at three scales: **global**, **class-specific**, and **local (force plots)**.

### AEF Embedding Interpretability (Best Model: ID 1 — AEF + LightGBM)

A limited subset of AEF dimensions drives the majority of model discriminative capacity:

| Embedding | Mean SHAP | Role |
|:---------:|:----------:|:-----|
| **A07** | 0.5688 | Primary discriminator — dense vegetation vs. impervious surfaces |
| **A24** | 0.5337 | Hydrological indicator (Water Bodies, Water Courses) |
| **A51** | 0.5127 | Secondary vegetation structure encoding |

Class-specific SHAP patterns reveal that AEF dimensions function as **semantically interpretable "exclusion filters"**:

- **Forest:** high A07 values → strong positive SHAP; low A07 → Road/Rail
- **Water Bodies & Courses:** A24 consistently generates high positive SHAP
- **Hazelnut:** discriminated by collective, moderate contributions from A33, A61, A10, A39 (SHAP range −2.0 to +2.0) — reflecting multi-dimensional phenological encoding
- **Wetland:** primarily driven by A16 (transitional moisture conditions)
- **Urban:** well-separated through A35 and A36

The **spatial distribution of cumulative Hazelnut SHAP values** (Fig 2) reveals high positive intensities clustering coherently in the northeastern coastal foothills (~30°45'E–31°E, 40°50'N–41°10'N), accurately tracking intensive orchard cultivation zones. The asymmetric SHAP range (−7.42 to **+19.15**) reflects high-confidence positive prediction in core hazelnut areas.

### Sentinel-2 SHAP Analysis (Baseline: ID 16 — S2-All + XGBoost)

- **NDTI_Jun** emerges as the globally dominant feature (mean |SHAP| = 0.7091), mirroring A07's role in the AEF model
- **NDTI_Oct** provides complementary post-season information (0.4058), but June remains ~2× more impactful
- Hazelnut discrimination relies on **B11_Jun** (peak canopy SWIR) and **B6_Jun** (red-edge chlorophyll)
- Cumulative Hazelnut SHAP spatial range is substantially more constrained (−7.58 to +**7.40**) vs. AEF (+19.15) — quantitatively demonstrating the superior discriminative certainty of foundation model embeddings for this perennial crop

<br>

<p align="center">
  <img src="/assets/Hazelnut_SHAP_AlphaEarth.png" width="45%" alt="AlphaEarth">
  <img src="/assets/Hazelnut_SHAP_Sentinel_2.png" width="45%" alt="Sentinel-2">
</p>

<p align="center">
  <em><b>Figure 2</b> — Spatial distribution of cumulative SHAP values for the Hazelnut class. Left: AlphaEarth LightGBM model (ID 1). Right: Sentinel-2 XGBoost model (ID 16).</em>
</p>

<br>

## Data Availability

| Resource | Access |
|----------|--------|
| **Sentinel-2 Level-2A** | [Copernicus Open Access Hub](https://scihub.copernicus.eu) · GEE: `COPERNICUS/S2_SR_HARMONIZED` |
| **AlphaEarth Foundation Embeddings** | Google Earth Engine: `Satellite Embedding V1` (2017–2025, 10 m) |
| **Classification code & configs** | This repository |
| **LULC maps (interactive)** | [GEE Web Application](#) *(link to be updated)* |
| **Detailed Metric Tables** | [View Folder](./tables/) |
| **Training Notebooks** | [View Folder](./notebooks/) |

<br>

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{Sertel2025LULC_AEF,
  title   = {Machine Learning-Based {LULC} Mapping with {AlphaEarth} Foundation Embeddings
             and {Sentinel-2} Multitemporal Features: A Comparative Study Focusing on
             Hazelnut (\textit{Corylus avellana} {L.}) Orchard},
  author  = {Sertel, Elif and Ilmak, Dogu and Aksoy, Samet and Ustaoglu, Beyza},
  journal = {International Journal of Digital Earth},
  year    = {2025},
  note    = {Under review}
}
```
