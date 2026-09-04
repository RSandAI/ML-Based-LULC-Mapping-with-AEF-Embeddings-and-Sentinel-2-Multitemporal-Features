!pip install shap xgboost lightgbm joblib -q

from google.colab import drive
drive.mount('/gdrive')

import os
import time
import json
import warnings
from copy import deepcopy
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

warnings.filterwarnings("ignore")

from sklearn.model_selection    import (GroupShuffleSplit, GridSearchCV,
                                         StratifiedGroupKFold, learning_curve)
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.ensemble           import RandomForestClassifier
from sklearn.svm                import LinearSVC  # SVC
from sklearn.tree               import DecisionTreeClassifier
from sklearn.naive_bayes        import GaussianNB
from sklearn.metrics            import (accuracy_score, f1_score, confusion_matrix,
                                         classification_report, precision_score,
                                         recall_score, cohen_kappa_score,
                                         matthews_corrcoef, roc_auc_score,
                                         balanced_accuracy_score)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
import shap

import builtins

def print(msg, *args):
    try:
        text = msg % args if args else msg
    except Exception:
        text = str(msg) + " " + " ".join(str(a) for a in args)
    builtins.print(text, flush=True)

"""### **1. YAPILANDIRMA**"""

def log_environment(cfg):

    import platform
    import datetime

    manifest = {
        'timestamp'       : datetime.datetime.now().isoformat(),
        'python_version'  : platform.python_version(),
        'random_state'    : cfg.RANDOM_STATE,
        'package_versions': {
            'scikit-learn': __import__('sklearn').__version__,
            'xgboost'     : __import__('xgboost').__version__,
            'lightgbm'    : __import__('lightgbm').__version__,
            'shap'        : shap.__version__,
            'numpy'       : np.__version__,
            'pandas'      : pd.__version__,
        }
    }

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    manifest_path = os.path.join(cfg.OUTPUT_DIR, 'run_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Ortam/versiyon manifesti kaydedildi → %s", manifest_path)
    return manifest

class Config:
    DATASET_LABEL = 'AlphaEarth'

    CSV_PATH       = '/gdrive/MyDrive/AlphaEarth/alphaearth.csv'
    CLASS_COLUMN   = 'Class'
    POLYGON_COLUMN = 'polygon_id'
    DROP_COLUMNS   = ['CRS']

    CLASS_LABEL_MAP = {
        '10' : 'Hazelnut',
        '20' : 'Forest',
        '30' : 'Permanent Cropland',
        '50' : 'Grassland',
        '60' : 'Sparsely Vegetated',
        '70' : 'Arable Land',
        '80' : 'Urban',
        '90' : 'Road and Rail',
        '100': 'Water Course',
        '110': 'Water Bodies',
        '120': 'Wetland',
    }

    TEST_SIZE  = 0.10
    VAL_SIZE   = 0.20

    GRIDSEARCH_SAMPLES_PER_CLASS = 1000

    TRAIN_SAMPLES_PER_CLASS = None

    CORR_THRESHOLD = None

    SAMPLE_SIZE_TEST_SIZES = [5_000, 15_000, 30_000]

    OUTPUT_DIR   = '/gdrive/MyDrive/AlphaEarth/'
    RANDOM_STATE = 42

    FIGURE_DPI = 300

    POLYGON_METADATA_PATH = '/gdrive/MyDrive/AlphaEarth/alphaearth_polygon_metadata.csv'

    SPATIAL_BLOCK_SIZE_M    = None
    SPATIAL_BLOCK_MULTIPLIER = 3.0

    BOOTSTRAP_N       = 1000
    REPEATED_SPLIT_K  = 10

"""### **2. VERİ YÖNETİMİ**"""

class DataManager:

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_and_prepare(self):

        print("=" * 60)
        print("Veri yükleniyor: %s", self.cfg.CSV_PATH)

        df = pd.read_csv(self.cfg.CSV_PATH)
        print("Boyut: %d satır × %d kolon", df.shape[0], df.shape[1])
        print("Sınıf dağılımı:\n%s", df[self.cfg.CLASS_COLUMN].value_counts().to_string())

        if self.cfg.POLYGON_COLUMN not in df.columns:
            raise ValueError(
                f"'{self.cfg.POLYGON_COLUMN}' kolonu bulunamadı. "
                "gee_embedding_pipeline.py'nin güncel versiyonunu çalıştır."
            )
        polygon_ids = df[self.cfg.POLYGON_COLUMN].values

        drop = self.cfg.DROP_COLUMNS + [self.cfg.POLYGON_COLUMN]
        df.drop(columns=[c for c in drop if c in df.columns], inplace=True)

         before = len(df)
        mask   = df.notna().all(axis=1)
        df     = df[mask]
        polygon_ids = polygon_ids[mask.values]
        removed = before - len(df)
        if removed:
            print("%d NaN satırı çıkarıldı.", removed)

        feature_cols = [c for c in df.columns if c != self.cfg.CLASS_COLUMN]
        X     = df[feature_cols].copy()
        y_raw = df[self.cfg.CLASS_COLUMN].copy()

        if self.cfg.CLASS_LABEL_MAP:
            y_raw = y_raw.astype(str).map(
                {str(k): v for k, v in self.cfg.CLASS_LABEL_MAP.items()}
            ).fillna(y_raw.astype(str))

        print("Özellik sayısı (ham): %d", X.shape[1])

        if self.cfg.CORR_THRESHOLD is not None:
            X = self._remove_high_corr_features(X, self.cfg.CORR_THRESHOLD)
            print("Özellik sayısı (korelasyon filtresi sonrası): %d", X.shape[1])

        label_encoder = LabelEncoder()
        y             = label_encoder.fit_transform(y_raw)
        class_names   = [str(c) for c in label_encoder.classes_]

        print("Sınıflar: %s", class_names)
        print("Benzersiz poligon sayısı: %d", len(np.unique(polygon_ids)))

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        joblib.dump(label_encoder,   f"{self.cfg.OUTPUT_DIR}label_encoder.pkl")
        joblib.dump(list(X.columns), f"{self.cfg.OUTPUT_DIR}feature_columns.pkl")
        print("Artifact'lar kaydedildi (encoder, feature_columns) → %s", self.cfg.OUTPUT_DIR)
        print("NOT: scaler.pkl split SONRASI, sadece train verisiyle fit_scaler() ile üretilecek.")

        return X, y, class_names, label_encoder, polygon_ids

    def polygon_split(self, X, y, polygon_ids):
        gss_test = GroupShuffleSplit(n_splits=1, test_size=self.cfg.TEST_SIZE,
                                     random_state=self.cfg.RANDOM_STATE)
        trainval_idx, test_idx = next(gss_test.split(X, y, groups=polygon_ids))

        poly_trainval = polygon_ids[trainval_idx]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=self.cfg.VAL_SIZE,
                                    random_state=self.cfg.RANDOM_STATE)
        train_rel, val_rel = next(
            gss_val.split(X.iloc[trainval_idx], y[trainval_idx], groups=poly_trainval)
        )
        train_idx = trainval_idx[train_rel]
        val_idx   = trainval_idx[val_rel]

        n_total = len(y)
        print("Poligon bazlı split özeti:")
        print("  Train : %8d piksel (%%%.0f) | %d poligon",
                 len(train_idx), len(train_idx)/n_total*100,
                 len(np.unique(polygon_ids[train_idx])))
        print("  Val   : %8d piksel (%%%.0f) | %d poligon",
                 len(val_idx), len(val_idx)/n_total*100,
                 len(np.unique(polygon_ids[val_idx])))
        print("  Test  : %8d piksel (%%%.0f) | %d poligon",
                 len(test_idx), len(test_idx)/n_total*100,
                 len(np.unique(polygon_ids[test_idx])))

        train_polys = set(polygon_ids[train_idx])
        val_polys   = set(polygon_ids[val_idx])
        test_polys  = set(polygon_ids[test_idx])
        assert len(train_polys & test_polys) == 0, "Train-Test poligon çakışması!"
        assert len(train_polys & val_polys)  == 0, "Train-Val poligon çakışması!"
        assert len(val_polys   & test_polys) == 0, "Val-Test poligon çakışması!"
        print("Poligon sızıntısı kontrolü: GEÇTI")

        return (X.iloc[train_idx], y[train_idx], polygon_ids[train_idx],
                X.iloc[val_idx],   y[val_idx],   polygon_ids[val_idx],
                X.iloc[test_idx],  y[test_idx],  polygon_ids[test_idx])

    def fit_scaler(self, X_train, X_val, X_test):
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                       columns=X_train.columns, index=X_train.index)
        X_val_scaled   = pd.DataFrame(scaler.transform(X_val),
                                       columns=X_val.columns, index=X_val.index)
        X_test_scaled  = pd.DataFrame(scaler.transform(X_test),
                                       columns=X_test.columns, index=X_test.index)

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        joblib.dump(scaler, f"{self.cfg.OUTPUT_DIR}scaler.pkl")
        print("Scaler SADECE train üzerinde fit edildi (data leakage önlendi) → %sscaler.pkl",
              self.cfg.OUTPUT_DIR)

        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    @staticmethod
    def balanced_sample(X, y, samples_per_class: int, random_state: int, polygon_ids=None):
        rng     = np.random.default_rng(random_state)
        indices = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            n = min(samples_per_class, len(cls_idx))
            indices.extend(rng.choice(cls_idx, n, replace=False))
        indices = np.array(indices)

        if polygon_ids is not None:
            return X.iloc[indices], y[indices], polygon_ids[indices]
        return X.iloc[indices], y[indices]

    def load_polygon_metadata(self) -> pd.DataFrame:
        meta = pd.read_csv(self.cfg.POLYGON_METADATA_PATH)
        print("Polygon metadata yüklendi: %d poligon | %s", len(meta), self.cfg.POLYGON_METADATA_PATH)
        return meta

    @staticmethod
    def suggest_block_size(metadata_df: pd.DataFrame, multiplier: float = 3.0):
        coords       = metadata_df[['centroid_x', 'centroid_y']].values
        tree         = cKDTree(coords)
        dists, _     = tree.query(coords, k=2)
        nn_dist      = dists[:, 1]
        median_nn    = float(np.median(nn_dist))
        suggested    = median_nn * multiplier

        print("En-yakın-komşu mesafesi (poligon centroidleri arası, m): "
              "medyan=%.1f | p10=%.1f | p90=%.1f",
              median_nn, np.percentile(nn_dist, 10), np.percentile(nn_dist, 90))
        print("Önerilen spatial block boyutu (medyan NN x %.1f): %.0f m", multiplier, suggested)

        return suggested, nn_dist

    def assign_spatial_blocks(self, polygon_ids: np.ndarray, metadata_df: pd.DataFrame,
                               block_size_m: float) -> np.ndarray:
        meta = metadata_df.set_index('polygon_id')
        cx   = meta.loc[polygon_ids, 'centroid_x'].values
        cy   = meta.loc[polygon_ids, 'centroid_y'].values

        block_x   = np.floor(cx / block_size_m).astype(int)
        block_y   = np.floor(cy / block_size_m).astype(int)
        block_ids = np.array([f"{bx}_{by}" for bx, by in zip(block_x, block_y)])

        print("Spatial block ataması: %d benzersiz blok (boyut=%.0f m) | %d poligon",
              len(np.unique(block_ids)), block_size_m, len(np.unique(polygon_ids)))
        return block_ids

    def spatial_block_split(self, X, y, polygon_ids, block_ids):
        gss_test = GroupShuffleSplit(n_splits=1, test_size=self.cfg.TEST_SIZE,
                                     random_state=self.cfg.RANDOM_STATE)
        trainval_idx, test_idx = next(gss_test.split(X, y, groups=block_ids))

        block_trainval = block_ids[trainval_idx]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=self.cfg.VAL_SIZE,
                                    random_state=self.cfg.RANDOM_STATE)
        train_rel, val_rel = next(
            gss_val.split(X.iloc[trainval_idx], y[trainval_idx], groups=block_trainval)
        )
        train_idx = trainval_idx[train_rel]
        val_idx   = trainval_idx[val_rel]

        n_total = len(y)
        print("Spatial block split özeti:")
        print("  Train : %8d piksel (%%%.0f) | %d blok | %d poligon",
              len(train_idx), len(train_idx)/n_total*100,
              len(np.unique(block_ids[train_idx])), len(np.unique(polygon_ids[train_idx])))
        print("  Val   : %8d piksel (%%%.0f) | %d blok | %d poligon",
              len(val_idx), len(val_idx)/n_total*100,
              len(np.unique(block_ids[val_idx])), len(np.unique(polygon_ids[val_idx])))
        print("  Test  : %8d piksel (%%%.0f) | %d blok | %d poligon",
              len(test_idx), len(test_idx)/n_total*100,
              len(np.unique(block_ids[test_idx])), len(np.unique(polygon_ids[test_idx])))

        train_blocks = set(block_ids[train_idx])
        val_blocks   = set(block_ids[val_idx])
        test_blocks  = set(block_ids[test_idx])
        assert len(train_blocks & test_blocks) == 0, "Train-Test blok çakışması!"
        assert len(train_blocks & val_blocks)  == 0, "Train-Val blok çakışması!"
        assert len(val_blocks   & test_blocks) == 0, "Val-Test blok çakışması!"
        print("Spatial block sızıntısı kontrolü: GEÇTI")

        return (X.iloc[train_idx], y[train_idx], polygon_ids[train_idx], block_ids[train_idx],
                X.iloc[val_idx],   y[val_idx],   polygon_ids[val_idx],   block_ids[val_idx],
                X.iloc[test_idx],  y[test_idx],  polygon_ids[test_idx],  block_ids[test_idx])

    def summarize_split(self, polygon_ids: np.ndarray, y: np.ndarray, class_names,
                         metadata_df: pd.DataFrame, partition_labels: np.ndarray) -> pd.DataFrame:
        meta = metadata_df.set_index('polygon_id')
        df = pd.DataFrame({
            'polygon_id': polygon_ids,
            'class'     : [class_names[c] for c in y],
            'partition' : partition_labels,
        })

        rows = []
        for (cls, part), grp in df.groupby(['class', 'partition']):
            unique_polys = grp['polygon_id'].unique()
            areas        = meta.loc[unique_polys, 'area_m2']
            rows.append({
                'class'                  : cls,
                'partition'              : part,
                'n_polygons'             : len(unique_polys),
                'n_pixels'               : len(grp),
                'mapped_area_ha'         : round(areas.sum() / 10_000, 2),
                'median_polygon_size_m2' : round(areas.median(), 1),
            })

        summary = pd.DataFrame(rows).sort_values(['class', 'partition']).reset_index(drop=True)
        return summary

    @staticmethod
    def _remove_high_corr_features(X: pd.DataFrame, threshold: float) -> pd.DataFrame:
        corr_matrix = X.corr().abs()
        upper       = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        if to_drop:
            print("%d yüksek korelasyonlu özellik çıkarıldı: %s%s",
                     len(to_drop), to_drop[:5], '...' if len(to_drop) > 5 else '')
        return X.drop(columns=to_drop)

"""### **3. MODEL KAYIT DEFTERİ**"""

class ModelRegistry:
    @staticmethod
    def get_models_and_grids(random_state: int = 42):
        return [
            (
                'RandomForest',
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', RandomForestClassifier(
                        class_weight='balanced',
                        random_state=random_state,
                        n_jobs=8
                    )),
                ]),
                {
                    'clf__n_estimators'    : [100, 200, 300],
                    'clf__max_depth'       : [10, 20, None],
                    'clf__min_samples_leaf': [1, 2],
                    'clf__max_features'    : ['sqrt', 'log2'],
                }
            ),
            (
                'DecisionTree',
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', DecisionTreeClassifier(
                        class_weight='balanced',
                        random_state=random_state
                    )),
                ]),
                {
                    'clf__max_depth'        : [10, 20, 30],
                    'clf__min_samples_split': [2, 5],
                    'clf__min_samples_leaf' : [1, 2],
                }
            ),
            (
                'XGBoost',
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', XGBClassifier(
                        eval_metric='mlogloss',
                        random_state=random_state,
                        n_jobs=8,
                        verbosity=0
                    )),
                ]),
                {
                    'clf__n_estimators'    : [100, 200],
                    'clf__learning_rate'   : [0.05, 0.1],
                    'clf__max_depth'       : [4, 6],
                    'clf__subsample'       : [0.8, 1.0],
                    'clf__colsample_bytree': [0.6, 0.8],
                }
            ),
            (
                'LightGBM',
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LGBMClassifier(
                        class_weight='balanced',
                        random_state=random_state,
                        n_jobs=8,
                        verbosity=-1
                    )),
                ]),
                {
                    'clf__n_estimators' : [100, 200],
                    'clf__learning_rate': [0.05, 0.1],
                    'clf__num_leaves'   : [31, 63],
                    'clf__max_depth'    : [-1, 10],
                }
            ),
            (
                'LinearSVC',
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LinearSVC(
                        class_weight='balanced',
                        random_state=random_state,
                        dual=False,
                        max_iter=5000
                    )),
                ]),
                {
                    'clf__C': [0.1, 1, 10],
                }
            ),
        ]

"""#### **3b. BELİRSİZLİK**"""

class BootstrapEvaluator:
    @staticmethod
    def _resample_indices(polygon_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        unique_polys  = np.unique(polygon_ids)
        sampled_polys = rng.choice(unique_polys, size=len(unique_polys), replace=True)
        idx_map       = {p: np.where(polygon_ids == p)[0] for p in unique_polys}
        return np.concatenate([idx_map[p] for p in sampled_polys])

    @staticmethod
    def polygon_clustered_ci(y_true, y_pred, polygon_ids, metric_fn,
                             n_boot: int = 1000, ci: float = 95,
                             random_state: int = 42, **metric_kwargs) -> dict:
        rng         = np.random.default_rng(random_state)
        y_true      = np.asarray(y_true)
        y_pred      = np.asarray(y_pred)
        polygon_ids = np.asarray(polygon_ids)

        point_estimate = metric_fn(y_true, y_pred, **metric_kwargs)
        boot_scores    = np.empty(n_boot)
        for b in range(n_boot):
            idx = BootstrapEvaluator._resample_indices(polygon_ids, rng)
            boot_scores[b] = metric_fn(y_true[idx], y_pred[idx], **metric_kwargs)

        lower_p = (100 - ci) / 2
        upper_p = 100 - lower_p
        ci_low, ci_high = np.percentile(boot_scores, [lower_p, upper_p])

        return {'point_estimate': point_estimate, 'ci_low': ci_low,
                'ci_high': ci_high, 'boot_scores': boot_scores}

    @staticmethod
    def paired_comparison(y_true, y_pred_a, y_pred_b, polygon_ids, metric_fn,
                          n_boot: int = 1000, random_state: int = 42,
                          **metric_kwargs) -> dict:
        rng         = np.random.default_rng(random_state)
        y_true      = np.asarray(y_true)
        y_pred_a    = np.asarray(y_pred_a)
        y_pred_b    = np.asarray(y_pred_b)
        polygon_ids = np.asarray(polygon_ids)

        point_diff = (metric_fn(y_true, y_pred_a, **metric_kwargs)
                      - metric_fn(y_true, y_pred_b, **metric_kwargs))
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx = BootstrapEvaluator._resample_indices(polygon_ids, rng)
            score_a = metric_fn(y_true[idx], y_pred_a[idx], **metric_kwargs)
            score_b = metric_fn(y_true[idx], y_pred_b[idx], **metric_kwargs)
            diffs[b] = score_a - score_b

        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        p_approx = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))

        return {'point_diff': point_diff, 'ci_low': ci_low,
                'ci_high': ci_high, 'p_approx': p_approx, 'diffs': diffs}

"""### **4. EĞİTİM VE DEĞERLENDİRME**"""

class Trainer:
    def __init__(self, cfg: Config, data_manager: DataManager):
        self.cfg  = cfg

    def run_gridsearch(self, models, X_gs, y_gs, groups_gs, cv: int = 5):
        print("=" * 60)
        print("GridSearchCV başlatılıyor — %d örnek | %d-fold StratifiedGroupKFold (poligon bazlı)",
              len(y_gs), cv)
        print("=" * 60)

        sgkf        = StratifiedGroupKFold(n_splits=cv, shuffle=True,
                                           random_state=self.cfg.RANDOM_STATE)
        best_models = []

        for name, pipeline, param_grid in models:
            print("GridSearch: %s ...", name)
            t0  = time.time()
            clf = GridSearchCV(
                estimator=pipeline, param_grid=param_grid,
                cv=sgkf, scoring='f1_weighted',
                n_jobs=8, pre_dispatch='2*n_jobs'
            )
            clf.fit(X_gs, y_gs, groups=groups_gs)
            elapsed = time.time() - t0

            best_clf    = clf.best_estimator_.named_steps['clf']
            best_params = {k.replace('clf__', ''): v for k, v in clf.best_params_.items()}

            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
            cv_results_path = os.path.join(self.cfg.OUTPUT_DIR, f'{name}_gridsearch_full_results.csv')
            pd.DataFrame(clf.cv_results_).to_csv(cv_results_path, index=False)
            print("  Tüm CV konfigürasyonları kaydedildi (%d aday) → %s",
                  len(clf.cv_results_['params']), cv_results_path)

            print("  En iyi params : %s", best_params)
            print("  CV F1 skoru   : %.4f  (%.1fs)", clf.best_score_, elapsed)
            best_models.append((name, best_clf, best_params))

        return best_models

    def run_split_sensitivity(self, best_models, X, y, polygon_ids, block_ids,
                              class_names, output_dir):
        comparison_rows = []

        for split_name, group_array in [('polygon_random', polygon_ids),
                                        ('spatial_block',  block_ids)]:
            print("=" * 60)
            print("Split Sensitivity — %s", split_name)
            print("=" * 60)

            gss_test = GroupShuffleSplit(n_splits=1, test_size=self.cfg.TEST_SIZE,
                                         random_state=self.cfg.RANDOM_STATE)
            trainval_idx, test_idx = next(gss_test.split(X, y, groups=group_array))
            group_trainval = group_array[trainval_idx]
            gss_val = GroupShuffleSplit(n_splits=1, test_size=self.cfg.VAL_SIZE,
                                        random_state=self.cfg.RANDOM_STATE)
            train_rel, _ = next(
                gss_val.split(X.iloc[trainval_idx], y[trainval_idx], groups=group_trainval)
            )
            train_idx = trainval_idx[train_rel]

            X_tr_raw, y_tr = X.iloc[train_idx], y[train_idx]
            X_te_raw, y_te = X.iloc[test_idx],  y[test_idx]

            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns=X.columns, index=X_tr_raw.index)
            X_te = pd.DataFrame(scaler.transform(X_te_raw),     columns=X.columns, index=X_te_raw.index)

            for name, clf_template, _params in best_models:
                clf = deepcopy(clf_template)
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)

                f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
                oa = accuracy_score(y_te, y_pred)

                comparison_rows.append({
                    'split_strategy'   : split_name,
                    'model'            : name,
                    'n_train_groups'   : len(np.unique(group_array[train_idx])),
                    'n_test_groups'    : len(np.unique(group_array[test_idx])),
                    'test_f1_weighted' : round(f1, 4),
                    'test_overall_accuracy': round(oa, 4),
                })
                print("  %s: F1=%.4f | OA=%.4f", name, f1, oa)

        df_cmp = pd.DataFrame(comparison_rows)
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, 'split_sensitivity_comparison.csv')
        df_cmp.to_csv(out_csv, index=False)

        pivot = df_cmp.pivot(index='model', columns='split_strategy', values='test_f1_weighted')
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot.plot(kind='bar', ax=ax)
        ax.set_ylabel('Test F1 (weighted)')
        ax.set_title('Split Stratejisi Duyarlılık Analizi:\nPoligon-Random vs Spatial-Block')
        ax.legend(title='Split Stratejisi')
        plt.tight_layout()
        out_png = os.path.join(output_dir, 'split_sensitivity_comparison.png')
        plt.savefig(out_png, dpi=self.cfg.FIGURE_DPI)
        plt.close(fig)

        print("Split sensitivity karşılaştırması kaydedildi → %s ve %s", out_csv, out_png)
        return df_cmp

    def run_bootstrap_analysis(self, best_models_fitted, X_test, y_test, polygon_ids_test,
                               output_dir, n_boot: int = 1000):
        print("=" * 60)
        print("Bootstrap CI Analizi (n_boot=%d, poligon-kümeli)", n_boot)
        print("=" * 60)

        predictions = {name: clf.predict(X_test) for name, clf, _ in best_models_fitted}

        ci_rows = []
        for name, y_pred in predictions.items():
            f1_res = BootstrapEvaluator.polygon_clustered_ci(
                y_test, y_pred, polygon_ids_test,
                lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0),
                n_boot=n_boot, random_state=self.cfg.RANDOM_STATE
            )
            oa_res = BootstrapEvaluator.polygon_clustered_ci(
                y_test, y_pred, polygon_ids_test, accuracy_score,
                n_boot=n_boot, random_state=self.cfg.RANDOM_STATE
            )
            ci_rows.append({
                'model'           : name,
                'f1_weighted'     : round(f1_res['point_estimate'], 4),
                'f1_ci_low'       : round(f1_res['ci_low'], 4),
                'f1_ci_high'      : round(f1_res['ci_high'], 4),
                'overall_accuracy': round(oa_res['point_estimate'], 4),
                'oa_ci_low'       : round(oa_res['ci_low'], 4),
                'oa_ci_high'      : round(oa_res['ci_high'], 4),
            })
            print("  %s: F1=%.4f [%.4f, %.4f] | OA=%.4f [%.4f, %.4f]", name,
                  f1_res['point_estimate'], f1_res['ci_low'], f1_res['ci_high'],
                  oa_res['point_estimate'], oa_res['ci_low'], oa_res['ci_high'])

        ci_df = pd.DataFrame(ci_rows)
        os.makedirs(output_dir, exist_ok=True)
        ci_df.to_csv(os.path.join(output_dir, 'bootstrap_ci_by_model.csv'), index=False)

        pair_rows = []
        for name_a, name_b in combinations(predictions.keys(), 2):
            cmp_res = BootstrapEvaluator.paired_comparison(
                y_test, predictions[name_a], predictions[name_b], polygon_ids_test,
                lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0),
                n_boot=n_boot, random_state=self.cfg.RANDOM_STATE
            )
            pair_rows.append({
                'model_a'    : name_a,
                'model_b'    : name_b,
                'f1_diff'    : round(cmp_res['point_diff'], 4),
                'diff_ci_low': round(cmp_res['ci_low'], 4),
                'diff_ci_high': round(cmp_res['ci_high'], 4),
                'p_approx'   : round(cmp_res['p_approx'], 4),
            })
            print("  %s vs %s: F1 farkı=%.4f [%.4f, %.4f] | p~%.3f",
                  name_a, name_b, cmp_res['point_diff'],
                  cmp_res['ci_low'], cmp_res['ci_high'], cmp_res['p_approx'])

        pair_df = pd.DataFrame(pair_rows)
        pair_df.to_csv(os.path.join(output_dir, 'bootstrap_paired_comparison.csv'), index=False)

        print("Bootstrap CI ve paired karşılaştırma kaydedildi → %s", output_dir)
        return ci_df, pair_df

    def run_repeated_split_evaluation(self, best_models, X, y, polygon_ids,
                                      output_dir, K: int = 10):
        print("=" * 60)
        print("Çoklu Split Tekrarı (K=%d, poligon-bazlı random split)", K)
        print("=" * 60)

        rows = []
        for k in range(K):
            seed = self.cfg.RANDOM_STATE + k

            gss_test = GroupShuffleSplit(n_splits=1, test_size=self.cfg.TEST_SIZE,
                                         random_state=seed)
            trainval_idx, test_idx = next(gss_test.split(X, y, groups=polygon_ids))
            poly_trainval = polygon_ids[trainval_idx]
            gss_val = GroupShuffleSplit(n_splits=1, test_size=self.cfg.VAL_SIZE,
                                        random_state=seed)
            train_rel, _ = next(
                gss_val.split(X.iloc[trainval_idx], y[trainval_idx], groups=poly_trainval)
            )
            train_idx = trainval_idx[train_rel]

            X_tr_raw, y_tr = X.iloc[train_idx], y[train_idx]
            X_te_raw, y_te = X.iloc[test_idx],  y[test_idx]

            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns=X.columns, index=X_tr_raw.index)
            X_te = pd.DataFrame(scaler.transform(X_te_raw),     columns=X.columns, index=X_te_raw.index)

            for name, clf_template, _params in best_models:
                clf = deepcopy(clf_template)
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                rows.append({
                    'repeat'               : k,
                    'model'                : name,
                    'test_f1_weighted'     : f1_score(y_te, y_pred, average='weighted', zero_division=0),
                    'test_overall_accuracy': accuracy_score(y_te, y_pred),
                })

            print("  Tekrar %d/%d tamamlandı", k + 1, K)

        raw_df = pd.DataFrame(rows)
        os.makedirs(output_dir, exist_ok=True)
        raw_df.to_csv(os.path.join(output_dir, 'repeated_split_raw.csv'), index=False)

        summary_rows = []
        for name, grp in raw_df.groupby('model'):
            summary_rows.append({
                'model'     : name,
                'f1_mean'   : round(grp['test_f1_weighted'].mean(), 4),
                'f1_std'    : round(grp['test_f1_weighted'].std(), 4),
                'f1_ci_low' : round(np.percentile(grp['test_f1_weighted'], 2.5), 4),
                'f1_ci_high': round(np.percentile(grp['test_f1_weighted'], 97.5), 4),
                'oa_mean'   : round(grp['test_overall_accuracy'].mean(), 4),
                'oa_std'    : round(grp['test_overall_accuracy'].std(), 4),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(output_dir, 'repeated_split_summary.csv'), index=False)

        print("Çoklu split tekrarı özeti:")
        print(summary_df.to_string(index=False))
        print("Kaydedildi → %s", os.path.join(output_dir, 'repeated_split_summary.csv'))

        return raw_df, summary_df

    def run_sample_size_test(self, best_models, X_train, y_train,
                             X_val, y_val, class_names, output_dir):
        print("=" * 60)
        print("Sample-size testi başlatılıyor — boyutlar: %s",
                 self.cfg.SAMPLE_SIZE_TEST_SIZES)
        print("=" * 60)

        records = []

        for name, clf, _ in best_models:
            print("  Model: %s", name)
            for n in self.cfg.SAMPLE_SIZE_TEST_SIZES:
                X_sub, y_sub = DataManager.balanced_sample(
                    X_train, y_train, n, self.cfg.RANDOM_STATE
                )
                actual_n = len(y_sub)

                t0 = time.time()
                clf.fit(X_sub, y_sub)
                elapsed = time.time() - t0

                y_pred = clf.predict(X_val)
                f1     = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                acc    = accuracy_score(y_val, y_pred)
                kappa  = cohen_kappa_score(y_val, y_pred)

                print("    n=%-6d → F1=%.4f | Acc=%.4f | Kappa=%.4f | %.1fs",
                         actual_n, f1, acc, kappa, elapsed)
                records.append({
                    'model'       : name,
                    'target_n'    : n,
                    'actual_n'    : actual_n,
                    'val_f1_w'    : round(f1, 4),
                    'val_accuracy': round(acc, 4),
                    'val_kappa'   : round(kappa, 4),
                    'fit_time_s'  : round(elapsed, 2),
                })

        df_ss = pd.DataFrame(records)
        df_ss.to_csv(f"{output_dir}sample_size_test.csv", index=False)
        print("Sample-size sonuçları kaydedildi → sample_size_test.csv")

        self._plot_sample_size_results(df_ss, output_dir)
        return df_ss

    def _plot_sample_size_results(self, df_ss: pd.DataFrame, output_dir: str):
        palette = ['#1565C0', '#C62828', '#2E7D32', '#6A1B9A', '#E65100', '#00695C']
        models  = df_ss['model'].unique()

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=self.cfg.FIGURE_DPI)
        fig.patch.set_facecolor('#FAFAFA')
        ax.set_facecolor('#FAFAFA')

        for i, model in enumerate(models):
            sub   = df_ss[df_ss['model'] == model].sort_values('actual_n')
            color = palette[i % len(palette)]
            ax.plot(sub['actual_n'], sub['val_f1_w'],
                    marker='o', linewidth=2, markersize=7,
                    color=color, label=model, zorder=3)
            for _, row in sub.iterrows():
                ax.annotate(f"{row['val_f1_w']:.3f}",
                            (row['actual_n'], row['val_f1_w']),
                            textcoords='offset points', xytext=(0, 9),
                            ha='center', fontsize=8, color=color, fontweight='bold')

        ax.set_xlabel('Training Samples per Class', fontsize=12, fontweight='bold', labelpad=8)
        ax.set_ylabel('Weighted F1 Score (Validation)', fontsize=12, fontweight='bold', labelpad=8)
        ax.set_title('AlphaEarth | Effect of Training Data Size on Model Performance',
                     fontsize=13, fontweight='bold', pad=14)
        ax.legend(fontsize=10, framealpha=0.95, edgecolor='#dddddd',
                  loc='lower right', frameon=True)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.tick_params(axis='both', labelsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.8)
        ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)

        plt.tight_layout()
        plt.savefig(f"{output_dir}sample_size_comparison.png",
                    bbox_inches='tight', dpi=self.cfg.FIGURE_DPI)
        plt.show()
        plt.close()

    @staticmethod
    def compute_metrics(y_true, y_pred, y_prob, split_name):
        m = {
            f'{split_name}_accuracy'         : round(accuracy_score(y_true, y_pred), 4),
            f'{split_name}_balanced_accuracy': round(balanced_accuracy_score(y_true, y_pred), 4),
            f'{split_name}_precision_w'      : round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            f'{split_name}_recall_w'         : round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            f'{split_name}_f1_weighted'      : round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            f'{split_name}_f1_macro'         : round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
            f'{split_name}_cohen_kappa'      : round(cohen_kappa_score(y_true, y_pred), 4),
            f'{split_name}_mcc'              : round(matthews_corrcoef(y_true, y_pred), 4),
        }
        if y_prob is not None:
            try:
                m[f'{split_name}_roc_auc_ovr'] = round(
                    roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'), 4
                )
            except Exception:
                m[f'{split_name}_roc_auc_ovr'] = float('nan')
        return m

    def _save_confusion_matrix(self, cm, cm_norm, class_names,
                                model_name, split_name, output_dir, variant):
        n    = len(class_names)
        cmap = LinearSegmentedColormap.from_list('cm_bilim', ['#F7FBFF', '#2171B5', '#08306B'])

        cell     = max(0.9, 9.5 / n)
        fs_cell  = max(6, 11 - n // 3)
        fs_label = max(8, 11 - n // 4)

        fig, ax = plt.subplots(figsize=(n * cell, n * cell * 0.88), dpi=self.cfg.FIGURE_DPI)
        fig.patch.set_facecolor('white')

        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')

        for i in range(n):
            for j in range(n):
                nv = cm_norm[i, j]
                rv = cm[i, j]
                tc = 'white' if nv > 0.50 else '#1a1a1a'
                if variant == 'normalized':
                    ax.text(j, i, f'{nv:.3f}', ha='center', va='center',
                            fontsize=fs_cell, fontweight='bold', color=tc)
                else:
                    ax.text(j, i - 0.13, f'{nv:.3f}', ha='center', va='center',
                            fontsize=fs_cell, fontweight='bold', color=tc)
                    ax.text(j, i + 0.22, f'n={rv:,}', ha='center', va='center',
                            fontsize=max(5, fs_cell - 2), color=tc, alpha=0.80)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=fs_label)
        ax.set_yticklabels(class_names, fontsize=fs_label)
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold', labelpad=8)
        ax.set_ylabel('True Class',      fontsize=12, fontweight='bold', labelpad=8)

        split_tr    = 'Validation' if split_name == 'val' else 'Test'
        variant_tr  = 'Normalized' if variant == 'normalized' else 'Sample Count'
        ax.set_title(
            f'Confusion Matrix ({split_tr} Set) - {model_name} '
            f'Trained on {self.cfg.DATASET_LABEL} Data',
            fontsize=12, fontweight='bold', pad=12
        )

        for x in np.arange(-0.5, n, 1):
            ax.axhline(x, color='white', linewidth=0.5)
            ax.axvline(x, color='white', linewidth=0.5)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label('Ratio', fontsize=10)

        plt.tight_layout()
        fname = f"{output_dir}{model_name}_cm_{split_name}_{variant}.png"
        plt.savefig(fname, bbox_inches='tight', dpi=self.cfg.FIGURE_DPI)
        plt.show()
        plt.close()
        print("CM kaydedildi: %s", fname)

    def _plot_confusion_matrices(self, cm, cm_norm, class_names,
                                  model_name, split_name, output_dir):
        self._save_confusion_matrix(
            cm, cm_norm, class_names, model_name, split_name, output_dir, 'normalized'
        )
        self._save_confusion_matrix(
            cm, cm_norm, class_names, model_name, split_name, output_dir, 'counts'
        )

    def evaluate_classifier(self, clf, X_train, y_train,
                             X_val, y_val, X_test, y_test,
                             class_names, output_dir):
        model_name = clf.__class__.__name__
        print("-" * 55)
        print("Eğitiliyor: %s", model_name)

        t0       = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0

        results = {'model': model_name, 'fit_time': round(fit_time, 2)}

        for split_name, X_eval, y_eval in [('val', X_val, y_val), ('test', X_test, y_test)]:
            y_pred = clf.predict(X_eval)
            y_prob = clf.predict_proba(X_eval) if hasattr(clf, 'predict_proba') else None
            m      = self.compute_metrics(y_eval, y_pred, y_prob, split_name)
            results.update(m)

            print("[%s] Accuracy=%.4f | BalAcc=%.4f | F1w=%.4f | F1mac=%.4f | "
                     "Kappa=%.4f | MCC=%.4f",
                     split_name.upper(),
                     m[f'{split_name}_accuracy'],
                     m[f'{split_name}_balanced_accuracy'],
                     m[f'{split_name}_f1_weighted'],
                     m[f'{split_name}_f1_macro'],
                     m[f'{split_name}_cohen_kappa'],
                     m[f'{split_name}_mcc'])

            report_dict = classification_report(
                y_eval, y_pred, target_names=class_names,
                digits=4, output_dict=True
            )
            pd.DataFrame(report_dict).transpose().to_csv(
                f"{output_dir}{model_name}_classification_report_{split_name}.csv"
            )

            cm      = confusion_matrix(y_eval, y_pred)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            self._plot_confusion_matrices(
                cm, cm_norm, class_names, model_name, split_name, output_dir
            )

        print("Eğitim süresi: %.1fs", fit_time)

        if hasattr(clf, 'feature_importances_'):
            fi_df = pd.DataFrame({
                'Feature'   : X_train.columns,
                'Importance': clf.feature_importances_
            }).sort_values('Importance', ascending=False)
            fi_df.to_csv(f"{output_dir}{model_name}_feature_importances.csv", index=False)
            self._plot_feature_importance(fi_df, model_name, output_dir)

        joblib.dump(clf, f"{output_dir}{model_name}_model.pkl")
        return results

    def _plot_feature_importance(self, fi_df, model_name, output_dir):
        top20 = fi_df.head(20)

        max_val = top20['Importance'].max()
        colors  = [plt.cm.Blues(0.4 + 0.6 * (v / max_val)) for v in top20['Importance'][::-1]]

        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=self.cfg.FIGURE_DPI)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#FAFAFA')

        bars = ax.barh(top20['Feature'][::-1], top20['Importance'][::-1],
                       color=colors, edgecolor='white', linewidth=0.6, height=0.65)
        for bar, val in zip(bars, top20['Importance'][::-1]):
            ax.text(bar.get_width() + max_val * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8.5, color='#333333')

        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold', labelpad=8)
        ax.set_title(f'AlphaEarth | Top 20 Feature Importances — {model_name}',
                     fontsize=13, fontweight='bold', pad=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.8)
        ax.grid(axis='x', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_xlim(0, max_val * 1.15)
        plt.tight_layout()
        plt.savefig(f"{output_dir}{model_name}_feature_importances.png",
                    bbox_inches='tight', dpi=self.cfg.FIGURE_DPI)
        plt.show()
        plt.close()

"""### **5. SHAP ANALİZİ**"""

class SHAPAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, model, X_test, y_test, class_names, output_dir,
            samples_per_class: int = 50):

        model_name = model.__class__.__name__
        shap_dir   = os.path.join(output_dir, 'SHAP') + os.sep
        os.makedirs(shap_dir, exist_ok=True)
        print(f"SHAP Analizi Başlatıldı — Model: {model_name}")

        rng     = np.random.default_rng(42)
        indices = []
        for cls in np.unique(y_test):
            cls_idx = np.where(y_test == cls)[0]
            n = min(samples_per_class, len(cls_idx))
            indices.extend(rng.choice(cls_idx, n, replace=False))

        X_sub = X_test.iloc[indices].reset_index(drop=True)

        model_output_used         = None
        feature_perturbation_used = None

        try:
            if "LinearSVC" in model_name or "SVC" in model_name:
                explainer = shap.Explainer(model, X_sub)
                model_output_used         = 'raw (linear decision function)'
                feature_perturbation_used = 'interventional (Explainer varsayılanı)'
                shap_values = explainer.shap_values(X_sub)
            else:
                try:
                    explainer = shap.TreeExplainer(
                        model, data=X_sub,
                        model_output='probability',
                        feature_perturbation='interventional'
                    )
                    shap_values = explainer.shap_values(X_sub, check_additivity=False)
                    model_output_used         = 'probability'
                    feature_perturbation_used = 'interventional'
                except Exception as e_prob:
                    print(f"model_output='probability' desteklenmiyor ({e_prob}); "
                          f"ham margin/log-odds uzayına düşülüyor "
                          f"(feature_perturbation='tree_path_dependent'). Bu modelin "
                          f"SHAP büyüklükleri diğer modellerle DOĞRUDAN karşılaştırılamaz.")
                    explainer = shap.TreeExplainer(
                        model, feature_perturbation='tree_path_dependent'
                    )
                    shap_values = explainer.shap_values(X_sub, check_additivity=False)
                    model_output_used         = 'raw margin/log-odds (probability desteklenmedi)'
                    feature_perturbation_used = 'tree_path_dependent'
        except Exception as e:
            print(f"Genel Explainer başarısız oldu, KernelExplainer deneniyor (Yavaş olabilir): {e}")
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_sub, 10))
            shap_values = explainer.shap_values(X_sub)
            model_output_used         = 'predict (KernelExplainer yedek plan)'
            feature_perturbation_used = 'N/A (KernelExplainer)'

        np.save(f"{shap_dir}{model_name}_shap_values.npy", shap_values)

        expected_value = getattr(explainer, 'expected_value', None)
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value.tolist()

        shap_config = {
            'model_name'              : model_name,
            'shap_version'            : shap.__version__,
            'samples_per_class'       : samples_per_class,
            'n_explained_samples'     : len(X_sub),
            'model_output'            : model_output_used,
            'feature_perturbation'    : feature_perturbation_used,
            'background_dataset'      : f'X_sub (test setinden sınıf başına {samples_per_class} stratified örnek)',
            'expected_value'          : expected_value,
        }
        with open(f"{shap_dir}{model_name}_shap_config.json", 'w', encoding='utf-8') as f:
            json.dump(shap_config, f, indent=2, default=str, ensure_ascii=False)
        print(f"SHAP config kaydedildi → {shap_dir}{model_name}_shap_config.json")
        print(f"  model_output={model_output_used} | feature_perturbation={feature_perturbation_used}")

        n_classes = len(class_names)

        if isinstance(shap_values, list):
            get_sv = lambda i: shap_values[i]
        else:
            get_sv = lambda i: shap_values[:, :, i]

        for i, cls_name in enumerate(class_names):
            sv = get_sv(i)
            plt.figure(figsize=(8, 6), dpi=300)
            shap.summary_plot(sv, features=X_sub,
                              feature_names=X_sub.columns.tolist(),
                              max_display=20, show=False)
            plt.title(f"{cls_name} — {model_name}", fontsize=13, fontweight='bold')
            plt.tight_layout()
            safe_name = cls_name.replace(' ', '_').replace('/', '_')
            plt.savefig(f"{shap_dir}shap_summary_{model_name}_{safe_name}.png",
                        bbox_inches='tight')
            plt.show()
            plt.close()

        ncols = 2
        nrows = (n_classes + 1) // 2

        fig = plt.figure(figsize=(14 * ncols / 2, 5.5 * nrows),
                        dpi=self.cfg.FIGURE_DPI)
        fig.patch.set_facecolor('white')

        gs = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.25)

        for i, cls_name in enumerate(class_names):
            sv = get_sv(i)

            if i == n_classes - 1 and n_classes % 2 == 1:
                ax = fig.add_subplot(gs[-1, :])
            else:
                r, c = divmod(i, ncols)
                ax = fig.add_subplot(gs[r, c])

            plt.sca(ax)
            shap.summary_plot(
                sv,
                features=X_sub,
                feature_names=X_sub.columns.tolist(),
                max_display=10,
                show=False,
                plot_size=None,
                color_bar=False
            )

            ax.set_title(cls_name, fontsize=13, fontweight='bold', pad=8)
            ax.tick_params(labelsize=10)

        fig.suptitle(
            f'AlphaEarth | SHAP Summary — All Classes (Top 10)\n{model_name}',
            fontsize=18, fontweight='bold', y=0.92
        )

        plt.tight_layout(rect=[0, 0.07, 1, 0.97])

        cbar_ax = fig.add_axes([0.08, 0.062, 0.84, 0.008])

        gradient = np.linspace(0, 1, 512).reshape(1, -1)
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "custom_shap",
            ["#008afb", "#ff0051"]
        )
        cbar_ax.imshow(gradient, aspect='auto', cmap=cmap)
        cbar_ax.set_axis_off()

        fig.text(0.08, 0.045, "Low", ha='left', va='top', fontsize=12)
        fig.text(0.92, 0.045, "High", ha='right', va='top', fontsize=12)

        fig.text(0.5, 0.035, "Feature Value", ha='center', va='top', fontsize=16)

        plt.savefig(
            f"{shap_dir}shap_summary_{model_name}_all_classes.png",
            bbox_inches='tight',
            dpi=self.cfg.FIGURE_DPI
        )
        plt.show()
        plt.close()

        if isinstance(shap_values, list):
            sv_mean = np.mean([np.abs(get_sv(i)) for i in range(n_classes)], axis=0)
        else:
            sv_mean = np.mean(np.abs(shap_values), axis=2)

        mean_abs = sv_mean.mean(axis=0)
        top_n    = min(20, len(mean_abs))
        fi_shap  = pd.Series(mean_abs, index=X_sub.columns).sort_values(ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(10, 7), dpi=self.cfg.FIGURE_DPI)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#FAFAFA')

        max_val = fi_shap.max()
        colors  = [plt.cm.RdYlBu_r(0.2 + 0.75 * (v / max_val)) for v in fi_shap.values]

        bars = ax.barh(fi_shap.index, fi_shap.values,
                       color=colors, edgecolor='white', linewidth=0.6,
                       height=0.65, zorder=3)

        for bar, val in zip(bars, fi_shap.values):
            ax.text(bar.get_width() + max_val * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', ha='left',
                    fontsize=8.5, color='#333333', fontweight='bold')

        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r',
                                    norm=plt.Normalize(vmin=fi_shap.min(),
                                                       vmax=fi_shap.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Mean |SHAP Value|', fontsize=9, labelpad=8)
        cbar.ax.tick_params(labelsize=8)

        ax.axvline(fi_shap.mean(), color='#E65100', linewidth=1.2,
                   linestyle='--', alpha=0.7, zorder=2,
                   label=f'Mean = {fi_shap.mean():.4f}')

        ax.set_title(f'AlphaEarth | Mean |SHAP| — Top {top_n} Features\n{model_name}',
                     fontsize=13, fontweight='bold', pad=14)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_xlim(0, max_val * 1.22)
        ax.tick_params(axis='both', labelsize=9.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.8)
        ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.45, zorder=0)
        ax.legend(fontsize=9, framealpha=0.9, edgecolor='#dddddd', loc='lower right')

        plt.tight_layout()
        plt.savefig(f"{shap_dir}shap_mean_abs_{model_name}.png",
                    bbox_inches='tight', dpi=self.cfg.FIGURE_DPI)
        plt.show()
        plt.close()

"""### **6. SONUÇ RAPORLAMA**"""

class Reporter:

    METRIC_LABELS = {
        'test_f1_weighted'      : 'F1 Weighted',
        'test_f1_macro'         : 'F1 Macro',
        'test_balanced_accuracy': 'Balanced Accuracy',
        'test_cohen_kappa'      : 'Cohen Kappa',
        'test_mcc'              : 'MCC',
        'test_accuracy'         : 'Overall Accuracy',
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def save_individual_results(self, results: list, output_dir: str):
        for res in results:
            model_name = res['model']
            model_dir  = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            pd.DataFrame([res]).to_csv(
                f"{model_dir}/{model_name}_metrics.csv", index=False
            )
            print("Bireysel metrikler kaydedildi: %s", model_dir)

    def save_combined_results(self, results: list, output_dir: str):
        df = pd.DataFrame(results)
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].round(4)
        df.sort_values('test_f1_weighted', ascending=False, inplace=True)
        path = f"{output_dir}all_models_combined_results.csv"
        df.to_csv(path, index=False)
        print("Birleşik sonuçlar kaydedildi: %s", path)
        return df

    def plot_model_comparison(self, results: list, output_dir: str):
        df_res = pd.DataFrame(results)

        print("=" * 60)
        print("MODEL KARŞILAŞTIRMASI (Test Seti)")
        print("=" * 60)
        display_cols = ['model', 'test_f1_weighted', 'test_balanced_accuracy',
                        'test_cohen_kappa', 'test_mcc', 'fit_time']
        display_cols = [c for c in display_cols if c in df_res.columns]
        print("\n%s", df_res[display_cols].sort_values(
            'test_f1_weighted', ascending=False).to_string(index=False))

        metrics = [(k, v) for k, v in self.METRIC_LABELS.items() if k in df_res.columns]
        n_m     = len(metrics)
        ncols   = min(3, n_m)
        nrows   = (n_m + ncols - 1) // ncols

        palette = ['#1565C0', '#C62828', '#2E7D32', '#6A1B9A', '#E65100', '#00695C']

        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5.8 * ncols, 4.8 * nrows),
                                  dpi=self.cfg.FIGURE_DPI)
        fig.patch.set_facecolor('white')
        axes = np.array(axes).flatten()

        for idx, (metric, label) in enumerate(metrics):
            ax   = axes[idx]
            df_s = df_res.sort_values(metric, ascending=False)
            ax.set_facecolor('#FAFAFA')

            bars = ax.bar(df_s['model'], df_s[metric],
                          color=[palette[i % len(palette)] for i in range(len(df_s))],
                          edgecolor='white', linewidth=0.8, width=0.52, zorder=3)

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.007,
                        f'{h:.3f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='#222222')

            ax.set_title(label, fontsize=12, fontweight='bold', pad=8)
            ax.set_ylabel(label, fontsize=10)
            ax.set_ylim(0, 1.13)
            ax.set_xticklabels(df_s['model'], rotation=32, ha='right', fontsize=9.5)
            ax.tick_params(axis='y', labelsize=9.5)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_linewidth(0.7)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)

        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('AlphaEarth | Model Performance Comparison — Test Set',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f"{output_dir}model_comparison.png",
                    bbox_inches='tight', dpi=self.cfg.FIGURE_DPI)
        plt.show()
        plt.close()
        print("Karşılaştırma grafiği kaydedildi: model_comparison.png")

    def plot_radar_chart(self, results: list, output_dir: str):
        df_res  = pd.DataFrame(results)
        metrics = [k for k in self.METRIC_LABELS if k in df_res.columns]
        labels  = [self.METRIC_LABELS[m] for m in metrics]
        n       = len(metrics)
        if n < 3:
            return

        angles  = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        palette = ['#1565C0', '#C62828', '#2E7D32', '#6A1B9A', '#E65100', '#00695C']

        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.cfg.FIGURE_DPI,
                               subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('white')

        for i, row in df_res.iterrows():
            values  = [row[m] for m in metrics]
            values += values[:1]
            color   = palette[i % len(palette)]
            ax.plot(angles, values, linewidth=2.2, color=color,
                    label=row['model'], zorder=3)
            ax.fill(angles, values, alpha=0.07, color=color)
            for ang, val in zip(angles[:-1], values[:-1]):
                ax.text(ang, val + 0.04, f'{val:.2f}', ha='center', va='center',
                        fontsize=7.5, color=color, fontweight='bold')

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.20', '0.40', '0.60', '0.80', '1.00'], fontsize=8.5,
                           color='#666666')
        ax.set_title('AlphaEarth | Model Performance Profile — Test Set',
                     fontsize=13, fontweight='bold', pad=22)
        ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.18), fontsize=10,
                  framealpha=0.95, edgecolor='#dddddd')
        ax.grid(linestyle='--', linewidth=0.6, alpha=0.6)
        ax.spines['polar'].set_linewidth(0.8)

        plt.tight_layout()
        plt.savefig(f"{output_dir}model_comparison_radar.png",
                    bbox_inches='tight', dpi=self.cfg.FIGURE_DPI)
        plt.show()
        plt.close()
        print("Radar grafiği kaydedildi: model_comparison_radar.png")

    def print_summary_table(self, df_combined: pd.DataFrame):
        cols = ['model'] + [k for k in self.METRIC_LABELS if k in df_combined.columns] + ['fit_time']
        cols = [c for c in cols if c in df_combined.columns]
        print("Özet Tablo:\n%s", df_combined[cols].to_string(index=False))

"""### **7. ÇALIŞTIRMA**"""

cfg          = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

log_environment(cfg)

data_manager      = DataManager(cfg)
trainer           = Trainer(cfg, data_manager)
reporter          = Reporter(cfg)
shap_analyzer     = SHAPAnalyzer(cfg)

# ── 1:
X, y, class_names, label_encoder, polygon_ids = data_manager.load_and_prepare()

# ── 2
(X_train_raw, y_train, polygon_ids_train,
 X_val_raw,   y_val,   polygon_ids_val,
 X_test_raw,  y_test,  polygon_ids_test) = data_manager.polygon_split(X, y, polygon_ids)

# ── 2b
X_train, X_val, X_test, scaler = data_manager.fit_scaler(X_train_raw, X_val_raw, X_test_raw)

# ── 3
X_gs_raw, y_gs, polygon_ids_gs = DataManager.balanced_sample(
    X_train_raw, y_train,
    cfg.GRIDSEARCH_SAMPLES_PER_CLASS,
    cfg.RANDOM_STATE,
    polygon_ids=polygon_ids_train
)

# ── 4
models      = ModelRegistry.get_models_and_grids(cfg.RANDOM_STATE)
best_models = trainer.run_gridsearch(models, X_gs_raw, y_gs, polygon_ids_gs)

# ── 4b
polygon_metadata = data_manager.load_polygon_metadata()

if cfg.SPATIAL_BLOCK_SIZE_M is not None:
    block_size_m = cfg.SPATIAL_BLOCK_SIZE_M
else:
    block_size_m, _nn_dist = DataManager.suggest_block_size(
        polygon_metadata, multiplier=cfg.SPATIAL_BLOCK_MULTIPLIER
    )

block_ids_all = data_manager.assign_spatial_blocks(polygon_ids, polygon_metadata, block_size_m)

# ── 4c
split_sensitivity_df = trainer.run_split_sensitivity(
    best_models, X, y, polygon_ids, block_ids_all, class_names, cfg.OUTPUT_DIR
)

# ── 4d
partition_labels = np.array(
    ['train'] * len(polygon_ids_train) +
    ['val']   * len(polygon_ids_val)   +
    ['test']  * len(polygon_ids_test)
)
all_polygon_ids_split = np.concatenate([polygon_ids_train, polygon_ids_val, polygon_ids_test])
all_y_split            = np.concatenate([y_train, y_val, y_test])

split_summary_df = data_manager.summarize_split(
    all_polygon_ids_split, all_y_split, class_names, polygon_metadata, partition_labels
)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
split_summary_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'split_summary_by_class_partition.csv'), index=False)
print("Partition/class özet tablosu kaydedildi → %s",
      os.path.join(cfg.OUTPUT_DIR, 'split_summary_by_class_partition.csv'))

# ── 4e
(_Xb_tr, yb_tr, polyb_tr, _blk_tr,
 _Xb_val, yb_val, polyb_val, _blk_val,
 _Xb_test, yb_test, polyb_test, _blk_test) = data_manager.spatial_block_split(
    X, y, polygon_ids, block_ids_all
)
partition_labels_block = np.array(
    ['train'] * len(polyb_tr) + ['val'] * len(polyb_val) + ['test'] * len(polyb_test)
)
all_polygon_ids_block = np.concatenate([polyb_tr, polyb_val, polyb_test])
all_y_block            = np.concatenate([yb_tr, yb_val, yb_test])

split_summary_block_df = data_manager.summarize_split(
    all_polygon_ids_block, all_y_block, class_names, polygon_metadata, partition_labels_block
)
split_summary_block_path = os.path.join(cfg.OUTPUT_DIR, 'split_summary_by_class_partition_spatial_block.csv')
split_summary_block_df.to_csv(split_summary_block_path, index=False)
print("Spatial-block partition/class özet tablosu kaydedildi → %s", split_summary_block_path)

# ── 5
trainer.run_sample_size_test(
    best_models, X_train, y_train, X_val, y_val,
    class_names, cfg.OUTPUT_DIR
)

# ── 6
if cfg.TRAIN_SAMPLES_PER_CLASS:
    X_tr, y_tr = DataManager.balanced_sample(
        X_train, y_train,
        cfg.TRAIN_SAMPLES_PER_CLASS,
        cfg.RANDOM_STATE
    )
else:
    X_tr, y_tr = X_train, y_train

results = []
for name, best_clf, _ in best_models:
    model_dir = os.path.join(cfg.OUTPUT_DIR, name) + os.sep
    os.makedirs(model_dir, exist_ok=True)

    res = trainer.evaluate_classifier(
        best_clf, X_tr, y_tr,
        X_val, y_val, X_test, y_test,
        class_names, model_dir
    )
    results.append(res)
    reporter.save_individual_results([res], model_dir)

# ── 6b
bootstrap_ci_df, bootstrap_pair_df = trainer.run_bootstrap_analysis(
    best_models, X_test, y_test, polygon_ids_test, cfg.OUTPUT_DIR,
    n_boot=cfg.BOOTSTRAP_N
)

# ── 6c
repeated_raw_df, repeated_summary_df = trainer.run_repeated_split_evaluation(
    best_models, X, y, polygon_ids, cfg.OUTPUT_DIR, K=cfg.REPEATED_SPLIT_K
)

# ── 7
df_combined = reporter.save_combined_results(results, cfg.OUTPUT_DIR)
reporter.print_summary_table(df_combined)

reporter.plot_model_comparison(results, cfg.OUTPUT_DIR)
reporter.plot_radar_chart(results, cfg.OUTPUT_DIR)

# ── 8
_shap_done = False
for _name, _clf, _ in sorted(
        best_models,
        key=lambda t: f1_score(y_test, t[1].predict(X_test), average='weighted'),
        reverse=True):
    try:
        shap_analyzer.run(_clf, X_test, y_test, class_names, cfg.OUTPUT_DIR)
        print("SHAP tamamlandı — model: %s", _name)
        _shap_done = True
        break
    except Exception as e:
        print("SHAP atlandı (%s): %s", _name, e)

if not _shap_done:
    print("Hiçbir model SHAP ile uyumlu değil, adım atlandı.")

print("Pipeline tamamlandı. Çıktılar: %s", cfg.OUTPUT_DIR)