"""Model definitions. Every model is a full pipeline (imputation, scaling, resampling, classifier),
so preprocessing is learned only from the training fold and nothing leaks from validation/test data."""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import SEED


def _impute():
    # Median, not mean: the ratios have extreme outliers that drag the mean far from typical values
    return SimpleImputer(strategy="median", add_indicator=False)


def _scale():
    # Financial ratios are heavy-tailed (one ranges from -11.9M to 685k). A quantile transform maps
    # each feature to a normal shape, which linear models, SVMs, NB and neural nets need.
    return QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=SEED)


def build_models(pos_weight: float) -> dict:
    """pos_weight = (#non-bankrupt / #bankrupt) in the training data, used to reweight the rare class."""
    return {
        "Baseline (always 'safe')": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("impute", _impute()), ("scale", _scale()),
            ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)),
        ]),
        "Naive Bayes": ImbPipeline([
            ("impute", _impute()), ("scale", _scale()),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", GaussianNB()),
        ]),
        "SVM (RBF)": Pipeline([
            ("impute", _impute()), ("scale", _scale()),
            # SVMs output margins, not probabilities; calibration (Platt scaling) converts them
            ("clf", CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, class_weight="balanced"),
                                           method="sigmoid", ensemble=False)),
        ]),
        "Neural Network (MLP)": ImbPipeline([
            ("impute", _impute()), ("scale", _scale()),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                                  early_stopping=True, max_iter=500, random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("impute", _impute()),
            ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=SEED)),
        ]),
        # Gradient boosting handles missing values natively, so no imputation is needed
        "XGBoost": XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, reg_lambda=1.0, scale_pos_weight=pos_weight,
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=SEED,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=pos_weight,
            n_jobs=-1, random_state=SEED, verbose=-1,
        ),
    }


def build_imbalance_variants(pos_weight: float) -> dict:
    """Same XGBoost model under three ways of handling the 95/5 class imbalance."""
    base = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED)
    return {
        "No handling": XGBClassifier(**base),
        "Class weighting": XGBClassifier(**base, scale_pos_weight=pos_weight),
        "SMOTE oversampling": ImbPipeline([
            ("impute", _impute()), ("smote", SMOTE(random_state=SEED)), ("clf", XGBClassifier(**base)),
        ]),
    }
