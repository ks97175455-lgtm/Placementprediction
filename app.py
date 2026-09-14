import os
import io
import base64
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, silhouette_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, RandomForestClassifier
)
from sklearn.cluster import KMeans

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'placement_predict_50k Dataset.csv')

CLASSIFICATION_DROP = ['StudentID', 'PlacementStatus', 'Salary Package', 'IsAnomaly']
LINEAR_DROP = ['StudentID', 'Salary Package', 'PlacementStatus', 'IsAnomaly']
NUM_FEATURES = ['CGPA', 'AttendancePercent', 'AptitudeTestScore', 'CodingTestScore']

MODEL_LABELS = {
    'logistic': 'Logistic Regression',
    'tree': 'Decision Tree',
    'gradient': 'Gradient Boosting',
    'adaboost': 'AdaBoost',
    'bagging': 'Bagging',
    'boosting': 'Boosting',
    'random-forest': 'Random Forest'
}


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')
    return pd.read_csv(DATA_PATH)


def fig_b64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def summary(df):
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    total = int(rows * cols)
    return {
        'rows': int(rows),
        'columns': int(cols),
        'missing': missing,
        'missing_percentage': round((missing / total) * 100, 2) if total else 0,
        'duplicates': int(df.duplicated().sum()),
        'placed': int(pd.to_numeric(df['PlacementStatus'], errors='coerce').fillna(0).sum()),
        'placement_rate': round(pd.to_numeric(df['PlacementStatus'], errors='coerce').mean() * 100, 2),
        'average_package': round(pd.to_numeric(df['Salary Package'], errors='coerce').mean(), 2),
    }


def make_preprocessor(X, scale_numeric=True):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []
    if numeric:
        num_steps = [('imputer', SimpleImputer(strategy='median'))]
        if scale_numeric:
            num_steps.append(('scaler', StandardScaler()))
        transformers.append(('num', Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical))
    return ColumnTransformer(transformers=transformers, remainder='drop')


def classification_data():
    df = load_data().copy()
    X = df.drop(columns=[c for c in CLASSIFICATION_DROP if c in df.columns])
    y = pd.to_numeric(df['PlacementStatus'], errors='coerce').fillna(0).astype(int)
    return X, y


def build_classifier(kind):
    if kind == 'logistic':
        return LogisticRegression(max_iter=1200, solver='liblinear', random_state=42)
    if kind == 'tree':
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42)
    if kind == 'gradient':
        return GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=42)
    if kind == 'adaboost':
        return AdaBoostClassifier(n_estimators=120, learning_rate=0.5, random_state=42)
    if kind == 'bagging':
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42),
            n_estimators=50, random_state=42, n_jobs=-1
        )
    if kind == 'boosting':
        return GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=2, random_state=42)
    if kind == 'random-forest':
        return RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    raise ValueError(f'Unknown model: {kind}')


def get_feature_importance(model, feature_names):
    try:
        prep = model.named_steps['prep']
        estimator = model.named_steps['model']
        transformed = prep.get_feature_names_out()
        if hasattr(estimator, 'feature_importances_'):
            vals = np.asarray(estimator.feature_importances_)
        elif hasattr(estimator, 'coef_'):
            vals = np.abs(np.asarray(estimator.coef_)[0])
        else:
            return []
        pairs = sorted(zip(transformed, vals), key=lambda x: x[1], reverse=True)[:12]
        return [{'feature': f.replace('num__', '').replace('cat__', ''), 'importance': round(float(v), 4)} for f, v in pairs]
    except Exception:
        return []


@lru_cache(maxsize=16)
def cached_classifier(kind):
    X, y = classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ('prep', make_preprocessor(X_train, scale_numeric=(kind == 'logistic'))),
        ('model', build_classifier(kind))
    ])
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else pred

    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.imshow(cm)
    ax.set_title(f'{MODEL_LABELS[kind]} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1], ['Not Placed', 'Placed'])
    ax.set_yticks([0, 1], ['Not Placed', 'Placed'])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value:,}', ha='center', va='center')
    cm_image = fig_b64(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{MODEL_LABELS[kind]} - ROC Curve')
    ax.legend()
    roc_image = fig_b64(fig)

    return {
        'name': MODEL_LABELS[kind], 'kind': kind,
        'accuracy': round(acc * 100, 2), 'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2), 'f1': round(f1 * 100, 2),
        'auc': round(float(auc), 3), 'train': len(X_train), 'test': len(X_test),
        'features': list(X.columns), 'image': cm_image, 'roc_image': roc_image,
        'confusion_matrix': cm.tolist(),
        'report': classification_report(y_test, pred, target_names=['Not Placed', 'Placed'], zero_division=0),
        'importance': get_feature_importance(model, X.columns),
        'train_seconds': round(train_seconds, 3),
        '_model': model
    }


def public_result(result):
    return {k: v for k, v in result.items() if k != '_model'}


def run_linear():
    df = load_data().copy()
    y = pd.to_numeric(df['Salary Package'], errors='coerce')
    X = df.drop(columns=[c for c in LINEAR_DROP if c in df.columns])
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
    model = Pipeline([('prep', make_preprocessor(X_train, scale_numeric=True)), ('model', LinearRegression())])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** .5
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    sample_n = min(700, len(y_test))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y_test), sample_n, replace=False)
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    ax.scatter(np.asarray(y_test)[idx], np.asarray(pred)[idx], alpha=.3, s=18)
    mn, mx = float(np.min(y_test)), float(np.max(y_test))
    ax.plot([mn, mx], [mn, mx], '--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual Salary Package (LPA)')
    ax.set_ylabel('Predicted Salary Package (LPA)')
    ax.set_title('Linear Regression - Actual vs Predicted')
    ax.legend()
    return {
        'name': 'Linear Regression', 'rmse': round(float(rmse), 3), 'mae': round(float(mae), 3),
        'r2': round(float(r2), 3), 'train': len(X_train), 'test': len(X_test),
        'features': list(X.columns), 'target': 'Salary Package', 'image': fig_b64(fig)
    }


@lru_cache(maxsize=1)
def cached_linear():
    return run_linear()


def run_preprocessing():
    df = load_data().copy()
    numeric = df[NUM_FEATURES].apply(pd.to_numeric, errors='coerce')
    missing_before = int(numeric.isna().sum().sum())
    filled = numeric.fillna(numeric.median())
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(filled), columns=NUM_FEATURES)
    outlier_counts = {}
    for col in NUM_FEATURES:
        q1, q3 = numeric[col].quantile(.25), numeric[col].quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[col] = int(((numeric[col] < lo) | (numeric[col] > hi)).sum())
    return {
        'rows': len(df), 'features': NUM_FEATURES, 'missing_before': missing_before,
        'missing_after': int(scaled.isna().sum().sum()), 'outliers': int(sum(outlier_counts.values())),
        'outlier_counts': outlier_counts,
        'scaled_head': scaled.head(10).round(3).to_dict(orient='records')
    }


def run_eda():
    df = load_data()
    images = []
    labels = []
    plots = [
        ('Placement Status Distribution', lambda ax: df['PlacementStatus'].value_counts().sort_index().plot(kind='bar', ax=ax)),
        ('CGPA Distribution', lambda ax: ax.hist(df['CGPA'].dropna(), bins=20)),
        ('Placement Rate by College Tier', lambda ax: df.groupby('CollegeTier')['PlacementStatus'].mean().mul(100).sort_index().plot(kind='bar', ax=ax)),
        ('CGPA vs Salary Package', lambda ax: ax.scatter(df['CGPA'], df['Salary Package'], alpha=.22, s=12)),
        ('Coding Score vs Placement Rate', lambda ax: pd.qcut(df['CodingTestScore'], 5, duplicates='drop').groupby(lambda x: True) if False else None),
    ]
    for title, draw in plots[:4]:
        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        draw(ax)
        ax.set_title(title)
        if title.startswith('CGPA vs'):
            ax.set_xlabel('CGPA'); ax.set_ylabel('Salary Package (LPA)')
        elif title.startswith('Placement Rate'):
            ax.set_xlabel('College Tier'); ax.set_ylabel('Placement Rate (%)')
        elif title.startswith('CGPA Distribution'):
            ax.set_xlabel('CGPA'); ax.set_ylabel('Students')
        else:
            ax.set_xlabel('Placement Status (0 = Not Placed, 1 = Placed)'); ax.set_ylabel('Students')
        images.append(fig_b64(fig)); labels.append(title)
    return list(zip(labels, images))


def cluster_result(k=3):
    df = load_data().copy()
    features = ['CGPA', 'AptitudeTestScore']
    X = df[features].apply(pd.to_numeric, errors='coerce').dropna()
    scaler = StandardScaler(); Z = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(Z)
    centers = scaler.inverse_transform(model.cluster_centers_)
    sil = silhouette_score(Z, labels)
    data = df.loc[X.index].copy(); data['Cluster'] = labels
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    cluster_profiles = []
    for i in range(k):
        subset = data[data['Cluster'] == i]
        cluster_profiles.append({
            'cluster': i, 'students': int(len(subset)),
            'avg_cgpa': round(float(subset['CGPA'].mean()), 2),
            'avg_aptitude': round(float(subset['AptitudeTestScore'].mean()), 2),
            'placement_rate': round(float(subset['PlacementStatus'].mean() * 100), 2)
        })
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.scatter(X['CGPA'], X['AptitudeTestScore'], c=labels, s=20, alpha=.55)
    ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=240, edgecolors='black', linewidths=1.5, label='Centroids')
    ax.set_xlabel('CGPA'); ax.set_ylabel('Aptitude Test Score'); ax.set_title(f'Student Segmentation using K-Means (K = {k})'); ax.legend()
    return {
        'features': features, 'k': k, 'rows': len(X), 'silhouette': round(float(sil), 3),
        'centers': np.round(centers, 3).tolist(), 'counts': {str(a): int(b) for a, b in counts.items()},
        'profiles': cluster_profiles, 'image': fig_b64(fig)
    }


@lru_cache(maxsize=8)
def cached_cluster(k):
    return cluster_result(k)


def elbow_result():
    df = load_data(); X = df[['CGPA', 'AptitudeTestScore']].apply(pd.to_numeric, errors='coerce').dropna()
    Z = StandardScaler().fit_transform(X)
    ks = list(range(1, 9)); wcss = []
    for k in ks:
        wcss.append(float(KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z).inertia_))
    selected = 3
    fig, ax = plt.subplots(figsize=(7.2, 4.8)); ax.plot(ks, wcss, marker='o', linewidth=2); ax.scatter([selected], [wcss[selected-1]], marker='X', s=180, edgecolors='black', label='Selected K = 3'); ax.set_xlabel('Number of Clusters (K)'); ax.set_ylabel('WCSS'); ax.set_title('Elbow Method'); ax.set_xticks(ks); ax.legend()
    return {'features': ['CGPA', 'AptitudeTestScore'], 'rows': len(X), 'k': selected, 'wcss': {str(k): round(v, 3) for k, v in zip(ks, wcss)}, 'image': fig_b64(fig)}


@lru_cache(maxsize=1)
def cached_elbow(): return elbow_result()


@app.route('/')
def dashboard():
    df = load_data(); s = summary(df); preview = df.head(10).fillna('—').to_dict('records')
    return render_template('index.html', active='dashboard', summary=s, columns=list(df.columns), preview=preview)


@app.route('/dataset')
def dataset():
    df = load_data(); return render_template('index.html', active='dataset', summary=summary(df), columns=list(df.columns), preview=df.head(30).fillna('—').to_dict('records'))


@app.route('/data-quality')
def data_quality():
    df = load_data(); missing = df.isna().sum().sort_values(ascending=False); details = [{'feature': c, 'missing': int(v), 'percentage': round(v / len(df) * 100, 2)} for c, v in missing.items() if v > 0]
    return render_template('index.html', active='data-quality', summary=summary(df), missing_details=details)


@app.route('/eda')
def eda(): return render_template('index.html', active='eda', images=run_eda())


@app.route('/preprocessing')
def preprocessing(): return render_template('index.html', active='preprocessing', data=run_preprocessing())


@app.route('/linear-regression')
def linear(): return render_template('index.html', active='linear-regression', result=cached_linear())


def model_page(kind):
    active_map = {'logistic': 'logistic-regression', 'tree': 'decision-tree', 'gradient': 'gradient-boosting', 'adaboost': 'adaboost', 'bagging': 'bagging', 'boosting': 'boosting', 'random-forest': 'random-forest'}
    return render_template('index.html', active=active_map[kind], result=public_result(cached_classifier(kind)))

@app.route('/logistic-regression')
def logistic(): return model_page('logistic')
@app.route('/decision-tree')
def tree(): return model_page('tree')
@app.route('/gradient-boosting')
def gradient(): return model_page('gradient')
@app.route('/adaboost')
def ada(): return model_page('adaboost')
@app.route('/bagging')
def bagging(): return model_page('bagging')
@app.route('/boosting')
def boosting(): return model_page('boosting')
@app.route('/random-forest')
def random_forest(): return model_page('random-forest')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result = None
    form = request.form.to_dict() if request.method == 'POST' else {}
    if request.method == 'POST':
        try:
            base = load_data().drop(columns=['PlacementStatus', 'Salary Package', 'IsAnomaly'], errors='ignore')
            base = base.drop(columns=['StudentID'], errors='ignore')
            row = {}
            for col in base.columns:
                value = form.get(col, '')
                if pd.api.types.is_numeric_dtype(base[col]):
                    row[col] = float(value) if value != '' else np.nan
                else:
                    row[col] = value if value != '' else np.nan
            X, y = classification_data()
            model_result = cached_classifier('gradient')
            pred = int(model_result['_model'].predict(pd.DataFrame([row]))[0])
            prob = float(model_result['_model'].predict_proba(pd.DataFrame([row]))[0, 1])
            result = {'status': pred, 'label': 'PLACEMENT LIKELY' if pred else 'PLACEMENT AT RISK', 'probability': round(prob * 100, 2), 'model': 'Gradient Boosting', 'importance': model_result['importance'][:6]}
        except Exception as e:
            result = {'error': str(e)}
    input_fields = [c for c in classification_data()[0].columns if c in NUM_FEATURES or c in ['Gender', 'City', 'CollegeTier', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs', 'Internships', 'Projects', 'Certifications', 'Publications', 'SoftSkillsRating', 'MockInterviewScore', 'ExtraCurricular']]
    options = {c: sorted(load_data()[c].dropna().astype(str).unique().tolist())[:100] for c in input_fields if load_data()[c].dtype == 'object'}
    return render_template('index.html', active='prediction', prediction=result, input_fields=input_fields, form=form, options=options)


@app.route('/model-comparison')
def comparison():
    kinds = ['logistic', 'tree', 'gradient', 'adaboost', 'bagging', 'boosting', 'random-forest']
    results = [public_result(cached_classifier(k)) for k in kinds]
    results.sort(key=lambda x: x['f1'], reverse=True)
    return render_template('index.html', active='model-comparison', results=results, best=results[0])


@app.route('/k-means')
def kmeans(): return render_template('index.html', active='k-means', result=cached_cluster(3))
@app.route('/k-means-elbow')
def kmeans_elbow(): return render_template('index.html', active='k-means-elbow', result=cached_elbow())


@app.route('/cross-validation')
def cross_validation():
    X, y = classification_data()
    Xs, _, ys, _ = train_test_split(X, y, test_size=.15, random_state=42, stratify=y)
    # Use a representative sample for responsive UI while retaining stratification.
    if len(Xs) > 12000:
        idx = Xs.sample(12000, random_state=42).index
        Xs, ys = Xs.loc[idx], ys.loc[idx]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for kind in ['logistic', 'tree', 'gradient', 'random-forest']:
        pipe = Pipeline([('prep', make_preprocessor(Xs, scale_numeric=(kind == 'logistic'))), ('model', build_classifier(kind))])
        scores = cross_val_score(pipe, Xs, ys, cv=cv, scoring='f1', n_jobs=1)
        rows.append({'name': MODEL_LABELS[kind], 'folds': [round(float(v), 3) for v in scores], 'mean': round(float(scores.mean()), 3), 'std': round(float(scores.std()), 3)})
    return render_template('index.html', active='cross-validation', cv_results=rows)


@app.route('/about')
def about(): return render_template('index.html', active='about')

@app.route('/health')
def health(): return jsonify({'status': 'running', 'project': 'Placement Prediction Analytics'})


@app.errorhandler(Exception)
def handle_error(error):
    return render_template('index.html', active='error', error=str(error)), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
