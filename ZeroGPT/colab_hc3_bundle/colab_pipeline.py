import ast
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
import pandas as pd
from gpt_zero.classical import BaselineTrainingConfig, ClassicalBaselineSuite, train_classical_baselines
from gpt_zero.gptzero_like import CausalLMPerplexityScorer, FeatureExtractionConfig, GPTZeroLikeDetector, ScorerConfig, train_gptzero_like_detector
from gpt_zero.hc3 import PrepareHC3Config, deduplicate_samples, normalize_hc3_rows, prepare_hc3_dataset
from gpt_zero.io_utils import dump_json, ensure_dir, read_table, reset_dir, timestamp_run_id, write_table
from gpt_zero.metrics import evaluate_predictions, fixed_fpr_metric_name, target_fpr_metric_name
from gpt_zero.schemas import ensure_sample_schema
from gpt_zero.tfidf import TfidfFeatureConfig
VALID_SCORE_SPLITS = ('train', 'val', 'test')
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
DATA_DIR = ARTIFACTS_DIR / 'data' / 'hc3'
BASELINE_MODEL_DIR = ARTIFACTS_DIR / 'models' / 'baselines'
GPTZERO_MODEL_DIR = ARTIFACTS_DIR / 'models' / 'gptzero_like'
BASELINE_RUN_DIR = ARTIFACTS_DIR / 'runs' / 'hc3_baselines_run'
GPTZERO_RUN_DIR = ARTIFACTS_DIR / 'runs' / 'hc3_gptzero_run'
REFERENCE_GPTZERO_METRICS_DIR = ARTIFACTS_DIR / 'runs' / 'hc3_gptzero_full_run' / 'metrics'
DEFAULT_COLAB_TARGET_FPRS = (0.01, 0.001, 0.0001)
TEST_DATASET_DIR = PROJECT_ROOT / 'test_dataset'
METRICS_SHARE_DIR = PROJECT_ROOT / 'metrics_share'
TEST_DATASET_SAMPLE_DIR = ARTIFACTS_DIR / 'data' / 'test_dataset_samples'
TEST_DATASET_RUN_DIR = ARTIFACTS_DIR / 'runs' / 'test_dataset'
FILTERED_TRAINING_DATA_DIR = ARTIFACTS_DIR / 'data' / 'hc3_without_test_dataset'
HC3_UNIFIED_TEST_FILENAME = 'hc3_unified_10000_seed42_clean_test.csv'
SHARED_METRIC_TARGETS = ((0.01, 'metrics_at_1pct_fpr'), (0.001, 'metrics_at_0.1pct_fpr'), (0.0001, 'metrics_at_0.01pct_fpr'))
SHARED_PRESENTATION_TARGET_FPR = 0.001
SHARED_PRESENTATION_TARGET_TPR = 0.8

def _normalize_score_splits(values):
    if not values:
        return list(VALID_SCORE_SPLITS)
    normalized = []
    for value in values:
        for item in str(value).split(','):
            candidate = item.strip().lower()
            if not candidate:
                continue
            if candidate not in VALID_SCORE_SPLITS:
                raise ValueError(f"Unsupported split '{candidate}'. Expected one of: {', '.join(VALID_SCORE_SPLITS)}")
            if candidate not in normalized:
                normalized.append(candidate)
    return normalized or list(VALID_SCORE_SPLITS)

def _resolve_split_path(data_dir, split, split_paths=None):
    if split_paths and split in split_paths:
        explicit_path = Path(split_paths[split])
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"Explicit dataset split path for '{split}' does not exist: {explicit_path}")
    data_dir = Path(data_dir)
    candidates = (data_dir / f'{split}.parquet', data_dir / f'{split}.csv', data_dir / f'{split}.jsonl', data_dir / f'{split}.json')
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find a dataset split for '{split}' under {data_dir}. Expected one of: {', '.join((path.name for path in candidates))}")

def find_split_path(data_dir, split, split_paths=None):
    return _resolve_split_path(data_dir, split, split_paths)

def _clean_scalar(value):
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass
    return str(value)

def _normalize_text_key(value):
    return ' '.join(_clean_scalar(value).split())

def _prompt_key(domain, prompt):
    return (_normalize_text_key(domain), _normalize_text_key(prompt))

def _decode_list_cell(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            decoded = loader(text)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(decoded, list):
            return [str(item) for item in decoded if str(item).strip()]
        return [str(decoded)]
    return [text]

def _dataset_name_from_path(path):
    stem = Path(path).stem.lower()
    characters = [character if character.isalnum() else '_' for character in stem]
    slug = ''.join(characters)
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug.strip('_') or 'dataset'

def _load_hc3_unified_rows(path):
    source = Path(path)
    frame = pd.read_csv(source)
    required_columns = {'source', 'question', 'human_answers', 'chatgpt_answers'}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f'{source} is missing required HC3 unified columns: {missing}')
    rows = []
    for row in frame.to_dict(orient='records'):
        rows.append({**row, 'source': _clean_scalar(row.get('source')), 'question': _clean_scalar(row.get('question')), 'human_answers': _decode_list_cell(row.get('human_answers')), 'chatgpt_answers': _decode_list_cell(row.get('chatgpt_answers'))})
    return rows

def _candidate_hc3_unified_sources():
    return [PROJECT_ROOT / 'HC3-Dataset-Samples' / 'hc3_unified_10000_seed42_clean.csv', PROJECT_ROOT.parent / 'HC3-Dataset-Samples' / 'hc3_unified_10000_seed42_clean.csv', PROJECT_ROOT.parent.parent / 'HC3-Dataset-Samples' / 'hc3_unified_10000_seed42_clean.csv']

def ensure_hc3_unified_test_file(*, source_file=None, test_dataset_dir=TEST_DATASET_DIR, overwrite=False):
    destination = ensure_dir(test_dataset_dir) / HC3_UNIFIED_TEST_FILENAME
    if destination.exists() and (not overwrite):
        return destination
    candidates = [Path(source_file)] if source_file else []
    candidates.extend(_candidate_hc3_unified_sources())
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, destination)
            return destination
    raise FileNotFoundError(f"Could not find hc3_unified_10000_seed42_clean.csv to copy into test_dataset. Checked: {', '.join((str(path) for path in candidates))}")

def _heldout_prompt_keys_from_test_dataset(test_dataset_dir):
    dataset_dir = Path(test_dataset_dir)
    keys = set()
    for csv_path in sorted(dataset_dir.glob('*.csv')):
        frame = pd.read_csv(csv_path, usecols=lambda column: column in {'source', 'question'})
        if {'source', 'question'}.difference(frame.columns):
            continue
        keys.update((_prompt_key(row['source'], row['question']) for row in frame.to_dict(orient='records')))
    if not keys:
        raise RuntimeError(f'No HC3 unified test prompts were found in {dataset_dir}')
    return keys

def prepare_training_data_without_test_dataset(*, data_dir=DATA_DIR, test_dataset_dir=TEST_DATASET_DIR, output_dir=FILTERED_TRAINING_DATA_DIR, filter_splits=('train', 'val')):
    heldout_keys = _heldout_prompt_keys_from_test_dataset(test_dataset_dir)
    output = reset_dir(output_dir, preserve_names=('.gitkeep',))
    split_paths = {}
    split_counts = {}
    for split in VALID_SCORE_SPLITS:
        try:
            source_path = _resolve_split_path(data_dir, split)
        except FileNotFoundError:
            continue
        frame = read_table(source_path)
        original_rows = int(len(frame))
        if split in filter_splits:
            row_keys = frame.apply(lambda row: _prompt_key(row.get('domain'), row.get('prompt')), axis=1)
            keep_mask = ~row_keys.isin(heldout_keys)
            frame = frame.loc[keep_mask].reset_index(drop=True)
        else:
            frame = frame.reset_index(drop=True)
        if split in filter_splits and frame.empty and (original_rows > 0):
            raise RuntimeError(f'Filtering test_dataset prompts removed every row from the {split} split. Use a larger non-overlapping training source before retraining detectors.')
        destination = output / f'{split}.csv'
        write_table(frame, destination)
        split_paths[split] = str(destination)
        split_counts[split] = {'source_rows': original_rows, 'output_rows': int(len(frame)), 'removed_rows': original_rows - int(len(frame))}
    manifest = {'data_dir': str(data_dir), 'test_dataset_dir': str(test_dataset_dir), 'heldout_prompt_count': len(heldout_keys), 'filter_splits': list(filter_splits), 'split_counts': split_counts, 'split_paths': split_paths}
    dump_json(manifest, output / 'manifest.json')
    return manifest

def prepare_test_dataset_files(*, test_dataset_dir=TEST_DATASET_DIR, output_dir=TEST_DATASET_SAMPLE_DIR):
    test_dataset_dir = Path(test_dataset_dir)
    output = reset_dir(output_dir, preserve_names=('.gitkeep',))
    datasets = []
    for compact_path in sorted(test_dataset_dir.glob('*.csv')):
        dataset_name = _dataset_name_from_path(compact_path)
        rows = _load_hc3_unified_rows(compact_path)
        sample_frame = normalize_hc3_rows(rows, dataset_name=dataset_name)
        sample_frame, deduplication = deduplicate_samples(sample_frame)
        sample_frame = ensure_sample_schema(sample_frame)
        sample_frame['split'] = 'test'
        sample_frame = sample_frame.sort_values(['domain', 'label', 'sample_id']).reset_index(drop=True)
        dataset_dir = ensure_dir(output / dataset_name)
        sample_path = dataset_dir / 'test.csv'
        write_table(sample_frame, sample_path)
        datasets.append({'dataset_name': dataset_name, 'dataset_used': compact_path.name, 'compact_path': str(compact_path), 'sample_path': str(sample_path), 'num_samples': int(len(sample_frame)), 'deduplication': deduplication})
    if not datasets:
        raise FileNotFoundError(f'No CSV test datasets found in {test_dataset_dir}')
    manifest = {'test_dataset_dir': str(test_dataset_dir), 'output_dir': str(output), 'datasets': datasets}
    dump_json(manifest, output / 'manifest.json')
    return manifest

def _finite_float(value, default=0.0):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default

def _finite_float_or_none(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None

def _shared_model_metadata(detector_name, config):
    if detector_name == 'gptzero_like':
        additional_models = [str(config.lm_model)] if config is not None else []
        return ('gptzero_like_logistic_regression', additional_models)
    if detector_name == 'svm_tfidf':
        return ('linear_svm_tfidf', [])
    if detector_name == 'xgboost_tfidf':
        return ('xgboost_tfidf', [])
    return (detector_name, [])

def _shared_operating_point_from_roc(roc_group, target_fpr):
    if roc_group is None or roc_group.empty:
        return None
    valid = roc_group.loc[pd.to_numeric(roc_group['fpr'], errors='coerce') <= target_fpr].copy()
    if valid.empty:
        return None
    valid['tpr'] = pd.to_numeric(valid['tpr'], errors='coerce')
    valid['fpr'] = pd.to_numeric(valid['fpr'], errors='coerce')
    valid = valid.dropna(subset=['fpr', 'tpr'])
    if valid.empty:
        return None
    best_tpr = valid['tpr'].max()
    return valid.loc[valid['tpr'].sub(best_tpr).abs() <= 1e-12].sort_values('fpr').iloc[-1]

def _shared_fixed_fpr_values_from_roc(row, roc_group, target_fpr):
    operating_point = _shared_operating_point_from_roc(roc_group, target_fpr)
    if operating_point is None:
        return {}
    positives = _finite_float(row.get('fn')) + _finite_float(row.get('tp'))
    negatives = _finite_float(row.get('tn')) + _finite_float(row.get('fp'))
    if positives <= 0 or negatives < 0:
        return {}
    actual_tpr = _finite_float(operating_point.get('tpr'))
    actual_fpr = _finite_float(operating_point.get('fpr'))
    tp = actual_tpr * positives
    fp = actual_fpr * negatives
    fn = positives - tp
    tn = negatives - fp
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / positives if positives > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / (positives + negatives) if positives + negatives > 0 else 0.0
    return {fixed_fpr_metric_name('threshold', target_fpr): _finite_float(operating_point.get('threshold')), fixed_fpr_metric_name('actual_fpr', target_fpr): actual_fpr, target_fpr_metric_name(target_fpr): recall, fixed_fpr_metric_name('accuracy', target_fpr): accuracy, fixed_fpr_metric_name('precision', target_fpr): precision, fixed_fpr_metric_name('recall', target_fpr): recall, fixed_fpr_metric_name('f1', target_fpr): f1}

def _shared_metric_value(row, column, fallback_values):
    value = _finite_float_or_none(row.get(column))
    if value is not None:
        return value
    return _finite_float(fallback_values.get(column))

def _shared_metrics_block(row, target_fpr, roc_group=None):
    fallback_values = _shared_fixed_fpr_values_from_roc(row, roc_group, target_fpr)
    recall = _shared_metric_value(row, fixed_fpr_metric_name('recall', target_fpr), fallback_values)
    return {'f1': _shared_metric_value(row, fixed_fpr_metric_name('f1', target_fpr), fallback_values), 'accuracy': _shared_metric_value(row, fixed_fpr_metric_name('accuracy', target_fpr), fallback_values), 'precision': _shared_metric_value(row, fixed_fpr_metric_name('precision', target_fpr), fallback_values), 'recall': recall, 'tpr': recall, 'auc_roc': _finite_float(row.get('roc_auc'))}

def _shared_fpr_at_target_tpr(roc_group, target_tpr):
    if roc_group is None or roc_group.empty:
        return 0.0
    frame = roc_group.copy()
    frame['tpr'] = pd.to_numeric(frame['tpr'], errors='coerce')
    frame['fpr'] = pd.to_numeric(frame['fpr'], errors='coerce')
    frame = frame.dropna(subset=['fpr', 'tpr'])
    valid = frame.loc[frame['tpr'] >= target_tpr]
    if valid.empty:
        return 0.0
    sort_columns = ['fpr']
    ascending = [True]
    if 'threshold' in valid.columns:
        sort_columns.append('threshold')
        ascending.append(False)
    return _finite_float(valid.sort_values(sort_columns, ascending=ascending).iloc[0]['fpr'])

def _shared_presentation_metrics_block(row, roc_group):
    fixed_fpr_values = _shared_fixed_fpr_values_from_roc(row, roc_group, SHARED_PRESENTATION_TARGET_FPR)
    tpr_column = target_fpr_metric_name(SHARED_PRESENTATION_TARGET_FPR)
    tpr_at_target_fpr = _shared_metric_value(row, tpr_column, fixed_fpr_values)
    return {'tpr_at_0_1pct_fpr': tpr_at_target_fpr, 'fpr_at_80pct_tpr': _shared_fpr_at_target_tpr(roc_group, SHARED_PRESENTATION_TARGET_TPR), 'roc_auc': _finite_float(row.get('roc_auc'))}

def export_metrics_share_json(*, metrics_dir, metrics_share_dir=METRICS_SHARE_DIR, experiment_name, dataset_used, config=None):
    metrics_dir = Path(metrics_dir)
    destination = ensure_dir(metrics_share_dir)
    summary_path = metrics_dir / 'metrics_summary.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing metrics summary: {summary_path}')
    summary = pd.read_csv(summary_path)
    summary = summary.loc[summary['split'].astype(str) == 'test'].copy()
    roc_path = metrics_dir / 'roc_points.csv'
    roc = pd.read_csv(roc_path) if roc_path.exists() else pd.DataFrame()
    exported = []
    for _, row in summary.iterrows():
        detector_name = str(row['detector_name'])
        split = str(row.get('split', 'test'))
        roc_group = None
        if not roc.empty:
            roc_group = roc.loc[(roc['detector_name'].astype(str) == detector_name) & (roc['split'].astype(str) == split)]
        model_used, additional_models = _shared_model_metadata(detector_name, config)
        payload = {'experiment_name': f'{experiment_name}_{detector_name}', 'detection_method': detector_name, 'model_used': model_used, 'dataset_used': dataset_used, 'num_samples': int(row.get('num_samples', 0) or 0), 'additional_details': {'additional_models_used': additional_models, 'notes': 'Exported from ZeroGPT Colab evaluation on test_dataset.'}}
        for target_fpr, key in SHARED_METRIC_TARGETS:
            payload[key] = _shared_metrics_block(row, target_fpr, roc_group)
        payload['presentation_metrics'] = _shared_presentation_metrics_block(row, roc_group)
        output_path = destination / f'{experiment_name}_{detector_name}.json'
        dump_json(payload, output_path)
        exported.append({'path': str(output_path), 'payload': payload})
    return exported

def _shared_roc_group_for_row(roc, row):
    if roc.empty:
        return roc
    mask = pd.Series(True, index=roc.index)
    for column in ('dataset_name', 'dataset_used', 'detector_name', 'split'):
        if column in roc.columns and column in row.index:
            mask &= roc[column].astype(str).eq(str(row.get(column)))
    return roc.loc[mask]

def rewrite_metrics_share_jsons(*, metrics_share_dir=METRICS_SHARE_DIR, config=None, summary_filename='all_test_dataset_metrics_summary.csv', roc_filename='all_test_dataset_roc_points.csv'):
    """Rewrite metrics_share JSONs from existing combined CSV outputs.

    This does not score models or regenerate predictions. It only reads the
    combined metrics summary and ROC point files already present in
    metrics_share, then overwrites the per-dataset JSON exports.
    """
    destination = ensure_dir(metrics_share_dir)
    summary_path = destination / summary_filename
    roc_path = destination / roc_filename
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing combined metrics summary: {summary_path}')
    if not roc_path.exists():
        raise FileNotFoundError(f'Missing combined ROC points: {roc_path}')
    summary = pd.read_csv(summary_path)
    summary = summary.loc[summary['split'].astype(str) == 'test'].copy()
    roc = pd.read_csv(roc_path)
    rewritten = []
    for _, row in summary.iterrows():
        detector_name = _clean_scalar(row.get('detector_name'))
        dataset_used = _clean_scalar(row.get('dataset_used'))
        dataset_name = _clean_scalar(row.get('dataset_name')) or _dataset_name_from_path(dataset_used)
        output_path = destination / f'{dataset_name}_{detector_name}.json'
        existing_payload = {}
        if output_path.exists():
            existing_payload = json.loads(output_path.read_text(encoding='utf-8'))
        model_used, additional_models = _shared_model_metadata(detector_name, config)
        existing_details = existing_payload.get('additional_details', {}) if isinstance(existing_payload, dict) else {}
        if not additional_models:
            additional_models = list(existing_details.get('additional_models_used', []))
        roc_group = _shared_roc_group_for_row(roc, row)
        payload = {'experiment_name': f'{dataset_name}_{detector_name}', 'detection_method': detector_name, 'model_used': existing_payload.get('model_used', model_used) if isinstance(existing_payload, dict) else model_used, 'dataset_used': dataset_used, 'num_samples': int(_finite_float(row.get('num_samples'))), 'additional_details': {'additional_models_used': additional_models, 'notes': existing_details.get('notes', 'Exported from ZeroGPT Colab evaluation on test_dataset.')}}
        for target_fpr, key in SHARED_METRIC_TARGETS:
            payload[key] = _shared_metrics_block(row, target_fpr, roc_group)
        payload['presentation_metrics'] = _shared_presentation_metrics_block(row, roc_group)
        dump_json(payload, output_path)
        rewritten.append({'path': str(output_path), 'payload': payload})
    manifest = {'metrics_share_dir': str(destination), 'summary_path': str(summary_path), 'roc_path': str(roc_path), 'rewritten_metric_files': [item['path'] for item in rewritten]}
    dump_json(manifest, destination / 'rewrite_metrics_share_jsons_manifest.json')
    return manifest

class ColabExperimentConfig:

    def __init__(self, data_dir=DATA_DIR, split_paths=None, baseline_model_dir=BASELINE_MODEL_DIR, gptzero_model_dir=GPTZERO_MODEL_DIR, baseline_run_dir=BASELINE_RUN_DIR, gptzero_run_dir=GPTZERO_RUN_DIR, lm_model='gpt2', device='cuda', local_files_only=False, row_batch_size=64, perplexity_batch_size=16, stride=512, max_length=None, max_sentences_per_text=None, score_splits=('test',), target_fpr=0.01, target_fprs=DEFAULT_COLAB_TARGET_FPRS, word_max_features=20000, char_max_features=15000, min_df=2, svm_c=1.0, xgb_batch_size=1024, xgb_estimators=120, xgb_max_depth=4, xgb_learning_rate=0.1, xgb_subsample=0.8, xgb_colsample_bytree=0.8, xgb_device='cuda', xgb_early_stopping_rounds=20, xgb_eval_log_interval=10):
        self.data_dir = data_dir
        self.split_paths = split_paths
        self.baseline_model_dir = baseline_model_dir
        self.gptzero_model_dir = gptzero_model_dir
        self.baseline_run_dir = baseline_run_dir
        self.gptzero_run_dir = gptzero_run_dir
        self.lm_model = lm_model
        self.device = device
        self.local_files_only = local_files_only
        self.row_batch_size = row_batch_size
        self.perplexity_batch_size = perplexity_batch_size
        self.stride = stride
        self.max_length = max_length
        self.max_sentences_per_text = max_sentences_per_text
        self.score_splits = score_splits
        self.target_fpr = target_fpr
        self.target_fprs = target_fprs
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.min_df = min_df
        self.svm_c = svm_c
        self.xgb_batch_size = xgb_batch_size
        self.xgb_estimators = xgb_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_subsample = xgb_subsample
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.xgb_device = xgb_device
        self.xgb_early_stopping_rounds = xgb_early_stopping_rounds
        self.xgb_eval_log_interval = xgb_eval_log_interval

def project_paths():
    return {'project_root': str(PROJECT_ROOT), 'data_dir': str(DATA_DIR), 'baseline_model_dir': str(BASELINE_MODEL_DIR), 'gptzero_model_dir': str(GPTZERO_MODEL_DIR), 'baseline_run_dir': str(BASELINE_RUN_DIR), 'gptzero_run_dir': str(GPTZERO_RUN_DIR), 'reference_gptzero_metrics_dir': str(REFERENCE_GPTZERO_METRICS_DIR), 'test_dataset_dir': str(TEST_DATASET_DIR), 'metrics_share_dir': str(METRICS_SHARE_DIR), 'filtered_training_data_dir': str(FILTERED_TRAINING_DATA_DIR), 'notebook': str(PROJECT_ROOT / 'hc3_colab_workflow.ipynb')}

def load_hc3_manifest():
    manifest_path = DATA_DIR / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'No HC3 manifest found at {manifest_path}')
    import json
    return json.loads(manifest_path.read_text(encoding='utf-8'))

def rebuild_hc3_from_source(output_dir, *, input_file=None, hf_dataset='Hello-SimpleAI/HC3', test_size=0.2, val_size=0.1, random_state=42, max_samples_per_label=None):
    config = PrepareHC3Config(output_dir=Path(output_dir), input_file=Path(input_file) if input_file else None, hf_dataset=hf_dataset, test_size=test_size, val_size=val_size, random_state=random_state, max_samples_per_label=max_samples_per_label, group_by_prompt=True, deduplicate_texts=True)
    return prepare_hc3_dataset(config)

def train_baselines(config):
    feature_config = TfidfFeatureConfig(word_max_features=config.word_max_features, char_max_features=config.char_max_features, min_df=config.min_df)
    training_config = BaselineTrainingConfig(svm_c=config.svm_c, batch_size=config.row_batch_size, xgb_batch_size=config.xgb_batch_size, xgb_estimators=config.xgb_estimators, xgb_max_depth=config.xgb_max_depth, xgb_learning_rate=config.xgb_learning_rate, xgb_subsample=config.xgb_subsample, xgb_colsample_bytree=config.xgb_colsample_bytree, xgb_device=config.xgb_device, xgb_early_stopping_rounds=config.xgb_early_stopping_rounds, xgb_eval_log_interval=config.xgb_eval_log_interval)
    train_source = _resolve_split_path(config.data_dir, 'train', config.split_paths)
    try:
        val_source = _resolve_split_path(config.data_dir, 'val', config.split_paths)
    except FileNotFoundError:
        if config.split_paths and 'val' in config.split_paths:
            raise
        val_source = None
    return train_classical_baselines(train_source=train_source, model_dir=config.baseline_model_dir, feature_config=feature_config, training_config=training_config, val_source=val_source)

def train_gptzero(config):
    scorer_config = ScorerConfig(model_name=config.lm_model, device=config.device, stride=config.stride, max_length=config.max_length, local_files_only=config.local_files_only, perplexity_batch_size=config.perplexity_batch_size)
    feature_config = FeatureExtractionConfig(max_sentences_per_text=config.max_sentences_per_text)
    train_source = _resolve_split_path(config.data_dir, 'train', config.split_paths)
    try:
        val_source = _resolve_split_path(config.data_dir, 'val', config.split_paths)
    except FileNotFoundError:
        if config.split_paths and 'val' in config.split_paths:
            raise
        val_source = None
    return train_gptzero_like_detector(train_source=train_source, val_source=val_source, model_dir=config.gptzero_model_dir, scorer_config=scorer_config, feature_config=feature_config, batch_size=config.row_batch_size)

def score_models(*, data_dir=DATA_DIR, split_paths=None, run_dir, run_id=None, baseline_model_dir=None, gptzero_model_dir=None, batch_size=64, score_splits=('test',)):
    data_dir = Path(data_dir)
    run_dir = reset_dir(run_dir, preserve_names=('.gitkeep',))
    predictions_dir = ensure_dir(run_dir / 'predictions')
    requested_splits = _normalize_score_splits(score_splits)
    run_id = run_id or timestamp_run_id('colab_run')
    baselines_suite = None
    if baseline_model_dir is not None and Path(baseline_model_dir).exists():
        baselines_suite = ClassicalBaselineSuite.load(baseline_model_dir)
    gptzero_detector = None
    gptzero_scorer = None
    gptzero_cache_dir = None
    if gptzero_model_dir is not None and (Path(gptzero_model_dir) / 'gptzero_like.joblib').exists():
        gptzero_detector = GPTZeroLikeDetector.load(gptzero_model_dir)
        gptzero_scorer = CausalLMPerplexityScorer(gptzero_detector.scorer_config)
        gptzero_cache_dir = Path(gptzero_model_dir) / 'feature_cache'
    if baselines_suite is None and gptzero_detector is None:
        raise RuntimeError('No detectors were found. Provide a trained baselines directory and/or GPTZero-like model directory.')
    scored_splits = []
    for split in requested_splits:
        try:
            split_path = _resolve_split_path(data_dir, split, split_paths)
        except FileNotFoundError:
            if split_paths and split in split_paths:
                raise
            continue
        split_predictions = []
        if gptzero_detector is not None:
            feature_frame = gptzero_detector.build_feature_frame(split_path, scorer=gptzero_scorer, batch_size=batch_size, progress_label=f'gptzero-{split}', cache_dir=gptzero_cache_dir)
            split_predictions.append(gptzero_detector.predict_from_features(feature_frame, run_id))
        if baselines_suite is not None:
            split_predictions.append(baselines_suite.predict(split_path, run_id, batch_size=batch_size))
        combined = pd.concat(split_predictions, ignore_index=True)
        write_table(combined, predictions_dir / f'{split}.parquet')
        scored_splits.append(split)
    dump_json({'run_id': run_id, 'data_dir': str(data_dir), 'split_paths': {key: str(value) for key, value in split_paths.items()} if split_paths else None, 'baseline_model_dir': str(baseline_model_dir) if baseline_model_dir else None, 'gptzero_model_dir': str(gptzero_model_dir) if gptzero_model_dir else None, 'score_splits': scored_splits}, run_dir / 'run_config.json')
    return {'run_id': run_id, 'predictions_dir': str(predictions_dir), 'score_splits': scored_splits}

def evaluate_run(*, data_dir=DATA_DIR, split_paths=None, predictions_dir, output_dir, target_fpr=0.01, target_fprs=None):
    sample_frames = []
    prediction_frames = []
    for split in VALID_SCORE_SPLITS:
        try:
            sample_path = _resolve_split_path(data_dir, split, split_paths)
        except FileNotFoundError:
            if split_paths and split in split_paths:
                raise
            sample_path = None
        prediction_path = Path(predictions_dir) / f'{split}.parquet'
        if sample_path is not None and sample_path.exists():
            sample_frames.append(read_table(sample_path))
        if prediction_path.exists():
            prediction_frames.append(read_table(prediction_path))
    if not sample_frames:
        raise RuntimeError(f'No dataset splits found under {data_dir}')
    if not prediction_frames:
        raise RuntimeError(f'No prediction files found under {predictions_dir}')
    samples = pd.concat(sample_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return evaluate_predictions(samples, predictions, output_dir, target_fpr=target_fpr, target_fprs=target_fprs)

def run_test_dataset_evaluations(*, config, test_dataset_dir=TEST_DATASET_DIR, prepared_data_dir=TEST_DATASET_SAMPLE_DIR, run_root=TEST_DATASET_RUN_DIR, metrics_share_dir=METRICS_SHARE_DIR, batch_size=None, reset_metrics_share=True):
    prepared_manifest = prepare_test_dataset_files(test_dataset_dir=test_dataset_dir, output_dir=prepared_data_dir)
    run_root = reset_dir(run_root, preserve_names=('.gitkeep',))
    metrics_share_dir = reset_dir(metrics_share_dir, preserve_names=('.gitkeep',)) if reset_metrics_share else ensure_dir(metrics_share_dir)
    exported_json = []
    summary_frames = []
    roc_frames = []
    dataset_results = []
    for dataset_info in prepared_manifest['datasets']:
        dataset_name = dataset_info['dataset_name']
        dataset_used = dataset_info['dataset_used']
        sample_path = Path(dataset_info['sample_path'])
        dataset_run_dir = run_root / dataset_name
        scoring = score_models(data_dir=sample_path.parent, split_paths={'test': sample_path}, run_dir=dataset_run_dir, baseline_model_dir=config.baseline_model_dir, gptzero_model_dir=config.gptzero_model_dir, batch_size=batch_size or config.row_batch_size, score_splits=('test',))
        metrics_dir = dataset_run_dir / 'metrics'
        evaluation = evaluate_run(data_dir=sample_path.parent, split_paths={'test': sample_path}, predictions_dir=Path(scoring['predictions_dir']), output_dir=metrics_dir, target_fpr=config.target_fpr, target_fprs=config.target_fprs)
        shared_exports = export_metrics_share_json(metrics_dir=metrics_dir, metrics_share_dir=metrics_share_dir, experiment_name=dataset_name, dataset_used=dataset_used, config=config)
        exported_json.extend(shared_exports)
        summary_path = metrics_dir / 'metrics_summary.csv'
        roc_path = metrics_dir / 'roc_points.csv'
        if summary_path.exists():
            summary_frame = pd.read_csv(summary_path)
            summary_frame['dataset_name'] = dataset_name
            summary_frame['dataset_used'] = dataset_used
            summary_frames.append(summary_frame)
        if roc_path.exists():
            roc_frame = pd.read_csv(roc_path)
            roc_frame['dataset_name'] = dataset_name
            roc_frame['dataset_used'] = dataset_used
            roc_frames.append(roc_frame)
        dataset_results.append({'dataset_name': dataset_name, 'dataset_used': dataset_used, 'sample_path': str(sample_path), 'run_dir': str(dataset_run_dir), 'metrics_dir': str(metrics_dir), 'scoring': scoring, 'evaluation': evaluation, 'shared_metric_files': [item['path'] for item in shared_exports]})
    combined_summary_path = metrics_share_dir / 'all_test_dataset_metrics_summary.csv'
    combined_roc_path = metrics_share_dir / 'all_test_dataset_roc_points.csv'
    if summary_frames:
        write_table(pd.concat(summary_frames, ignore_index=True), combined_summary_path)
    if roc_frames:
        write_table(pd.concat(roc_frames, ignore_index=True), combined_roc_path)
    manifest = {'test_dataset_dir': str(test_dataset_dir), 'prepared_data_dir': str(prepared_data_dir), 'run_root': str(run_root), 'metrics_share_dir': str(metrics_share_dir), 'combined_summary_path': str(combined_summary_path), 'combined_roc_path': str(combined_roc_path), 'datasets': dataset_results, 'shared_metric_files': [item['path'] for item in exported_json]}
    dump_json(manifest, metrics_share_dir / 'manifest.json')
    return manifest

def run_baseline_reference(*, config=None, retrain=False, score_splits=('test',)):
    config = config or ColabExperimentConfig()
    if retrain or not (Path(config.baseline_model_dir) / 'metadata.json').exists():
        training = train_baselines(config)
    else:
        training = {'used_existing_models': True, 'model_dir': str(config.baseline_model_dir)}
    try:
        scoring = score_models(data_dir=config.data_dir, split_paths=config.split_paths, run_dir=config.baseline_run_dir, baseline_model_dir=config.baseline_model_dir, gptzero_model_dir=None, batch_size=config.row_batch_size, score_splits=score_splits)
    except Exception as exc:
        message = str(exc)
        if not retrain and ('InconsistentVersionWarning' in message or 'multi_class' in message or 'unpickle estimator' in message):
            raise RuntimeError('The saved baseline scikit-learn artifacts are incompatible with the current Colab environment. Reinstall the pinned requirements and rerun the notebook from the top. If the mismatch persists, run the baseline section with retrain=True so fresh models are created in the current environment.') from exc
        raise
    evaluation = evaluate_run(data_dir=config.data_dir, split_paths=config.split_paths, predictions_dir=Path(scoring['predictions_dir']), output_dir=Path(config.baseline_run_dir) / 'metrics', target_fpr=config.target_fpr, target_fprs=config.target_fprs)
    return {'training': training, 'scoring': scoring, 'evaluation': evaluation}

def run_gptzero_experiment(*, config=None, train_model=True, score_splits=('test',)):
    config = config or ColabExperimentConfig()
    if train_model:
        training = train_gptzero(config)
    else:
        training = {'used_existing_model': True, 'model_dir': str(config.gptzero_model_dir)}
    scoring = score_models(data_dir=config.data_dir, split_paths=config.split_paths, run_dir=config.gptzero_run_dir, baseline_model_dir=None, gptzero_model_dir=config.gptzero_model_dir, batch_size=config.row_batch_size, score_splits=score_splits)
    evaluation = evaluate_run(data_dir=config.data_dir, split_paths=config.split_paths, predictions_dir=Path(scoring['predictions_dir']), output_dir=Path(config.gptzero_run_dir) / 'metrics', target_fpr=config.target_fpr, target_fprs=config.target_fprs)
    return {'training': training, 'scoring': scoring, 'evaluation': evaluation}

def load_metrics(metrics_dir):
    metrics_dir = Path(metrics_dir)
    outputs = {}
    for name in ('metrics_summary.csv', 'metrics_by_domain.csv', 'roc_points.csv'):
        path = metrics_dir / name
        outputs[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return outputs

def load_reference_metrics():
    return {'baseline': load_metrics(BASELINE_RUN_DIR / 'metrics'), 'gptzero': load_metrics(REFERENCE_GPTZERO_METRICS_DIR)}

def compare_reference_summaries():
    baseline_summary = pd.read_csv(BASELINE_RUN_DIR / 'metrics' / 'metrics_summary.csv')
    gptzero_summary = pd.read_csv(REFERENCE_GPTZERO_METRICS_DIR / 'metrics_summary.csv')
    combined = pd.concat([baseline_summary, gptzero_summary], ignore_index=True)
    return combined.sort_values(['split', 'detector_name']).reset_index(drop=True)
__all__ = ['ARTIFACTS_DIR', 'BASELINE_MODEL_DIR', 'BASELINE_RUN_DIR', 'PROJECT_ROOT', 'ColabExperimentConfig', 'DATA_DIR', 'DEFAULT_COLAB_TARGET_FPRS', 'FILTERED_TRAINING_DATA_DIR', 'GPTZERO_MODEL_DIR', 'GPTZERO_RUN_DIR', 'METRICS_SHARE_DIR', 'REFERENCE_GPTZERO_METRICS_DIR', 'TEST_DATASET_DIR', 'TEST_DATASET_RUN_DIR', 'TEST_DATASET_SAMPLE_DIR', 'project_paths', 'compare_reference_summaries', 'ensure_hc3_unified_test_file', 'evaluate_run', 'export_metrics_share_json', 'find_split_path', 'load_hc3_manifest', 'load_metrics', 'load_reference_metrics', 'prepare_test_dataset_files', 'prepare_training_data_without_test_dataset', 'rebuild_hc3_from_source', 'rewrite_metrics_share_jsons', 'run_baseline_reference', 'run_gptzero_experiment', 'run_test_dataset_evaluations', 'score_models', 'train_baselines', 'train_gptzero']
