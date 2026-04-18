import random
import datasets
import csv
import json

SEPARATOR = '<<<SEP>>>'


DATASETS = [
    'writing',
    'english',
    'german',
    'pubmed',
    'hc3_all',
    'hc3_all_10000',
    'hc3_finance',
    'hc3_medicine',
    'hc3_qa',
    'hc3_eli5',
    'hc3_csai',
]


def load_pubmed(cache_dir):
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(cache_dir=None):
    writing_path = 'data/writingPrompts'
    
    with open(f'{writing_path}/valid.wp_source', 'r', encoding='utf-8', errors='replace') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r', encoding='utf-8', errors='replace') as f:
        stories = f.readlines()
    
    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered


def _parse_hc3_answers(raw_answers):
    try:
        parsed = json.loads(raw_answers)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, list):
        return [x.strip() for x in parsed if isinstance(x, str) and x.strip()]
    return []


def _load_hc3_csv(hc3_path):

    examples = []
    with open(hc3_path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = (row.get('question') or '').strip()
            answers = _parse_hc3_answers(row.get('human_answers') or '[]')

            for answer in answers:
                if question:
                    text = process_spaces(f'Question: {question} Answer: {answer}')
                else:
                    text = process_spaces(answer)
                if text:
                    examples.append(text)

    random.seed(0)
    random.shuffle(examples)

    return examples


def load_hc3_all(cache_dir=None):
    return _load_hc3_csv('data/hc3/hc3_unified_1000_seed42.csv')


def load_hc3_all_10000(cache_dir=None):
    return _load_hc3_csv('data/hc3/hc3_unified_10000_seed42.csv')


def load_hc3_finance(cache_dir=None):
    return _load_hc3_csv('data/hc3/hc3_finance_200_seed42.csv')


def load_hc3_medicine(cache_dir=None):
    return _load_hc3_csv('data/hc3/hc3_medicine_200_seed42.csv')


def load_hc3_qa(cache_dir=None):
    return _load_hc3_csv('data/hc3/hc3_open_qa_200_seed42.csv')


def load_hc3_eli5(cache_dir=None):
    return _load_hc3_csv('data/hc3/hc3_reddit_eli5_200_seed42.csv')


def load_hc3_csai(cache_dir=None):
    return _load_hc3_csv('data/hc3/hc3_wiki_csai_200_seed42.csv')


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = datasets.load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')