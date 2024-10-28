from torch.utils.data import Dataset
from datasets import load_dataset
import json
import random

langs_map = {'English': 'en', 'Swahili': 'sw', 'Chinese': 'zh', 'Bengali': 'bn',
                     'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja',
                     'Russian': 'ru', 'Thai': 'th','Telugu': 'te', 'Greek': 'el',
                     'Arabic': 'ar', 'Bulgarian': 'bg', 'Croatian': 'hr', 'Hungarian': 'hu',
                     'Italian': 'it', 'Lithuanian': 'lt', 'Macedonian': 'mk', 'Polish': 'pl',
                     'Portuguese': 'pt', 'Albanian': 'sq', 'Serbian': 'sr', 'Turkish': 'tr',
                     'Vietnamese': 'vi', 'Hindi': 'hi', 'Flemish': 'nl', 'Urdu': 'ur'}

def construct_prompt_mt(query, src_lang_name, trg_lang_name):
    instruction = f'Translate the following sentences from {src_lang_name} to {trg_lang_name}.'
    prompt_no_input = (
        'Below is an instruction that describes a task, paired with an input that provides further context. '
        'Write a response that appropriately completes the request.\n'
        f'### Instruction:\n{instruction}\n'
        f'### Input:\n{query}\n### Response:'
    )
    return prompt_no_input


class Data(Dataset):
    def __init__(self, dataset_path, split):
        super().__init__()
        if 'json' in dataset_path:
            self.ds = load_dataset(
                'json',
                data_files=dataset_path)['train']
        else:
            self.ds = load_dataset(dataset_path)[split]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x = self.ds[idx]
        return {
            'input': x['input'],
            'output': x['output']
        }
    
def read_dataset(path):
    if 'jsonl' in path:
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(json.loads(line))
    elif 'json' in path:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if isinstance(dataset, dict):
            if 'data' in dataset:
                dataset = dataset['data']
    else:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = f.readlines()
    return dataset


def read_lego(train_num=100000):
    languages = ['Bengali', 'Thai', 'Swahili', 'Japanese', 'Chinese', 'German', 'French', 'Russian',
                'Spanish']
    dataset_train = []
    for train_name in languages:
        train_name_map = langs_map[train_name]
        path_base = f'/data1/rzw/CODE/LangBridge/data/bilingual_pairs/en-{train_name_map}'
        path_src = f'{path_base}/train_100k.{train_name_map}'
        path_trg = f'{path_base}/train_100k.en'
        sources = read_dataset(path_src)[:train_num]
        targets = read_dataset(path_trg)[:train_num]
        train_set = [(source, target) for source, target in zip(sources, targets)]
        for source, target in train_set:
            dataset_train.append({
                'source': source,
                'target': target,
                'source_language': train_name,
                'target_language': 'English'
            })
    random.shuffle(dataset_train)
    return dataset_train


class MathDataset(Dataset):
    def __init__(self, dataset, training_stage: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.training_stage = training_stage
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.training_stage == 1:
            sample['prompt'] = construct_prompt_mt(sample['source'],
                                                   sample['source_language'],
                                                   sample['target_language'])
            return {
                'input': sample['prompt'],
                'output': sample['target']
            }
        else:
            raise ValueError("error task")