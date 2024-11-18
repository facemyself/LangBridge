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

def construct_prompt_ins(query, trg_lang_name):
    # return query
    prompt_no_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{query}\n({trg_lang_name})\n### Response:"
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
            # sample['prompt'] = construct_prompt_mt(sample['source'],
            #                                        sample['source_language'],
            #                                        sample['target_language'])
            return {
                'input': sample['source'],
                'output': sample['target']
            }
        elif self.training_stage == 2 or self.training_stage == 3:
            sample['prompt'] = construct_prompt_ins(sample['source'],
                                                   sample['target_language'])
            return {
                'input': sample['prompt'],
                'output': sample['target']
            }
        else:
            raise ValueError("error task")



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
        path_base = f'../data/bilingual_pairs/en-{train_name_map}'
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

def read_MulIn_EngOut_alpaca(train_num=100000):
    # languages = ['Bulgarian', 'Czech', 'German', 'English', 'Spanish', 'Finnish', 'French', 'Portuguese',
    #              'Russian', 'Chinese']
    languages = ['Bulgarian', 'German', 'English', 'Spanish', 'French', 'Portuguese',
                 'Russian', 'Chinese']
    dataset_train = []
    for train_name in languages:
        train_name_map = langs_map[train_name]
        path_base = f'../data/training-data'
        path_src = f'{path_base}/alpaca_data_cleaned.{train_name_map}.json'
        path_trg = f'{path_base}/alpaca_data_cleaned.en.json'
        sources = read_dataset(path_src)[:train_num]
        targets = read_dataset(path_trg)[:train_num]
        train_set = [(source, target) for source, target in zip(sources, targets)]
        for source, target in train_set:
            dataset_train.append({
                'source': source["instruction"] + "\n" + source["input"],
                'target': target["output"],
                'source_language': train_name,
                'target_language': 'English'
            })
    random.shuffle(dataset_train)
    return dataset_train


def read_MulIn_MulOut_alpaca(train_num=100000):
    # languages = ['Bulgarian', 'Czech', 'German', 'English', 'Spanish', 'Finnish', 'French', 'Portuguese',
    #              'Russian', 'Chinese']
    languages = ['German', 'English', 'Spanish', 'French', 'Portuguese',
                 'Russian', 'Chinese']
    dataset_train = []
    for train_name in languages:
        train_name_map = langs_map[train_name]
        path_base = f'../data/training-data/'
        path_src = f'{path_base}/alpaca_data_cleaned.{train_name_map}.json'
        #path_trg = f'{path_base}/alpaca_data_cleaned.en.json'
        train_set = read_dataset(path_src)[:train_num]
        #targets = read_dataset(path_trg)
        #train_set = [(source, target) for source, target in zip(sources, targets)]
        for sample in train_set:
            dataset_train.append({
                'source': sample["instruction"] + "\n" + sample["input"],
                'target': sample["output"],
                'source_language': train_name,
                'target_language': train_name
            })
    random.shuffle(dataset_train)
    return dataset_train
