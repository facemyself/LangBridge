import argparse
import json
import os
import random
import re

import pandas as pd


def convert_science_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "science_train.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, "science_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            fout.write(
                json.dumps(
                    {
                        "dataset": f"science.{example['dataset']}",
                        "id": f"science_{idx}",
                        "messages": [
                            {"role": "user", "content": example["input"]},
                            {"role": "assistant", "content": example["output"]},
                        ],
                    }
                )
                + "\n"
            )
