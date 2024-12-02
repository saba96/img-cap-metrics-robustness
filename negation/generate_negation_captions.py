import os
import json
import argparse

import numpy as np

import matplotlib.pyplot as plt

from collections import defaultdict

yes_no_qids = []

vqa = None


def load_vqa(split, data_dir):
    assert split in ['val']

    # Downloaded from https://visualqa.org/download.html
    with open(os.path.join(data_dir, f'vqa/v2_OpenEnded_mscoco_{split}2014_questions.json')) as f:
        questions = json.load(f)['questions']
    with open(os.path.join(data_dir, f'vqa/v2_mscoco_{split}2014_annotations.json')) as f:
        annotations = {ann['question_id']: ann for ann in json.load(f)['annotations']}
    with open(os.path.join(data_dir, f'coco/annotations_trainval2014/instances_{split}2014.json')) as f:
        images = {image['id']: image for image in json.load(f)['images']}

    samples = []
    for q in questions:
        question_id = q['question_id']
        image_id = q['image_id']
        image_file = os.path.join(data_dir, f'coco/{split}2014', images[image_id]['file_name'])
        question = q['question']
        answers = [ans['answer'] for ans in annotations[question_id]['answers']]
        answer_type = annotations[question_id]['answer_type']
        if answer_type == 'yes/no':
            yes_no_qids.append(question_id)    
        samples.append({'question_id': question_id, 'image_id': image_id, 'image_file': image_file, 'question': question, 'answers':answers, 'answer_type': answer_type})

    return samples

def filter_yes_no(result):
    filtered = []
    ids = []
    for res in result:
        if res['question_id'] in yes_no_qids:
            filtered.append(res)
            ids.append(res['question_id'])
    return filtered, ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/network/scratch/s/saba.ahmadi/data')
    args = parser.parse_args()

    print("Arguments:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")

    qid_answers = defaultdict(list)
    vqa = load_vqa('val', args.data_dir)
    for sample in vqa:
        qid_answers[sample['question_id']] = sample['answers']
    with open('../dataset/negation/correct_captions.json') as f:
        result = json.load(f)
    correct, correct_ids = filter_yes_no(result)
    print(len(correct))
    with open('../dataset/negation/preprocessed_negated_yes_no.json') as f:
         result = json.load(f)
    print(len(result))
    negated, negated_ids = filter_yes_no(result)   
    with open('../dataset/negation/correct_yes_no.json', 'w') as f:
       json.dump(correct, f)  
    with open('../dataset/negation/negated_correct_yes_no.json', 'w') as f:
         json.dump(negated, f) 

main()

    
