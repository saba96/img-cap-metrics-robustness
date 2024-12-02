import os
import json
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from pytorch_lightning import LightningDataModule
from PIL import Image

from collections import defaultdict

CAPTION_DATASETS = ['coco']


def load_imagetags(data_dir, split):
    # Downloaded from https://cs.stanford.edu/people/karpathy/deepimagesent
    with open(os.path.join(data_dir, f'coco/annotations_trainval2014/instances_{split}2014.json')) as f:
        annotations = json.load(f)['annotations']

    #vqa = load_vqa(data_dir,'val')

    # id is segmentation_id
    #{"segmentation": [[239.97,260.24,222.04,270.49,199.84,253.41,213.5,227.79,259.62,200.46,274.13,202.17,277.55,210.71,249.37,253.41,237.41,264.51,242.54,261.95,228.87,271.34]],"area": 2765.1486500000005,"iscrowd": 0,"image_id": 558840,"bbox": [199.84,200.46,77.71,70.88],"category_id": 58,"id": 156}
    imageid_tags = defaultdict(list)
    for ann in annotations:
        image_id = ann['image_id']
        area = ann['area']
        category_id = ann['category_id']
        id = ann['id']
        imageid_tags[image_id].append((area, category_id, id))
        #imageid_tags[image_id].append(ann)

    num_tags = []
    for id in imageid_tags.keys():
        num_tags.append(len(imageid_tags[id]))
    print('mean number of tags per image', np.mean(num_tags))
    with open(os.path.join(data_dir, f'coco/annotations_trainval2014/instances_train2014.json')) as f:
        tag_categories = json.load(f)['categories']    
    #categories[{"id": int, "name": str, "supercategory": str, "isthing": 0 or 1, "color": [R,G,B],}]
    categories = defaultdict()
    for category in tag_categories:
        id = category['id']
        name = category['name']
        supercategory = category['supercategory']
        categories[id] = (name, supercategory)

    return imageid_tags, categories



def make_sentence(words):
    vowels = ['i', 'o', 'a', 'u', 'e']
    sentence = ''
    #count = defaultdict()
    #for word in words:
    #    count[word] = 0
    #for word in words:
    #    count[word] += 1
    for i in range(len(words)):
        if words[i][0] in vowels:
            words[i] = 'an' + ' ' + words[i]
        else:
            words[i] = 'a' + ' ' + words[i]
    if len(words) == 1:
        sentence = 'There is ' + words[0] + '.'
    elif len(words) == 2:
        sentence =  'There is ' + words[0] + ' and ' + words[1] + '.'
    elif len(words) == 3:
        sentence = 'There is ' + words[0] + ', ' + words[1] + ' and ' + words[2] + '.'
    return sentence

def find_question_type(question, question_types):
    for question_type in question_types:
        question_type = question_type.rstrip()
        question_type = question_type + ' '
        if question.lower().find(question_type.lower()) == 0:
            return question_type
    return None

def filter_QA_captions(QA_captions, question_types, qid_question):
    filtered = []
    for caption in QA_captions:
        #if caption['answer'].find('.') == -1:
        question_id = caption['question_id']
        question = qid_question[question_id]
        question_type = find_question_type(question, question_types)
        if question_type != None:
            caption['question_type'] = question_type
            filtered.append(caption)
            #print(question, caption)
    print(len(filtered))
    return filtered
    
def load_questions(data_dir='/network/scratch/s/saba.ahmadi/data/', split='val'):
    with open(os.path.join(data_dir, f'vqa/v2_OpenEnded_mscoco_{split}2014_questions.json')) as f:
        questions = json.load(f)['questions']
    qid_question = defaultdict()
    for q in questions:
        question_id = q['question_id']
        question = q['question']
        qid_question[question_id] = question
    return qid_question

def load_answers(data_dir='/network/scratch/s/saba.ahmadi/data/', split='val'):
    with open(os.path.join(data_dir, f'vqa/v2_mscoco_{split}2014_annotations.json')) as f:
        annotations = {ann['question_id']: ann for ann in json.load(f)['annotations']}
    qid_answer = defaultdict()
    for question_id in annotations:
        #question_id = annotations[question_id]['question_id']
        answer = annotations[question_id]['multiple_choice_answer']
        qid_answer[question_id] = answer
    return qid_answer

def create_plausible_answers_dict(question_types, data_dir='/network/scratch/s/saba.ahmadi/data/', split='val'):
    questionType_answers = defaultdict(set)
    qid_questions= load_questions(data_dir, split)
    qid_answer = load_answers(data_dir, split)
    for question_id in qid_questions.keys():
        answer = qid_answer[question_id]
        question = qid_questions[question_id]
        question_type = find_question_type(question, question_types)
        if question_type:
            questionType_answers[question_type].add(answer)
    questionType_answersList = defaultdict(list)
    print('question_types', question_types)
    for question_type in questionType_answers.keys():
        print(question_type)
        print(questionType_answers[question_type])
        questionType_answersList[question_type] = list(questionType_answers[question_type])
    print(questionType_answersList)
    return questionType_answersList
#def generate_caps_plausible_answer(questions, asnwers):


def generate_caps_gt_answer(image_tags, categories):
    '''For each image , we generate three captions from one, two, three randomly selected tags from unique tag names of an image.
    structure of captions to be returned
    {'imgid': int, 'caption': string, hallucinated_caption: string, capation_type: 'string', tag: object, tag_name, selected_sibling_name: string, super_category, string}
    'question_id': int
    }
    Please note that we add question_id as a unique unique_id for future use in computing scores in umic
    @TODO: generate unique_id for all cases to avoid this dicrepency in names
    tag:
    {area: float, category_id: int, id: int, tag_name: string}'''
    one_tag = []
    two_tags = []
    three_tags = []
    unique_id = 0
    for image_id in image_tags.keys():
        image_tags_name = set()
        for image_tag in image_tags[image_id]:
            category_id = image_tag[1]
            category_name = categories[int(category_id)][0]
            image_tags_name.add(category_name)
        random_tags_sample = []
        if len(image_tags_name) >= 3:
            random_tags_sample = random.sample(list(image_tags_name), 3)  
            one_tag_cap_obj = {'imgid': image_id,'caption': make_sentence(random_tags_sample[:1]), 'generation_mode': 'one_tag', 'tags_list': random_tags_sample[0], 'id': unique_id} 
            two_tags_cap_obj = {'imgid': image_id,'caption': make_sentence(random_tags_sample[:2]), 'generation_mode': 'two_tags', 'tags_list': random_tags_sample[:2], 'id': unique_id} 
            three_tags_cap_obj = {'imgid': image_id,'caption': make_sentence(random_tags_sample), 'generation_mode': 'three_tags', 'tags_list': random_tags_sample, 'id': unique_id} 
            unique_id += 1
            one_tag.append(one_tag_cap_obj)
            two_tags.append(two_tags_cap_obj)
            three_tags.append(three_tags_cap_obj)
    with open('../dataset/different_fine_grained/one_tag_unique.json', 'w') as f:
        json.dump(one_tag, f)   
    with open('../dataset/different_fine_grained/two_tags_unique.json', 'w') as f:
        json.dump(two_tags, f) 
    with open('../dataset/different_fine_grained/three_tags_unique.json', 'w') as f:
        json.dump(three_tags, f)
    print('num samples:', len(one_tag))


def main():
    # image_tags, categories = load_imagetags('/home/mila/s/saba.ahmadi/scratch/data', 'val')
    # (question_id, image_id, image_file, question, answers)
    with open('../dataset/dataset/vqa_QA_gt_captions.json') as f:
        vqa_gt_caps = json.load(f)
    qid_question = load_questions()
    selected_question_types = []
    with open('../dataset/dataset/att_question_type.txt') as f:
        selected_question_types = f.readlines()
    selected_question_types.sort(key=len)
    selected_question_types.reverse()
    print(selected_question_types)
    filtered_QA_captions = filter_QA_captions(vqa_gt_caps, selected_question_types, qid_question)
    plausible_answers = create_plausible_answers_dict(selected_question_types)
    for caption in filtered_QA_captions:
        question_type = caption['question_type']
        plausible_answer = plausible_answers[question_type]

main()
