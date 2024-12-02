import os
import json
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import argparse

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
    
def load_questions(data_dir, split='val'):
    with open(os.path.join(data_dir, f'vqa/v2_OpenEnded_mscoco_{split}2014_questions.json')) as f:
        questions = json.load(f)['questions']
    qid_question = defaultdict()
    for q in questions:
        question_id = q['question_id']
        question = q['question']
        qid_question[question_id] = question
    return qid_question

def load_answers(data_dir, split='val'):
    with open(os.path.join(data_dir, f'vqa/v2_mscoco_{split}2014_annotations.json')) as f:
        annotations = {ann['question_id']: ann for ann in json.load(f)['annotations']}
    qid_answer = defaultdict()
    for question_id in annotations:
        #question_id = annotations[question_id]['question_id']
        answer = annotations[question_id]['multiple_choice_answer']
        qid_answer[question_id] = answer
    return qid_answer

def create_plausible_answers_dict(question_types, data_dir, split='val'):
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


def create_nonvalidated_plausible_answers_dict(question_types, data_dir):
    questionType_answers = defaultdict(set)
    qid_questions= load_questions(data_dir=data_dir, split='val')
    qid_answer = load_answers(data_dir=data_dir, split='val')
    for question_id in qid_questions.keys():
        answer = qid_answer[question_id]
        question = qid_questions[question_id]
        question_type = find_question_type(question, question_types)
        if question_type:
            questionType_answers[question_type].add(answer)
    questionType_answersList = defaultdict(list)
    for question_type in questionType_answers.keys():
        questionType_answersList[question_type] = list(questionType_answers[question_type])
    qid_questions= load_questions(data_dir=data_dir, split='train')
    qid_answer = load_answers(data_dir=data_dir, split='train')
    for question_id in qid_questions.keys():
        answer = qid_answer[question_id]
        question = qid_questions[question_id]
        question_type = find_question_type(question, question_types)
        if question_type:
            questionType_answers[question_type].add(answer)
    for question_type in questionType_answers.keys():
        questionType_answersList[question_type] = list(questionType_answers[question_type])
    #print(questionType_answersList)
    return questionType_answersList
#def generate_caps_plausible_answer(questions, asnwers):


def map_imageId_tagNames(image_tags, categories):
    '''For each image , we generate three captions from one, two, three randomly selected tags from unique tag names of an image.
    structure of captions to be returned
    {'imgid': int, 'caption': string, hallucinated_caption: string, capation_type: 'string', tag: object, tag_name, selected_sibling_name: string, super_category, string}
    'question_id': int
    }
    Please note that we add question_id as a unique unique_id for future use in computing scores in umic
    @TODO: generate unique_id for all cases to avoid this dicrepency in names
    tag:
    {area: float, category_id: int, id: int, tag_name: string}'''
    imageId_tagName = defaultdict(list)
    for image_id in image_tags.keys():
        image_tags_name = set()
        for image_tag in image_tags[image_id]:
            category_id = image_tag[1]
            category_name = categories[int(category_id)][0]
            image_tags_name.add(category_name)
        imageId_tagName[image_id] = list(image_tags_name)
    return imageId_tagName


def modify_caption_by_tag(res, tag):
    start_index = res["caption"].lower().find(res['answer'])
    if start_index > -1:
        #print(res['answer'])
        new_cap = res["caption"][:start_index] + tag + res["caption"][start_index+len(res['answer']):]
        new_cap = new_cap.strip().capitalize()
        return new_cap
    else:
        print('answer was not in the caption. Answer:', res['answer'], ' Caption:', res['caption'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/network/scratch/s/saba.ahmadi/data')
    args = parser.parse_args()

    print("Arguments:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")
    image_tags, categories = load_imagetags(args.data_dir, 'val')
    # (question_id, image_id, image_file, question, answers)
    with open('../dataset/dataset/vqa_QA_gt_captions.json') as f:
        vqa_gt_caps = json.load(f)
    with open('../dataset/dataset/manually_postprocessed_plausible_answers.json') as f:
        plausible_answers = json.load(f)
    qid_question = load_questions()
    selected_question_types = []
    with open('../dataset/dataset/att_question_type.txt') as f:
        selected_question_types = f.readlines()
    with open('../dataset/./different_fine_grained/answer_list.json') as f:
        vqa_answers = set(json.load(f))
    selected_question_types.sort(key=len)
    selected_question_types.reverse()
    print(selected_question_types)
    filtered_QA_captions = filter_QA_captions(vqa_gt_caps, selected_question_types, qid_question)
    imageId_tagNames = map_imageId_tagNames(image_tags, categories)
    #print(imageId_tagNames)
    non_validated_plausible_answers = create_nonvalidated_plausible_answers_dict(selected_question_types)
    #plausible_answers = create_plausible_answers_dict(selected_question_types)
    vqa_caps = []
    plausible_caps = []
    image_tag_caps = []
    random_caps = []
    unique_id = 0
    qid_question = load_questions()
    for caption in filtered_QA_captions:
        question_type = caption['question_type']
        plausible_answer = plausible_answers[question_type]
        non_validated_plausible_answer = non_validated_plausible_answers[question_type]
        image_tag_names = imageId_tagNames[int(caption['imgid'])]
        image_id = caption['imgid']
        if len(((vqa_answers - set(plausible_answer)) - set(image_tag_names))- set(non_validated_plausible_answer)) > 0 and len(list(plausible_answer)) > 0 and len((set(image_tag_names)-set(non_validated_plausible_answer))) > 0 :
            remaining_answers = ((vqa_answers - set(plausible_answer)) - set(image_tag_names))- set(non_validated_plausible_answer)
            sample_plausible_answer = random.sample(list(plausible_answer), 1)
            while caption['answer'] in sample_plausible_answer[0]:
                sample_plausible_answer = random.sample(list(plausible_answer), 1)
            sample_image_tag = random.sample(list(set(image_tag_names)-set(non_validated_plausible_answer)), 1)
            if caption['answer'] in sample_image_tag[0]:
                #sample_image_tag = random.sample(list(set(image_tag_names)-set(non_validated_plausible_answer)), 1)
                continue
            random_answers = random.sample(list(remaining_answers), 1)
            while caption['answer'] in random_answers[0]:
                random_answers = random.sample(list(remaining_answers), 1)
            if len(random_answers) == 1 and len(sample_image_tag) == 1 and len(sample_plausible_answer)==1 and (caption["caption"].lower().find(caption['answer']) >= 0):
                vqa_cap = {'imgid': image_id,'caption': caption['caption'], 'generation_mode': 'QA_caption', 'tag_name': caption['answer'],'id': unique_id} 
                plausible_cap = {'imgid': image_id,'caption': modify_caption_by_tag(caption, sample_plausible_answer[0]), 'generation_mode': 'palusible_answer', 'tag_name': sample_plausible_answer[0],'id': unique_id} 
                image_tag_cap = {'imgid': image_id,'caption': modify_caption_by_tag(caption, sample_image_tag[0]), 'generation_mode': 'image_tag', 'tag_name': sample_image_tag[0],'id': unique_id} 
                random_cap = {'imgid': image_id,'caption': modify_caption_by_tag(caption, random_answers[0]), 'generation_mode': 'random_answer', 'tag_name': random_answers[0],'id': unique_id} 
                unique_id += 1
                '''print('Question: ', qid_question[caption['question_id']],'Answer: ', caption['answer'])
                print('QA caption: ', vqa_cap['caption'])
                print('Plausible answer: ', modify_caption_by_tag(caption, sample_plausible_answer[0])) 
                print('Image tag: ',modify_caption_by_tag(caption, sample_image_tag[0]))
                print('Random Answer: ',modify_caption_by_tag(caption, random_answers[0]))
                print('**********************************************')'''
                vqa_caps.append(vqa_cap)
                plausible_caps.append(plausible_cap)
                image_tag_caps.append(image_tag_cap)
                random_caps.append(random_cap)
        '''print('question_type', question_type, caption['caption'])
        print('plausible_answer', plausible_answer)
        print('image_tag_names', image_tag_names)
        print('random_answers', random_answers)'''
    print(len(vqa_caps), len(plausible_caps), len(image_tag_caps), len(random_caps))
    with open('./dataset/different_fine_grained/vqa_caps.json', 'w') as f:
        json.dump(vqa_caps, f)   
    with open('./dataset/different_fine_grained/plausible_caps.json', 'w') as f:
        json.dump(plausible_caps, f)
    with open('./dataset/different_fine_grained/image_tag_caps.json', 'w') as f:
        json.dump(image_tag_caps, f)  
    with open('./dataset/different_fine_grained/random_caps.json', 'w') as f:
        json.dump(random_caps, f)
main()
