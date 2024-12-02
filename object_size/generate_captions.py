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

from random import shuffle

CAPTION_DATASETS = ['coco']



def load_vqa(data_dir, split):
    assert split in ['train', 'val']

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
        samples.append((question_id, image_id, image_file, question, answers))

    return samples


def modify_caption_by_tag(res, tag):
    start_index = res["caption"].lower().find(res['answer'])
    if start_index > -1:
        #print(res['answer'])
        new_cap = res["caption"][:start_index] + tag + res["caption"][start_index+len(res['answer']):]
        new_cap = new_cap.strip().capitalize()
        return new_cap
    else:
        print('answer was not in the caption. Answer:', res['answer'], ' Caption:', res['caption'])

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

def generate_size_related_caps(tags):
    small_object_correct = ''
    small_object_wrong = ''
    big_object_correct = ''
    big_object_incorrect = ''


def plot(all_areas, all_diffs):
    print('diff mean', np.mean(all_diffs), 'variance ', np.var(all_diffs))
    print('area mean', np.mean(all_areas), 'variance ', np.var(all_areas))
    bins = np.linspace(np.min(all_areas), np.max(all_areas), 20)
    _, ax = plt.subplots()
    #plt.hist(np.array(all_areas), bins)
    counts, bins = np.histogram(all_areas, bins= bins)
    plt.hist(bins[:-1], bins, weights=counts/len(all_areas))
    plt.title(f'Histogram of coco unique image tags area', fontsize = 12)
    plt.xlabel('Area')
    ax.set_xticks(np.linspace(np.min(all_areas), np.max(all_areas), 10), rotation=45)
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig(f'/home/mila/s/saba.ahmadi/clip-lm/vqa-eval/baselines/tag_size/image_unique_tags_area.png')
    plt.clf()
    print('diff range', np.min(all_diffs),np.max(all_diffs))
    bins = np.linspace(np.min(all_diffs),np.max(all_diffs), 20)
    counts, bins = np.histogram(all_diffs, bins= bins)
    plt.hist(bins[:-1], bins, weights=counts/len(all_diffs))
    #plt.hist(all_diffs, bins)
    ax.set_xticks(np.linspace(np.min(all_diffs),np.max(all_diffs), 10), rotation=45)
    plt.title(f'Histogram of difference between largest and smallest tag area for an image', fontsize = 8)
    plt.xlabel('Area')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig(f'/home/mila/s/saba.ahmadi/clip-lm/vqa-eval/baselines/tag_size/unique_diff_tags_area.png')
    plt.clf()


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

def scramble(sentence1, sentence2):
    order = None
    sentence1 = sentence1[:-1].split()
    sentence2 = sentence2[:-1].split()
    min_length = min(len(sentence1), len(sentence2))
    order = np.arange(0, min_length)
    shuffle(order) 
    #print(order)
    while ((order==np.arange(0, min_length)).all()):
        shuffle(order) 
        #print(order)           
    sentence1_shuffled =  ''
    sentence2_shuffled = ''
    for index in order:
        sentence1_shuffled += ' ' + sentence1[index].lower()
        sentence2_shuffled += ' ' + sentence2[index].lower()
    sentence1_shuffled = (sentence1_shuffled.strip()+' '.join(sentence1[min_length:])+'.').capitalize()
    sentence2_shuffled = (sentence2_shuffled.strip()+' '.join(sentence2[min_length:])+'.').capitalize()
    print(sentence1_shuffled, sentence2_shuffled)
    return sentence1_shuffled, sentence2_shuffled

def pick_tags(image_tags, categories):
    ''' We can have multiple tags with the same tag name, like we have many cars all of them have tag name car
        So I make a set of image tag names and compute cumulative tag area with the same name.
        Then I pick tag name with the smallest and the biggest area.
    '''
    all_areas = []
    all_diffs = []
    big_tag_caps = []
    small_tag_caps = []
    big_tag_caps_shuffled = []
    small_tag_caps_shuffled = []
    empty_caps = []
    unique_id = 0
    all_count = 0
    count = 0
    for image_id in image_tags.keys():
        name_area = defaultdict()
        areas = []
        all_count += 1
        for image_tag in image_tags[image_id]:
            area, category_id = image_tag[0], image_tag[1]
            category_name = categories[int(category_id)][0]
            if category_name not in name_area.keys():
                name_area[category_name] = area
            else:
                name_area[category_name] += area
        if len(name_area.keys()) >= 2:
            unordered = []
            for tag_name in name_area:
                areas.append(name_area[tag_name])
                all_areas.append(name_area[tag_name])
                unordered.append((name_area[tag_name], tag_name))
            sorted__area = np.sort(areas)
            diff_area = sorted__area[-1] - sorted__area[0]
            all_diffs.append(diff_area)
            sorted_areas = sorted(unordered)
            if (sorted_areas[-1][0] - sorted_areas[0][0]) > 15000:
                count += 1
                big_tag_sentence = make_sentence([sorted_areas[-1][1]])
                small_tag_sentence = make_sentence([sorted_areas[0][1]])
                big_tag_sentence_shuffled, small_tag_sentence_shuffled = scramble(big_tag_sentence, small_tag_sentence)
                big_tag_cap = {'imgid': image_id,'caption': big_tag_sentence, 'generation_mode': 'big_tag_correct', 'tag': sorted_areas[-1][1], 'area': sorted_areas[-1][0],'id': unique_id} 
                small_tag_cap = {'imgid': image_id,'caption': small_tag_sentence, 'generation_mode': 'small_tag_correct', 'tag': sorted_areas[0][1], 'area': sorted_areas[-1][0],'id': unique_id} 
                big_tag_cap_shuffled = {'imgid': image_id,'caption': big_tag_sentence_shuffled, 'generation_mode': 'big_tag_correct', 'tag': sorted_areas[-1][1], 'area': sorted_areas[-1][0],'id': unique_id} 
                small_tag_cap_shuffled = {'imgid': image_id,'caption': small_tag_sentence_shuffled, 'generation_mode': 'small_tag_correct', 'tag': sorted_areas[0][1], 'area': sorted_areas[-1][0],'id': unique_id} 
                empty_cap = {'imgid': image_id,'caption': '', 'generation_mode': 'empty_tag', 'tag': '', 'area': sorted_areas[-1][0],'id': unique_id} 
                big_tag_caps.append(big_tag_cap)
                small_tag_caps.append(small_tag_cap)
                big_tag_caps_shuffled.append(big_tag_cap_shuffled)
                small_tag_caps_shuffled.append(small_tag_cap_shuffled)
                empty_caps.append(empty_cap)
                unique_id += 1
    print(count/all_count, ' percent of images were included.')
    print(len(big_tag_caps))
    with open('../dataset/object_size/big_tags_captions.json', 'w') as f:
        json.dump(big_tag_caps, f)   
    with open('../dataset/object_size/small_tags_captions.json', 'w') as f:
        json.dump(small_tag_caps, f)
    with open('../dataset/object_size/big_tags_shuffled_captions.json', 'w') as f:
        json.dump(big_tag_caps_shuffled, f)   
    with open('../dataset/object_size/small_tags_shuffled_captions.json', 'w') as f:
        json.dump(small_tag_caps_shuffled, f)
    with open('../dataset/object_size/empty_captions.json', 'w') as f:
        json.dump(empty_caps, f)
    #plot(all_areas, all_diffs)

def main():
    sibling_sub_categories = defaultdict(set)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/network/scratch/s/saba.ahmadi/data')
    args = parser.parse_args()

    print("Arguments:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")
    image_tags, categories = load_imagetags(args.data_dir, 'val')
    print(categories)
    for category_id in categories.keys():
        category = categories[category_id]
        #sibling_sub_categories[category[1]].append(category[0])
        sibling_sub_categories[category[1]].add(category[0])
    pick_tags(image_tags, categories)

main()
