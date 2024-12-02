import os
import json
import argparse


from collections import defaultdict

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
        question_id =int(q['question_id'])
        image_id = q['image_id']
        image_file = os.path.join(data_dir, f'coco/{split}2014', images[image_id]['file_name'])
        question = q['question']
        answers = [ans['answer'] for ans in annotations[question_id]['answers']]
        samples.append({"question_id":question_id, "image_id":image_id, "image_file":image_file, "question":question, "answers":answers})
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqa_result', type=str, default='./vqa_captions_gt.json')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--data_dir', type=str, default='/network/scratch/s/saba.ahmadi/data')
    args = parser.parse_args()

    with open(args.vqa_result) as f:
        results = json.load(f)

    question_ids = []
    imgids = [] 
    captions = []
    answers = []
    for res in results:
        question_ids.append(res['question_id'])
        #imgid = str(res['imgid']).zfill(12)
        answer = res['answer']
        answers.append(answer)
        imgid = res['imgid']
        imgids.append(imgid)
        caption = res['caption'].strip()
        index = caption.find('\nLong Answer: ')
        caption = caption[index+ len('\nLong Answer: '):]
        # double check
        index = caption.find('\n')
        if index>0:
            caption = caption[:index]
        captions.append(caption.strip() + '.')
    results = [{'question_id': int(qid), 'imgid': str(image_id), 'caption': caption, 'answer': answer} for qid, image_id, caption, answer in zip(question_ids, imgids, captions, answers)]
    with open(args.save_path, 'w') as f:
        json.dump(results, f)
    print(f'\nResults saved to {args.save_path}')

main()
