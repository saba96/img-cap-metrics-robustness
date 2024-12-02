import json
import numpy as np

import scipy.stats

from collections import defaultdict

def main():
    with open('../dataset/different_fine_grained/vqa_caps_umic.json') as f:
        vqa_gt_caps = json.load(f)

    with open('../dataset/different_fine_grained/plausible_caps_umic.json') as f:
        plausible_caps = json.load(f)

    with open('../dataset/different_fine_grained/image_tag_caps_umic.json') as f:
        image_tag_caps = json.load(f)
    
    with open('../dataset/different_fine_grained/random_caps_umic.json') as f:
        random_caps = json.load(f)

    all_scores = []    
    gt_scores = []
    plausible_scores = []
    image_tag_scores = []
    random_scores = []
    for gt, plausible, image_tag, random in zip(vqa_gt_caps, plausible_caps, image_tag_caps, random_caps):
        gt_scores.append(gt['UMIC_score'])
        plausible_scores.append(plausible['UMIC_score'])
        image_tag_scores.append(image_tag['UMIC_score'])
        random_scores.append(random['UMIC_score'])
        score = (gt['UMIC_score'], plausible['UMIC_score'], image_tag['UMIC_score'], random['UMIC_score'])
        all_scores.append(score)
    min = np.min(all_scores)
    max = np.max(all_scores)
    range = max - min
    print('min, max: ', min, max, range)
    print('gt', np.mean((gt_scores-min)/ range), np.std((gt_scores-min)/ range))
    print('plausible', np.mean((plausible_scores-min)/ range), np.std((plausible_scores-min)/ range))
    print('image_tag', np.mean((image_tag_scores-min)/ range), np.std((image_tag_scores-min)/ range))
    print('random', np.mean((random_scores-min)/ range), np.std((random_scores-min)/ range))
    
    print(len(random_scores))
    print('t-test gt/plausible: ', scipy.stats.ttest_ind((gt_scores-min)/range,(plausible_scores-min)/range))
    print('t-test gt/image_tag: ', scipy.stats.ttest_ind((gt_scores-min)/range,(image_tag_scores-min)/range))
    print('t-test gt/random: ', scipy.stats.ttest_ind((gt_scores-min)/range,(random_scores-min)/range))
    print('t-test plausible/image_tag: ', scipy.stats.ttest_ind((plausible_scores-min)/range,(image_tag_scores-min)/range))
    
    plausible = ((plausible_scores-min)/range) - ((gt_scores-min)/range)
    image_tag = ((image_tag_scores-min)/range) - ((gt_scores-min)/range)
    print('t-test diff plausible/image_tag: ', scipy.stats.ttest_ind(plausible, image_tag))
    
    sorted_indices = np.argsort(all_scores, axis=1)
    orders = defaultdict()
    for indices in sorted_indices:
        order = str(indices[0]) + str(indices[1]) +  str(indices[2]) + str(indices[3])
        if order not in orders.keys():
            orders[order] = 1
        else:
            orders[order] += 1
    for order in orders:
        print(order, ': ', orders[order]/len(random_caps)*100, '%')
main()