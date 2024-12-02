import json
import numpy as np

import scipy.stats

from collections import defaultdict

def main():
    with open('../dataset/different_fine_grained/vqa_caps_pac_score.json') as f:
        vqa_gt_caps = json.load(f)

    with open('../dataset/different_fine_grained/plausible_caps_pac_score.json') as f:
        plausible_caps = json.load(f)

    with open('../dataset/different_fine_grained/image_tag_caps_pac_score.json') as f:
        image_tag_caps = json.load(f)
    
    with open('../dataset/different_fine_grained/random_caps_pac_score.json') as f:
        random_caps = json.load(f)

    all_scores = []
    gt = []
    plausible = []
    image_tag = []
    random = []
    for id  in vqa_gt_caps:
        gt.append(vqa_gt_caps[id]['PAC-S'])
        plausible.append(plausible_caps[id]['PAC-S'])
        image_tag.append(image_tag_caps[id]['PAC-S'])
        random.append(random_caps[id]['PAC-S'])
        print(vqa_gt_caps[id]['PAC-S'], plausible_caps[id]['PAC-S'], image_tag_caps[id]['PAC-S'], random_caps[id]['PAC-S'])
        score = (vqa_gt_caps[id]['PAC-S'], plausible_caps[id]['PAC-S'], image_tag_caps[id]['PAC-S'], random_caps[id]['PAC-S'])
        all_scores.append(score)
    min = np.min(all_scores)
    max = np.max(all_scores)
    range = max - min
    print('min, max: ', min, max, range)
    print('gt', np.mean((gt-min)/ range), np.std((gt-min)/ range))
    print('plausible', np.mean((plausible-min)/ range), np.std((plausible-min)/ range))
    print('image_tag', np.mean((image_tag-min)/ range), np.std((image_tag-min)/ range))
    print('random', np.mean((random-min)/ range), np.std((random-min)/ range))
    
    print('t-test gt/plausible: ', scipy.stats.ttest_ind((gt-min)/range,(plausible-min)/range))
    print('t-test gt/image_tag: ', scipy.stats.ttest_ind((gt-min)/range,(image_tag-min)/range))
    print('t-test gt/random: ', scipy.stats.ttest_ind((gt-min)/range,(random-min)/range))
    print('t-test plausible/image_tag: ', scipy.stats.ttest_ind((plausible-min)/range,(image_tag-min)/range))
    
    plausible = ((plausible-min)/range) - ((gt-min)/range)
    image_tag = ((image_tag-min)/range) - ((gt-min)/range)
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