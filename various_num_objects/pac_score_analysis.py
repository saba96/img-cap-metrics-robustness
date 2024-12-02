import json
import numpy as np

import scipy.stats

def main():
    with open('../dataset/various_num_objects/one_tag_unique_pac_score.json') as f:
        one_tag_caps = json.load(f)

    with open('../dataset/various_num_objects/two_tags_unique_pac_score.json') as f:
        two_tags_caps = json.load(f)

    with open('../dataset/various_num_objects/three_tags_unique_pac_score.json') as f:
        three_tags_caps = json.load(f)

    with open('../dataset/various_num_objects/one_tag_shuffled_pac_score.json') as f:
        one_tag_shuffled = json.load(f)

    with open('../dataset/various_num_objects/two_tags_shuffled_pac_score.json') as f:
        two_tags_shuffled = json.load(f)

    with open('../dataset/various_num_objects/three_tags_shuffled_pac_score.json') as f:
        three_tags_shuffled = json.load(f)
    

    # all_scores_three = []
    # all_scores_two = []
    # right_order_two = 0
    # right_order_three = 0
    all_scores = []
    one_tag_scores = []
    two_tags_scores = []
    three_tags_scores = []
    one_tag_shuffled_scores = []
    two_tags_shuffled_scores = []
    three_tags_shuffled_scores = []
    for id in one_tag_caps:
        all_scores.append(one_tag_caps[id]['PAC-S'])
        all_scores.append(two_tags_caps[id]['PAC-S'])
        all_scores.append(three_tags_caps[id]['PAC-S'])
        all_scores.append(one_tag_shuffled[id]['PAC-S'])
        all_scores.append(two_tags_shuffled[id]['PAC-S'])
        all_scores.append(three_tags_shuffled[id]['PAC-S'])
        one_tag_scores.append(one_tag_caps[id]['PAC-S'])
        two_tags_scores.append(two_tags_caps[id]['PAC-S'])
        three_tags_scores.append(three_tags_caps[id]['PAC-S'])
        one_tag_shuffled_scores.append(one_tag_shuffled[id]['PAC-S'])
        two_tags_shuffled_scores.append(two_tags_shuffled[id]['PAC-S'])
        three_tags_shuffled_scores.append(three_tags_shuffled[id]['PAC-S'])
    min = np.min(all_scores)
    max = np.max(all_scores)
    range = max - min
    print(min, max, range)
    print(np.mean((one_tag_scores-min)/range), ' \pm ', np.std((one_tag_scores-min)/range))
    print(np.mean((two_tags_scores-min)/range), ' \pm ', np.std((two_tags_scores-min)/range))
    print(np.mean((three_tags_scores-min)/range), ' \pm ', np.std((two_tags_scores-min)/range))
    print(np.mean((one_tag_shuffled_scores-min)/range), ' \pm ', np.std((one_tag_shuffled_scores-min)/range))
    print(np.mean((two_tags_shuffled_scores-min)/range), ' \pm ', np.std((two_tags_shuffled_scores-min)/range))
    print(np.mean((three_tags_shuffled_scores-min)/range), ' \pm ', np.std((three_tags_shuffled_scores-min)/range))
    print('t-test one/two: ', scipy.stats.ttest_ind((one_tag_scores-min)/range,(two_tags_scores-min)/range))
    print('t-test two/three: ', scipy.stats.ttest_ind((two_tags_scores-min)/range,(three_tags_scores-min)/range))
    print('t-test one/three: ', scipy.stats.ttest_ind((one_tag_scores-min)/range,(three_tags_scores-min)/range))
    
    print('t-test one/shuffled_one: ', scipy.stats.ttest_ind((one_tag_scores-min)/range,(one_tag_shuffled_scores-min)/range))
    print('t-test two/shuffled_two: ', scipy.stats.ttest_ind((two_tags_scores-min)/range,(two_tags_shuffled_scores-min)/range))
    print('t-test three/shuffled_three: ', scipy.stats.ttest_ind((three_tags_scores-min)/range,(three_tags_shuffled_scores-min)/range))
    
    print('t-test shuffled_three/two: ', scipy.stats.ttest_ind((three_tags_shuffled_scores-min)/range, (two_tags_scores-min)/range))
    print('t-test shuffled_two/one', scipy.stats.ttest_ind((two_tags_shuffled_scores-min)/range,(one_tag_scores-min)/range))
    
main()