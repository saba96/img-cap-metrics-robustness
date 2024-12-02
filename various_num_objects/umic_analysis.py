import json
import numpy as np
import scipy.stats

def main():
    with open('../dataset/various_num_objects/one_tag_unique_umic.json') as f:
        one_tag_caps = json.load(f)

    with open('../dataset/various_num_objects/two_tags_unique_umic.json') as f:
        two_tags_caps = json.load(f)

    with open('../dataset/various_num_objects/three_tags_unique_umic.json') as f:
        three_tags_caps = json.load(f)

    with open('../dataset/various_num_objects/one_tag_shuffled_captions_umic.json') as f:
        one_tag_shuffled = json.load(f)

    with open('../dataset/various_num_objects/two_tags_shuffled_captions_umic.json') as f:
        two_tags_shuffled = json.load(f)

    with open('../dataset/various_num_objects/three_tags_shuffled_captions_umic.json') as f:
        three_tags_shuffled = json.load(f)

    all_scores_three = []
    all_scores_two = []
    right_order_two = 0
    right_order_three = 0
    all_scores = []
    one_tag_scores = []
    two_tags_scores = []
    three_tags_scores = []
    one_tag_shuffled_scores = []
    two_tags_shuffled_scores = []
    three_tags_shuffled_scores = []
    for one_tag, two_tags, three_tags, one_tag_shuffled, two_tags_shuffled, three_tags_shuffled in zip(one_tag_caps, two_tags_caps, three_tags_caps, one_tag_shuffled, two_tags_shuffled, three_tags_shuffled):
        tuple = (one_tag['UMIC_score'], two_tags['UMIC_score'])
        triplet = (one_tag['UMIC_score'], two_tags['UMIC_score'], three_tags['UMIC_score'])
        all_scores_two.append(tuple)
        all_scores_three.append(triplet)
        all_scores.append(one_tag['UMIC_score'])
        all_scores.append(two_tags['UMIC_score'])
        all_scores.append(three_tags['UMIC_score'])
        one_tag_scores.append(one_tag['UMIC_score'])
        two_tags_scores.append(two_tags['UMIC_score'])
        three_tags_scores.append(three_tags['UMIC_score'])
        one_tag_shuffled_scores.append(one_tag_shuffled['UMIC_score'])
        two_tags_shuffled_scores.append(two_tags_shuffled['UMIC_score'])
        three_tags_shuffled_scores.append(three_tags_shuffled['UMIC_score'])
    min = np.min(all_scores_three)
    max = np.max(all_scores_three)
    range = max-min
    print(min, max, range)
    print(np.mean((one_tag_scores-min)/range), ' \pm ', np.std((one_tag_scores-min)/range))
    print(np.mean((two_tags_scores-min)/range), ' \pm ', np.std((two_tags_scores-min)/range))
    print(np.mean((three_tags_scores-min)/range), ' \pm ', np.std((two_tags_scores-min)/range))
    print(np.mean((one_tag_shuffled_scores-min)/range), ' \pm ', np.std((one_tag_shuffled_scores-min)/range))
    print(np.mean((two_tags_shuffled_scores-min)/range), ' \pm ', np.std((two_tags_shuffled_scores-min)/range))
    print(np.mean((three_tags_shuffled_scores-min)/range), ' \pm ', np.std((three_tags_shuffled_scores-min)/range))

    print(len(one_tag_scores))
    print('t-test one/two: ', scipy.stats.ttest_ind((one_tag_scores-min)/range,(two_tags_scores-min)/range))
    print('t-test two/three: ', scipy.stats.ttest_ind((two_tags_scores-min)/range,(three_tags_scores-min)/range))
    print('t-test one/three: ', scipy.stats.ttest_ind((one_tag_scores-min)/range,(three_tags_scores-min)/range))
    
    print('t-test one/shuffled_one: ', scipy.stats.ttest_ind((one_tag_scores-min)/range,(one_tag_shuffled_scores-min)/range))
    print('t-test two/shuffled_two: ', scipy.stats.ttest_ind((two_tags_scores-min)/range,(two_tags_shuffled_scores-min)/range))
    print('t-test three/shuffled_three: ', scipy.stats.ttest_ind((three_tags_scores-min)/range,(three_tags_shuffled_scores-min)/range))
    
    print('t-test shuffled_three/two: ', scipy.stats.ttest_ind((three_tags_shuffled_scores-min)/range, (two_tags_scores-min)/range))
    print('t-test shuffled_two/one', scipy.stats.ttest_ind((two_tags_shuffled_scores-min)/range,(one_tag_scores-min)/range))
    
    
    sorted_indices_two = np.argsort(all_scores_two, axis=1)
    for indices in sorted_indices_two:
        #print(indices)
        if (indices==[0, 1]).all():
            right_order_two += 1
    print('In', right_order_two, ' of cases umic increases based on correct related information added to the caption(checked for two tags).')
    sorted_indices_three = np.argsort(all_scores_three, axis=1)
    for indices in sorted_indices_three:
        if (indices==[0, 1, 2]).all():
            right_order_three += 1
    print('In', right_order_three, ' of cases umic increases based on correct related information added to the caption(checked for three tags).')
main()