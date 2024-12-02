import json
import numpy as np
import scipy.stats

from collections import defaultdict

def main():
    with open('../dataset/object_size/big_tags_captions_umic.json') as f:
        big_tags = json.load(f)

    with open('../dataset/object_size/small_tags_captions_umic.json') as f:
        small_tags = json.load(f)

    with open('../dataset/object_size/big_tags_shuffled_captions_umic.json') as f:
        big_tags_shuffled = json.load(f)
    
    with open('../dataset/object_size/small_tags_shuffled_captions_umic.json') as f:
        small_tags_shuffled = json.load(f)

    all_scores = []
    almost_equal_big = 0
    almost_equal_small = 0
    big_tags_shuffled_scores = []
    small_tags_shuffled_scores = []
    
    big_tags_scores = []
    small_tags_scores = []
    for big, small, big_shuffled, small_shuffled in zip(big_tags, small_tags, big_tags_shuffled, small_tags_shuffled):
        score = (big['UMIC_score'], small['UMIC_score'], big_shuffled['UMIC_score'], small_shuffled['UMIC_score'])
        big_tags_scores.append(big['UMIC_score'])
        small_tags_scores.append(small['UMIC_score'])
        big_tags_shuffled_scores.append(big_shuffled['UMIC_score'])
        small_tags_shuffled_scores.append(small_shuffled['UMIC_score'])
        # if np.abs(big['UMIC_score'] - big_shuffled['UMIC_score'])  < 0.001:
        #     almost_equal_big += 1
        #     print('sample big:', big, big_shuffled)
        # if np.abs(small['UMIC_score']- small_shuffled['UMIC_score']) < 0.001:
        #     almost_equal_small += 1
        #     print('sample small:', small, small_shuffled)
        all_scores.append(score)
    min = np.min(all_scores)
    max = np.max(all_scores)
    range = max-min
    print(min, max, range)
    print(np.mean((big_tags_scores-min)/range), np.std((big_tags_scores-min)/range))
    print(np.mean((small_tags_scores-min)/range), np.std((small_tags_scores-min)/range))
    print(np.mean((big_tags_shuffled_scores-min)/range), np.std((big_tags_shuffled_scores-min)/range))
    print(np.mean((small_tags_shuffled_scores-min)/range), np.std((small_tags_shuffled_scores-min)/range))
    print('**********')
    print('almost_equal_big', almost_equal_big/len(small_tags_shuffled)*100, '%')
    print('almost_equal_small', almost_equal_small/len(small_tags_shuffled)*100, '%')
    print('t-test small/big: ', scipy.stats.ttest_ind((small_tags_scores-min)/range,(big_tags_scores-min)/range))
    print('t-test small/shuffled: ', scipy.stats.ttest_ind((small_tags_scores-min)/range,(small_tags_shuffled_scores-min)/range))
    print('t-test big/shuffled: ', scipy.stats.ttest_ind((big_tags_scores-min)/range,(big_tags_shuffled_scores-min)/range))
    print('t-test small/big-shuffled: ', scipy.stats.ttest_ind((small_tags_scores-min)/range,(big_tags_shuffled_scores-min)/range))
    
    print('t-test big/small-shuffled: ', scipy.stats.ttest_ind((big_tags_scores-min)/range,(small_tags_shuffled_scores-min)/range))
    
    sorted_indices = np.argsort(all_scores, axis=1)
    orders = defaultdict()
    orders['12'] = 0
    orders['02'] = 0
    orders['13'] = 0
    for indices in sorted_indices:
        order = str(indices[0]) + str(indices[1]) +  str(indices[2]) + str(indices[3])
        if order not in orders.keys():
            orders[order] = 1
        else:
            orders[order] += 1
        if '12' in order:
            orders['12'] += 1
        if '02' in order:
            orders['02'] += 1
        if '13' in order:
            orders['13'] += 1
    
    index_dict = {'0': 'S_(big_object)', '1': 'S_(small_object)','2': 'S_(big_object_shuffled)', '3': 'S_(small_object_shuffled)'}
    for order in orders:
        order_string = ''
        for index in order:
            order_string += index_dict[index] + '<'
        print(order_string[:-1], ': ', orders[order]/len(small_tags_shuffled)*100, '%')
    
    print('big tags shuffled mean and std: $', np.mean(big_tags_shuffled_scores), ' \pm ', np.std(big_tags_shuffled_scores), '$')
    print('small tags shuffled mean and std: $', np.mean(small_tags_shuffled_scores), ' \pm ', np.std(small_tags_shuffled_scores), '$')
main()