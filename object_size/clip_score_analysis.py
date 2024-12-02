import json
import numpy as np
import scipy.stats

from collections import defaultdict

def main():
    with open('../dataset/object_size/big_tags_captions_clip_score.json') as f:
        big_tags = json.load(f)

    with open('../dataset/object_size/small_tags_captions_clip_score.json') as f:
        small_tags = json.load(f)

    with open('../dataset/object_size/big_tags_shuffled_captions_clip_score.json') as f:
        big_tags_shuffled = json.load(f)
    
    with open('../dataset/object_size/small_tags_shuffled_captions_clip_score.json') as f:
        small_tags_shuffled = json.load(f)

    with open('../dataset/object_size/big_tags_captions_umic.json') as f:
        big_tags_umic = json.load(f)

    with open('../dataset/object_size/small_tags_captions_umic.json') as f:
        small_tags_umic = json.load(f)

    with open('../dataset/object_size/big_tags_shuffled_captions_umic.json') as f:
        big_tags_shuffled_umic = json.load(f)
    
    with open('../dataset/object_size/small_tags_shuffled_captions_umic.json') as f:
        small_tags_shuffled_umic = json.load(f)

    all_scores = []
    almost_equal_big = 0
    almost_equal_small = 0
    big_tags_scores = []
    small_tags_scores = []
    big_tags_shuffled_scores = []
    small_tags_shuffled_scores = []
    for id  in big_tags:
       #print(big_tags[id]['CLIPScore'], small_tags[id]['CLIPScore'], big_tags_shuffled[id]['CLIPScore'], small_tags_shuffled[id]['CLIPScore'])
       #print(big_tags[id]['CLIPScore'], big_tags_shuffled[id]['CLIPScore'])
        if (np.abs(big_tags[id]['CLIPScore'] - big_tags_shuffled[id]['CLIPScore'])) < 0.01:
            almost_equal_big += 1
            #print('sample big: ', id)
        if (np.abs(small_tags[id]['CLIPScore']- small_tags_shuffled[id]['CLIPScore'])) < 0.01:
            almost_equal_small += 1
            #print('sample small: ', id)
        score = (big_tags[id]['CLIPScore'], small_tags[id]['CLIPScore'], big_tags_shuffled[id]['CLIPScore'], small_tags_shuffled[id]['CLIPScore'])
        all_scores.append(score)
        big_tags_scores.append(big_tags[id]['CLIPScore'])
        small_tags_scores.append(small_tags[id]['CLIPScore'])
        big_tags_shuffled_scores.append(big_tags_shuffled[id]['CLIPScore'])
        small_tags_shuffled_scores.append(small_tags_shuffled[id]['CLIPScore'])

    min = np.min(all_scores)
    max = np.max(all_scores)
    range = max-min
    print(min, max, range)
    print(np.mean((big_tags_scores-min)/range), np.std((big_tags_scores-min)/range))
    print(np.mean((small_tags_scores-min)/range), np.std((small_tags_scores-min)/range))
    print(np.mean((big_tags_shuffled_scores-min)/range), np.std((big_tags_shuffled_scores-min)/range))
    print(np.mean((small_tags_shuffled_scores-min)/range), np.std((small_tags_shuffled_scores-min)/range))
    print('sample_size ', len(big_tags_scores), len(small_tags_scores))
    print('t-test small/big: ', scipy.stats.ttest_ind((small_tags_scores-min)/range,(big_tags_scores-min)/range))
    print('t-test small/shuffled: ', scipy.stats.ttest_ind((small_tags_scores-min)/range,(small_tags_shuffled_scores-min)/range))
    print('t-test big/shuffled: ', scipy.stats.ttest_ind((big_tags_scores-min)/range,(big_tags_shuffled_scores-min)/range))
    print('t-test small/big-shuffled: ', scipy.stats.ttest_ind((small_tags_scores-min)/range,(big_tags_shuffled_scores-min)/range))
    print('t-test big/small-shuffled: ', scipy.stats.ttest_ind((big_tags_scores-min)/range,(small_tags_shuffled_scores-min)/range))
    
    print('**********')
    for big, small, big_shuffled, small_shuffled in zip(big_tags_umic, small_tags_umic, big_tags_shuffled_umic, small_tags_shuffled_umic):
        qid_str = str(big['question_id'])

    print('almost_equal_big', almost_equal_big/len(small_tags_shuffled)*100, '%')
    print('almost_equal_small', almost_equal_small/len(small_tags_shuffled)*100, '%')
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