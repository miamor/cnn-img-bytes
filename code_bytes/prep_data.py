import os
# from pre_define import sequence_length
sequence_length = 600

# root = '../data/seq'

# for setname in ['train', 'test']:
#     all = {
#         'malware': [],
#         'benign': []
#     }
#     with open('../data/{}_list.txt'.format(setname), 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             lbl = line.split('__')[0]
#             filename = line.split('__')[1].split('.')[0]

#             path = root+'/'+lbl+'/'+filename+'.txt'
#             print('path', path)
#             with open(path, 'r') as fi:
#                 content = fi.read()[:sequence_length*3]

#                 all[lbl].append(content)
    
#     for lbl in all:
#         with open('data_prepared/trim__{}_{}.txt'.format(lbl, setname), 'w') as f:
#             f.write('\n'.join(all[lbl]))






root = '../data/seq'

# for lbl in ['none', 'game_Linh']:
for lbl in ['game_Linh']:
    all = []
    for filename in os.listdir(root+'/'+lbl):
        path = root+'/'+lbl+'/'+filename
        print('path', path)
        with open(path, 'r') as f:
            content = f.read()[:sequence_length*3]

            all.append(content)
            
    with open('data_prepared/trim__{}_.txt'.format(lbl), 'w') as f:
        f.write('\n'.join(all))
