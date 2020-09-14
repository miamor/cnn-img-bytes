import os
import shutil

for setname in ['train', 'test']:
    with open('{}_list.txt'.format(setname), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            lbl = line.split('__')[0]
            name = line.split('__')[1].split('.')[0]
            print('name', name, lbl)
            # shutil.copy('{}/{}_L.png'.format(lbl, name), '{}/{}/{}_L.png'.format(setname, lbl, name))
            # for folder in ['L', 'RGB']:
            #     shutil.copy('image0/{}/{}/{}_{}.png'.format(lbl, folder, name, folder), '{}_{}/{}/{}_{}.png'.format(setname, folder, lbl, name, folder))
            shutil.copy('/media/tunguyen/Devs/MtaAV_stuff/prep_data/asm_final/{}/{}.asm'.format(lbl, name), '/media/tunguyen/Devs/MtaAV_stuff/VAE/assembly/data/asm_final/{}/{}/{}.asm'.format(setname, lbl, name))                