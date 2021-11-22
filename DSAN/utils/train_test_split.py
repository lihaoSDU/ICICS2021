import re
import os
import random
import shutil

if __name__ == '__main__':
    path = '../../datasets/VisDA-2017/trainset'
    for root, dirs, files in os.walk(path):

        if (len(root.split('/'))) == 6:
            img_list = os.listdir(root)
            random.shuffle(img_list)
            for i in range(0, int(len(img_list) * 0.9)):
                train_path = os.path.join(os.path.split(root)[0], 'train_set', os.path.split(root)[-1])
                if not os.path.exists(train_path):
                    os.makedirs(train_path)
                print(os.path.join(root, img_list[i]))
                shutil.copy(os.path.join(root, img_list[i]), train_path)

            for j in range(int(len(img_list) * 0.9), len(img_list)):
                test_path = os.path.join(os.path.split(root)[0], 'test_set', os.path.split(root)[-1])
                if not os.path.exists(test_path):
                    os.makedirs(test_path)
                print(os.path.join(root, img_list[j]))
                shutil.copy(os.path.join(root, img_list[j]), test_path)

# path = '../../datasets/VisDA-2017/validation/test'
#
# for root, dirs, files in os.walk(path):
#     if (len(root.split('/'))) == 7:
#         for idx, file in enumerate(files):
#             if idx < len(files) / 2:
#                 print(idx, os.path.join(root, file), re.sub('validation/test', 'validation/train', root))
#                 # shutil.move(os.path.join(root, file), re.sub('validation/test', 'validation/train', root))
