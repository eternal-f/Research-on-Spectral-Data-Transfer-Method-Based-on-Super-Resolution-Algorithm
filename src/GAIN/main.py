import os
import sys

path0 = r'F:\FCX\0 El Psy Congroo!!\453620'
path1 = r'F:\FCX\0 El Psy Congroo!!\453620' + '\\'
sys.path.append(path1)
# print(sys.path)
files = os.listdir(path0)
# print('files', files)
for filename in files:
    portion = os.path.splitext(filename)
    # print(portion)
    if portion[1] == '.webp':
        newname = portion[0] + '.jpg'
        filename_dir = path1 + filename
        newname_dir = path1 + newname
        os.rename(filename_dir, newname_dir)
