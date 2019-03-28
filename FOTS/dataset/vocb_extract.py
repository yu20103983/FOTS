#coding=utf-8
import os, sys, csv
icdarFolder = sys.argv[1]
vocb_flie = sys.argv[2]

def GetAllFiles(path):
    """
    获得path目录（文件）下的所有的文件序列，包括子目录
    """
    listFiles = []
    if not os.path.exists(path):
        return listFiles
    if os.path.isfile(path):
        listFiles.append(path)
        return listFiles
    pt = os.walk(path)
    for t in pt:
        if len(t[2]) > 0:
            listFiles.extend([os.path.join(t[0], fileName) for fileName in t[2]])
    return listFiles

def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32), text_tags
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '###':
                text_tags.append(None)
            else:
                text_tags.append(label.decode("utf-8"))

        return text_tags

strs = [load_annoataion(p) for p in GetAllFiles(icdarFolder)]
strs = [j for s in strs for j in s if j]
strs = [k for j in strs if len(j.strip()) > 0 for k in j.strip()]
# print strs

def all_list(arr):
    result = {}
    for i in set(arr):
        result[i] = arr.count(i)
    return result
strs_count = all_list(strs)
strs_count = sorted(strs_count.items(), key=lambda k: k[1])[::-1]
strs = [i[0] for i in strs_count]
s = '\n'.join(strs)
open(vocb_flie, 'w').write(s.encode('utf-8'))
