import numpy as np
from shapely.geometry import Polygon

'''
TODO: Merge with handle_textboxes somehow for higher efficiency.
TODO: Use some pattern to make this script concise, especially for 'std_intersection' & 'area_aware_intersection',
  standard_nms & area_aware_nms.
'''
def std_intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

def area_aware_intersection(g, p, vertical_iou_thresh = 0.5):
    g_poly = Polygon(g[:8].reshape((4, 2)))
    p_poly = Polygon(p[:8].reshape((4, 2)))
    if not g_poly.is_valid or not p_poly.is_valid:
        return 0
    inter = g_poly.intersection(p_poly)

    # when in the same column, do not merge
    # xmin, ymin, xmax, ymax = inter.bounds
    if inter and \
        inter.area / min(g_poly.area, p_poly.area) and \
        (inter.bounds[3] - inter.bounds[1]) / \
        max(g_poly.bounds[3]-g_poly.bounds[1], \
            p_poly.bounds[3]-p_poly.bounds[1]) \
        < vertical_iou_thresh:
        return 0

    union = g_poly.area + p_poly.area - inter.area
    if union == 0:
        return 0
    # else:
    #     return inter/union
    
    min_ovr = inter.area * 1.0 / min(g_poly.area, p_poly.area)
    # if min_ovr < 0.8:
    #     return inter/union
    # else:
    #     return min_ovr
    return min_ovr


def weighted_merge(g, p, iou_thresh = 0.5):
    # if g[8] > p[8]:
    #     return g
    # else:
    #     return p
    g_poly = Polygon(g[:8].reshape((4, 2)))
    p_poly = Polygon(p[:8].reshape((4, 2)))

    # print g[8], p[8]

    # if p[2] - p[0] < 200 and p_poly.intersection(g_poly).area/min(g_poly.area, p_poly.area) > iou_thresh:
    #     g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    #     g[8] = (g[8] + p[8])
    #     return g

    g[0] = min(g[0], p[0]) #min(x)
    g[1] = (g[1] * g[8] + p[1] * p[8]) / (g[8] + p[8])#miny
    g[2] = max(g[2], p[2]) #max(x)
    g[3] = (g[3] * g[8] + p[3] * p[8]) / (g[8] + p[8])#maxy
    g[4] = max(g[4], p[4]) 
    g[5] = (g[5] * g[8] + p[5]  * p[8]) / (g[8] + p[8])
    g[6] = min(g[6], p[6]) 
    g[7] = (g[7] * g[8] + p[7]* p[8]) / (g[8] + p[8])
    g[8] = (g[8] + p[8])

    return g

    # width = np.mean([g[2]-g[0], p[2]-p[0]])

    # if width < 300:
    #     g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    #     g[8] = (g[8] + p[8])
    #     return g

    # g[0] = min(g[0], p[0]) #min(x)
    # g[1] = (g[1] * g[8] + p[1] * p[8]) / (g[8] + p[8])#miny
    # g[2] = max(g[2], p[2]) #max(x)
    # g[3] = (g[3] * g[8] + p[3] * p[8]) / (g[8] + p[8])#maxy
    # g[4] = max(g[4], p[4]) 
    # g[5] = (g[5] * g[8] + p[5]  * p[8]) / (g[8] + p[8])
    # g[6] = min(g[6], p[6]) 
    # g[7] = (g[7] * g[8] + p[7]* p[8]) / (g[8] + p[8])
    # g[8] = (g[8] + p[8])

    # return g

def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([std_intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep]

def area_aware_nms(S, thres):
    boxes = sorted(S, key=lambda x: Polygon(np.array(x[:-1]).reshape((4, 2))).area)

    merged_boxes = []
    merged_boxes.append(boxes[0])
    visited = np.zeros((np.array(boxes).shape[0], 1))
    visited[0] = 1

    cur_index = 0

    while not np.all(visited):
        mbox = merged_boxes[cur_index]
        ovr = np.array([[i, area_aware_intersection(mbox, box)] for i, box in enumerate(boxes) if not visited[i]])
        if ovr.shape[0]:
            idxs = ovr[ovr[:, 1] >= thres]

            if idxs.shape[0]:
                for i in idxs[:, 0]:
                    mbox = weighted_merge(mbox, boxes[int(i)])
                    merged_boxes[cur_index] = mbox
                    visited[int(i)] = 1
            else:
                _idx = np.transpose(np.nonzero((visited-1)*(-1)))[0]
                merged_boxes.append(boxes[_idx[0]])
                visited[_idx[0]] = 1
                cur_index += 1
                continue

    return np.array(merged_boxes)

def nms_locality(polys, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    p = None
    for g in polys:
        if p is not None and area_aware_intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    # return standard_nms(np.array(S), thres)
    # S = standard_nms(np.array(S), thres)
    return area_aware_nms(S, thres)

if __name__ == '__main__':
    # 343,350,448,135,474,143,369,359
    print(Polygon(np.array([[343, 350], [448, 135],
                            [474, 143], [369, 359]])).area)