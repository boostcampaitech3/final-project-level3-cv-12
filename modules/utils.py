from .calcIou import checkIou2


def change_state(state1, state2):
    state_dict = {'zero' : {'new' : 'new',
                            'sub' : 'sub', #??
                            'add' : 'add', #??
                            'zero':'zero' #??
                           },
                  'new' : {'zero' : 'zero',
                           'sub' : 'sub',
                           'add' : 'add',
                           'new' : 'new' #??
                          },
                  'sub' : {'zero' : 'zero',
                           'sub' : 'sub',
                           'add' : 'add',
                           'new' : 'new' #??
                          },
                  'add' : {'zero' : 'zero',
                           'sub' : 'sub',
                           'add' : 'add',
                           'new' : 'new' #??
                          }
                 }
    
    return state_dict[state1][state2]
    # zero -> zero: 불가
    # zero -> new : new 변경
    # zero -> sub: 불가
    # zero -> add : 불가

    # new -> zero: zero 변경
    # new -> new : 불가
    # new -> sub: sub 변경
    # new -> add: add 변경

    # sub -> zero: zero 변경
    # sub -> new : 불가
    # sub -> sub : sub 유지
    # sub -> add : add 변경

    # add -> zero: zero 변경
    # add -> new: 불가
    # add -> sub: sub 변경
    # add -> add: add 유지
    
    
def sum_bbox(bboxes):
    xmin, ymin, xmax, ymax = 1e8, 1e8, 0, 0
    for x, y, w, h in bboxes:
        xmin = min(x, xmin)
        ymin = min(y, ymin)
        xmax = max(x+w, xmax)
        ymax = max(y+h, ymax)
    return xmin, ymin, xmax-xmin, ymax-ymin


def combine_state(state_cum, state_now, threshold=0.4):
    if not state_cum:
        return state_now
    
    state_sum = set()
    
    for bbox1, state1 in state_cum:
        added = 0
        discards = set()
        for bbox2, state2 in state_now:
            iou = checkIou2(bbox1, bbox2)
            if iou > threshold:
                try:
                    state = change_state(state1, state2)
                except:
                    print(f'{state1}, {state2}')
                    raise ValueError
                state_sum.add((bbox2, state))
                discards.add((bbox2, state2))
                added = 1
                            
        if not added:
            state_sum.add((bbox1, state1))
    
    for discard in discards:
        state_now.discard(discard)

    state_sum = state_sum | state_now
    
    return state_sum