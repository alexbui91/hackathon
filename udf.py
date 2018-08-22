def get_limit(seq, limit):
    if len(seq) > limit:
        return seq[:limit]
    else:
        return seq


def fix_float(x):
    res = 0.0
    try:
        if x and x != "\N" and x != "":
            res = float(x)
    except Exception as e:
        print(x)
    return res