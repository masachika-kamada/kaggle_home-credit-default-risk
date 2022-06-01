import os


def type_convert(data):
    with open(os.path.dirname(__file__) + "/type.txt") as f:
        raw = f.read()
    types = raw.split("dtype=")[1:]
    dict_type_convert = {}
    for i, t in enumerate(types):
        t = t.split("\"")[1]
        if t != "object":
            dict_type_convert[i] = t

    print(dict_type_convert)

    for d in data:
        for idx in dict_type_convert:
            convert_to = dict_type_convert[idx]
            if "int" in convert_to:
                if d[idx] != "":
                    d[idx] = int(d[idx])
                else:
                    d[idx] = None
            elif "float" in convert_to:
                if d[idx] != "":
                    d[idx] = float(d[idx])
                else:
                    d[idx] = None
