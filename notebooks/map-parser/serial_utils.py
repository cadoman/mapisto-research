import json
from shapely.geometry import mapping, shape

def save_results(results, filename):
    serializable_results = [
        {
            "color" : list(col.astype(float)),
            "polygon" : mapping(pol)
        } for col,pol in results
    ]
    json.dump(serializable_results, open(filename, 'w'))

def load_results(filename):
    data = json.load(open(filename, 'r'))
    return [(datum['color'], shape(datum['polygon'])) for datum in data]