import json
from shapely.geometry import mapping, shape
from map_parsing import Territory

def save_territories(territories, filename):
    serializable_territories = [
        {
            "color" : list(t.color.astype(float)),
            "polygon" : mapping(t.polygon)
        } for t in territories
    ]
    json.dump(serializable_territories, open(filename, 'w'))


def load_territories(filename):
    data = json.load(open(filename, 'r'))
    return [Territory(datum['color'], shape(datum['polygon'])) for datum in data]
