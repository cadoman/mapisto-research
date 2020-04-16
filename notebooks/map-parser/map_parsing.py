import numpy as np
import skimage.color, skimage.measure, skimage.filters, skimage.feature
from shapely.geometry import Polygon, box, MultiPolygon
from tqdm.notebook import tqdm
import shapely.ops

class Territory :
    def __init__(self, color, polygon):
        assert color is None or isinstance(color, (list, np.ndarray))
        assert isinstance(polygon, Polygon)
        if color is None:
            color = np.random.rand(3)
        self.color = np.array(color)
        self.polygon = polygon
        
    def partially_covered_by(self, surface):
        if not surface.is_valid:
            surface = surface.buffer(0)
        if not surface.is_valid:
            raise Exception("ca marche toujours pas")
            
        return self.polygon.intersection(surface).area> .2* self.polygon.area

    def roughly_same_as(self, other_territory):
        intersection_area = self.polygon.intersection(other_territory.polygon).area
        return intersection_area > .9*self.polygon.area and intersection_area > .9* other_territory.polygon.area


def colour_distance(rgb1, rgb2):
    [[lab1, lab2]] = skimage.color.rgb2lab([[rgb1, rgb2]])
    return np.linalg.norm(lab1 - lab2)

def extract_major_colors(map_image):
    one_every_few_rows =map_image[::2, ::2]
    count_colors = np.unique(one_every_few_rows.reshape(int(one_every_few_rows.size/3), 3) , axis=0,return_counts=True)
    stacked = np.column_stack((count_colors[0][:,0], count_colors[0][:,1], count_colors[0][:,2], count_colors[1]))
    stacked = [tuple(row) for row in stacked]
    stacked = np.array(stacked, dtype=[('r', float), ('g', float), ('b', float),  ('count', int)])
    major_colors = np.sort(stacked, order='count')[::-1]
    major_colors = [[tup[0], tup[1], tup[2]] for tup in major_colors][:300]
    res = []
    for color in major_colors:
            distances = [colour_distance(color, c) for c in res]
            if not distances or min(distances) > 10:
                res.append(color)
    return np.array(res)

def regroup_img_colors(img):
    major_colors = extract_major_colors(img)
    flat_img=img.reshape((int(img.size/3), 3))
    distance_matrix = ((np.repeat(flat_img, len(major_colors), axis=0).reshape(len(flat_img), len(major_colors), 3) - major_colors )**2).sum(axis=2)
    flat_clustered = major_colors[distance_matrix.argmin(axis=1)]
    return (flat_clustered.reshape(img.shape), major_colors)

def extract_relevant_colors(gray_img, forbidden_colors):
    count_colors = np.unique(gray_img, return_counts=True)
    count_colors = np.array(list(zip(count_colors[0], count_colors[1])), dtype=[('color', float), ('count', int)])
    grayscale_major_colors = np.sort(count_colors, order='count')[::-1]['color']
    sea_color = grayscale_major_colors[0]
    return [col for col in grayscale_major_colors if abs(col-sea_color)>.005]

def get_polygons(mask):
    contours = skimage.measure.find_contours(mask, 0.8)
    return [Polygon(list(zip(contour[:, 1], contour[:, 0]))) for contour in contours if len(contour) > 2]

def is_contained(polygon, other_polygons):
    for potential_container in other_polygons:
        if polygon.within(potential_container):
            return True
    return False

def extract_territories_by_color(clustered_image, colors_on_cluster, ignored_colors):
    distance_from_ignored = lambda color :  min([colour_distance(color, ignored) for ignored in ignored_colors])
    relevant_colors = [color for color in colors_on_cluster if distance_from_ignored(color) > .005]
    print(f'ignored {len(colors_on_cluster) - len(relevant_colors)} based on ignored_colors ')
    [relevant_colors_in_gray] = skimage.color.rgb2gray(np.array([relevant_colors]))

    grayscale_map = skimage.color.rgb2gray(clustered_image)
    mask_all = [grayscale_map==gray_color for gray_color in relevant_colors_in_gray]
    mask_all = [skimage.filters.median(mask, selem=np.ones((3, 3))) for mask in mask_all]
    territories = []
    for i, mask in enumerate(mask_all):
        territories = territories + [Territory(relevant_colors[i], p) for p in get_polygons(mask)]
    
    territories = [t for t in territories if t.polygon.area >60]
    territories = [t for t in territories if t.polygon.area/t.polygon.length >2]
    territories = [t for t in territories if t.polygon.area < clustered_image.shape[0]*clustered_image.shape[1]*.95 ]
    other_polygons = lambda t : [o.polygon for o in territories if o is not t]
    territories = [t for t in territories 
                if not is_contained(t.polygon,other_polygons(t))]
    return territories


def extract_color_and_mask(img, polygon):
    res = img.copy()
    coords = np.array(polygon.exterior.coords)
    rr, cc = skimage.draw.polygon(coords[:, 1], coords[:, 0])
    mask = np.ones(img.shape[:2]).astype(bool)
    mask[rr, cc] = False
    color= np.median(img[rr, cc], axis=0)
    res[mask] = 0
    return color, res

def remove_wrapper_polygons(polygons):
    with_area=np.array([[pol, pol.area] for pol in polygons])
    polygons_sorted = with_area[with_area[:,1].argsort()][::-1][:,0]
    to_remove = []
    for i, potential_wrapper in enumerate(polygons_sorted[:-1]):
        smaller_pols = polygons_sorted[i+1:]
        potential_contained = shapely.ops.unary_union(smaller_pols)
        intersection_area = potential_wrapper.intersection(potential_contained).area
        ratio = intersection_area/potential_wrapper.area
        if ratio > .6:
            to_remove.append(i)
    return [pol for i,pol in enumerate(polygons) if i not in to_remove]

def remove_contained(atomic_polygons):
    to_remove = []
    for pol in atomic_polygons:
        other = [p for p in atomic_polygons if p is not pol]
        if len(other)!=len(atomic_polygons) -1:
            raise Exception("wsh")
        common_area = [pol.intersection(o).area for o in other]
        if max(common_area) > .9 * pol.area:
            to_remove.append(pol)
    return [p for p in atomic_polygons if p not in to_remove]


def split_territory(territory_color, masked_img):
    edges = skimage.feature.canny(skimage.color.rgb2gray(masked_img), sigma=.5)
    fat_edges = skimage.morphology.dilation(edges, skimage.morphology.disk(1))
    inner_polygons = [p for p in get_polygons(fat_edges) if p.area > 60]
    if len(inner_polygons) > 1:
        inner_polygons_atomic = remove_wrapper_polygons(inner_polygons)
    else :
        inner_polygons_atomic = inner_polygons
    if len(inner_polygons_atomic) > 1:
        inner_polygons_atomic = remove_contained(inner_polygons_atomic)
    return inner_polygons_atomic


def extract_territories_by_gradient(territories, original_image, show_progress=True):
    res = []
    if show_progress :
        territories = tqdm(territories, desc="extract territories by gradient")
    for territory in territories:
        color, masked = extract_color_and_mask(original_image, territory.polygon)
        inner_polygons = split_territory(color, masked)
        if len(inner_polygons) > 1:
            for pol in inner_polygons:
                res.append(Territory(color, pol))
        else:
            res.append(territory)
    return res
    
def color_analysis(image, ignored_colors):
    clustered_img, major_colors = regroup_img_colors(image)
    territories_by_color = extract_territories_by_color(clustered_img, major_colors, ignored_colors)
    return territories_by_color
