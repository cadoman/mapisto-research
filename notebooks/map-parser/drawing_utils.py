import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import skimage.draw
from shapely.geometry import MultiPolygon

def display_img(img, title="", big=True) : 
    if big : 
        plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()

def fill_box(img, box, color):
    xa, ya, xb, yb = [int(n) for n in box.bounds]
    img[ya:yb, xa:xb] = color

def get_surrounding_colors(img, box):
    xa, ya, xb, yb = [int(n) for n in box.bounds]
    xa -= 1
    ya -=1
    xb+= 1
    yb+= 1
    colors_at_border =np.concatenate((
        img[ya, xa:xb],
        img[yb, xa:xb],
        img[ya:yb, xa],
        img[ya:yb, xb],
    ))
    return np.median(colors_at_border, axis=0)

def draw_polygon_contour(img, polygon, color, thickness=1):
    res = img.copy()
    if isinstance(polygon, MultiPolygon):
        for pol in tqdm(polygon.geoms):
            res = draw_polygon_contour(res, pol, color, thickness)
        
        return res
    else:
        mask = np.zeros(res.shape[:2]).astype(bool)
        coords = np.array(polygon.exterior.coords)
        rr, cc = skimage.draw.polygon_perimeter(coords[:, 1], coords[:, 0], shape=mask.shape, clip=True)
        mask[rr,cc] = True
        if thickness > 1:
            mask = skimage.morphology.dilation(mask, selem=skimage.morphology.disk(thickness))
        res[mask] = color
    return res

def draw_polygons(img, polygons, colors=None):
    assert colors is None or len(colors)==len(polygons)
    if colors is None :
        colors = np.random.rand(len(polygons), 3)
    res = img.copy()
    for i, polygon in tqdm(enumerate(polygons), total=len(polygons)) :
        coords = np.array(polygon.exterior.coords)
        rr, cc = skimage.draw.polygon(coords[:, 1], coords[:, 0])
        res[rr, cc] = colors[i]
    return res

def display_colours(color_list) :
    size = len(color_list)
    plt.figure(figsize=(18,8))
    plt.bar(range(size), [1]*size  , color=color_list, )
    plt.show()
