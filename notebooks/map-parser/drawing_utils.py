import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import skimage.draw
from shapely.geometry import MultiPolygon
from map_parsing import Territory
import shapely.affinity

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

def display_colours(color_list) :
    size = len(color_list)
    plt.figure(figsize=(18,8))
    plt.bar(range(size), [1]*size  , color=color_list, )
    plt.show()



def draw_contour(img, polygon, color, thickness=1, do_copy=True):
    if isinstance(polygon, (list, np.ndarray)) and not len(polygon):
        return img
    
    if isinstance(polygon, MultiPolygon):
        return draw_contour(img, list(polygon.geoms), color, thickness)
    
    if isinstance(polygon, Territory):
        return draw_contour(img, polygon.polygon, color, thickness)
    
    res = img.copy() if do_copy else img
    if isinstance(polygon, (list, np.ndarray)):
        if not len(polygon):
            return img
        else:
            drawnFirst = draw_contour(res, polygon[0], color, thickness, False)
            return draw_contour(drawnFirst, polygon[1:], color, thickness, False)
    polygon = shapely.affinity.scale(polygon, xfact=.5, yfact=.5, origin=(0,0))
    mask = np.zeros(res.shape[:2]).astype(bool)
    coords = np.array(polygon.exterior.coords)
    rr, cc = skimage.draw.polygon_perimeter(coords[:, 1], coords[:, 0], shape=mask.shape)
    mask[rr,cc] = True
    if thickness > 1:
        mask = skimage.morphology.dilation(mask, selem=skimage.morphology.disk(thickness))
    try :
        res[mask] = color
        return res
    except ValueError as e:
        print('res shape ', res.shape)
        print('mask shape  ', mask.shape)
        print('mask', mask)
        print('color : ', color)
        raise e

def draw_plain(img, polygon, color=None, do_copy=True):   
    if isinstance(polygon, MultiPolygon):
        return draw_plain(img, list(polygon.geoms), color)
    
    if isinstance(polygon, Territory):
        return draw_plain(img, polygon.polygon, color if color else polygon.color)
    
    res = img.copy() if do_copy else img
    if isinstance(polygon, (list, np.ndarray)):
        for p in tqdm(polygon, leave=False, desc="Drawing shapes..."):
            res = draw_plain(res, p, color, False)
        return res

    if color is None :
        color = np.random.rand(3)
    polygon = shapely.affinity.scale(polygon, xfact=.5, yfact=.5, origin=(0,0))
    coords = np.array(polygon.exterior.coords)
    rr, cc = skimage.draw.polygon(coords[:, 1], coords[:, 0], shape=img.shape[:2])
    try :
        res[rr, cc] = color
    except ValueError as e:
        print('res shape ', res.shape)
        print('rr shape  ', rr.shape)
        print('color : ', color)
        raise e
    return res


def draw_territories(base_img, *args):
    res = base_img[::2, ::2].copy() # divide area by 4
    title=""
    for arg in args:
        if isinstance(arg, str):
            title = arg
        else:
            (shapes, draw_type, color) = arg
            if color=='red':
                color =  [1,0,0]
            elif color=='green':
                color = [0,1,0]
            elif color=='blue' :
                color=[0,0,1]
            elif color=='white':
                color=[1,1,1]
            if draw_type=='stroke':
                res = draw_contour(res, shapes, color, False)
            elif draw_type == 'fill':
                res = draw_plain(res, shapes, color, False)
            else :
                raise Exception(f'Unknown draw type : {draw_type}')
    display_img(res, title)
