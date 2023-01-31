import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
from matplotlib.patches import Patch

def crop_image(img, boxes, image_width, image_height):
    """Crop images from img according to the list of boxes

    Args:
        img (_np_): the view image to crop
        boxes (_list_): bounding boxes information
        image_width (int): the height of the original view
        image_height (int): _description_
    return:
        img_crops(list): crops of the images
    """
    img_crops = []

    for box in boxes:
        loc_x, loc_y, width, height = box[0], box[1], box[2], box[3]
        crop = im_crop(img, loc_x, loc_y, width, height, image_width, image_height)
        img_crops.append(crop)
    
    return img_crops

def im_crop(frame_img, loc_x, loc_y,  width, height, image_width, image_height):
    """Crop an image with a single piece of box

    Args:
        frame_img (numpy): image to crop
        loc_x (int): _description_
        loc_y (int): _description_
        width (int): _description_
        height (int): _description_
        image_height (int): the height of the original view
        image_width (int): _description_

    Returns:
        roi: a image crop
    """
    box_size = max(width, height)
    # box_size = 128
    
    roi = frame_img[max(int(loc_y - box_size /2 ), 0) : min(int(loc_y + box_size/2), image_height), 
                    max(int(loc_x - box_size/2 ), 0) : min(int(loc_x + box_size/2 ), image_width)]

    return roi

def display_boxes(im, boxes, preds, scores):
    """Draw Boxes on Image According to Prediction
    Args:
        im (NUMPY): Image
        boxes: list:
            x_c (float): Center X
            y_c (float): Center Y
            w (float): Width
            h (float): Height
    """
    font = {'family': 'Arial',
            'size': 30
            }
    plt.rc('axes', linewidth=2)
    plt.rc('font', **font)
    plt.rcParams["font.weight"] = "bold"
    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(12,12))
    # Display the image
    plt.imshow(im, cmap='gray')
    # plt.show()
    # Get the current reference
    ax = plt.gca()
    for box, pred, score in zip(boxes, preds, scores):
        x_c, y_c, w, h = box[0], box[1], box[2], box[3] 
        # Create a Rectangle patch
        # boxcolor = 'r' if pred == 1 else 'g'
        if pred == 0: boxcolor = 'r'
        if pred == 1: boxcolor = 'g'
        if pred == 2: boxcolor = 'b'
        score = round(score.item(), 2)
        rect = Rectangle((x_c - w / 2 , y_c - h / 2), w,h, linewidth=2, edgecolor= boxcolor, facecolor='none')
        # plt.text(x_c - w / 2 , y_c - h / 2, score, fontsize=24)
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    COLORS = ['r', 'g', 'b']
    CELL_CLASSES = ["LT-HSC", "MPP", 'ST-HSC']
    CELL_CLASSES = ['old', 'young']
    handles = [
    Patch(edgecolor=color, label=label, fill=False) 
    for label, color in zip(CELL_CLASSES, COLORS)
    ]
    ax.set_axis_off()
    # ax.legend(handles=handles)
    
    plt.close()
    return fig