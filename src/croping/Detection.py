from timeit import default_timer as timer

import torch
import numpy as np
import os
import pandas as pd
import matplotlib
import cv2

from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from skimage.color import gray2rgb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from util_funcs import crop_image, display_boxes


def yolo_detect(yolo_model, im):
    """Inference the YOLO model

    Args:
        detection_model_path (pt): The path of the trained model
        im (cv2 image): Input view image

    Returns:
        cell_locations(numpy): the locations of the cells
    """

    # Model
    # model = torch.hub.load('ultralytics/yolov5', 'custom', detection_model_path)  # custom trained model
    # Images

    # Inference
    results = yolo_model(im)
    cell_locations = (results.pandas().xywh[0]).iloc[:, 0:4].values

    return cell_locations


def load_yolo(detection_model_path):
    """Load in yolo model

    """

    # Model
    # custom trained model
    return torch.hub.load('ultralytics/yolov5', 'custom', detection_model_path)


def load_classifier(cnn_filename):
    """Initialize a trained classifier

    Args:
        cnn_filename (string path): path to the trained classifier

    Returns:
        model_net: a torch model
    """
    # # Initialize network
    model_net = models.resnet50(pretrained=True)
    model_net = model_net.cuda() if torch.cuda.is_available() else model_net
    num_ftrs = model_net.fc.in_features

    model_net.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 3))
    model_net.fc = model_net.fc.cuda() if torch.cuda.is_available() else model_net.fc
    trained_model_PATH = cnn_filename

    #  Load model details

    if torch.cuda.is_available():
        model_net.load_state_dict(torch.load(trained_model_PATH))
    else:
        model_net.load_state_dict(torch.load(
            trained_model_PATH, map_location=torch.device('cpu')))

    model_net.eval()

    return model_net


def CNN_classify(cnn_classifier, single_cell_crops, trans, device):
    """Inference classifier model

    Args:
        cnn_classifier (torch model): _description_
        single_cell_crops (torch tensor): a batch of images
        input_size (int, optional): _description_. Defaults to 64.

    Returns:
        torch tensors: predicted types and predicted scores
    """

    with torch.no_grad():
        label_pred = []
        model_input = trans(single_cell_crops).to(device)
        # model_input = single_cell_crops
        output = cnn_classifier(model_input.float())
        score_output = F.softmax(output)
        score, label_pred = torch.max(score_output, 1)
        

    return label_pred, score.data


def display_time_cost(time_yolo, time_crop, time_cnn_pre, time_cnn):
    font = {'family': 'Arial',
            'size': 18
            }
    plt.rc('axes', linewidth=2)
    plt.rc('font', **font)
    time_total = np.array(time_yolo) + np.array(time_crop) + \
        np.array(time_cnn_pre) + np.array(time_cnn)
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(time_yolo[1:], label="YOLO Inference")
    ax.plot(time_crop[1:],  label="Cropping")
    ax.plot(time_cnn_pre[1:], label="CNN Preprocess")
    ax.plot(time_cnn[1:], label="CNN Inference")
    ax.plot(time_total[1:], label="total time")

    ax.set_xlabel("Image Frames")
    ax.set_ylabel("Time (Seconds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("Time_Cost.png")


def detect_and_classify(cnn_model_path, yolo_path, img_dir, is_save_to_local=True):
    """Detect with Yolo and Classify with CNN
        and then save to local

    Args:
        cnn_model_path (string): path to cnn model
        yolo_path (string): path to yolo model
        img_dir (string): directory of images to train

    Returns:
        results: Results in List
        results: Results in df
        results_arr: results in np.array

    """
    # cnn_model_path = '../checkpoints/PC3PBMC.pt'
    # yolo_path = "../checkpoints/yolov5.pt"

    cnn_classifier = load_classifier(cnn_model_path)
    yolo_model = load_yolo(yolo_path)
    time_yolo, time_crop, time_cnn_pre, time_cnn = [], [], [], []
    input_size = 64
    trans = transforms.Compose([transforms.Resize((input_size, input_size))])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # img_dir = "../data/June22NewData/MLValidation/PC3_PBMCs/1_to_10_10M_in_1mL/BF/"

    im = plt.imread(os.path.join(img_dir, os.listdir(img_dir)[0]))
    if len(im.shape)> 2:
        imgs = [plt.imread(os.path.join(img_dir, fileName))[:, :, :3] for fileName in (
            os.listdir(img_dir)) if fileName.endswith(".tif")]
    else:
        imgs = [plt.imread(os.path.join(img_dir, fileName)) for fileName in (
            os.listdir(img_dir)) if fileName.endswith(".tif")]
    # if imgs.dtype == np.dtype('uint8'):

    results = []
    df_total = []
    if not imgs:
        return 0
    else:
        image_height = imgs[0].shape[0]
        image_width = imgs[0].shape[1]
        imgs = [(im * 255).astype(np.uint8)
                for im in imgs] if imgs[0].dtype != np.uint8 else imgs

        for k, im in enumerate(imgs):
            print("Image Number # ", k)
            start = timer()
            start_ini = timer()
            cell_locations = yolo_detect(yolo_model, im)
            end = timer()
            time_yolo.append(end - start)
            print("YOLO time", end - start)

            start = timer()
            img_crops = crop_image(
                im, cell_locations, image_width, image_height)

            # This is for saving

            # [plt.imsave(os.path.join(img_dir, f'_MPP_{k}_{crop_id}.png'), single_crop,
            #             cmap='gray') for crop_id, single_crop in enumerate(img_crops)]
            end = timer()
            time_crop.append(end - start)
            print("Crops time", end - start)

            start = timer()
            if len(img_crops[0].shape) < 3:
                resized_img_crops = [
                    gray2rgb(resize(crop, [input_size, input_size])) for crop in img_crops]
            else:
                resized_img_crops = [
                    resize(crop, [input_size, input_size]) for crop in img_crops]
            resized_img_crops = torch.from_numpy(
                np.array(resized_img_crops))  # (B, W, H, 3)
            resized_img_crops = resized_img_crops.permute(
                0, 3, 1, 2)  # (B, 3, W, H)
            end = timer()
            time_cnn_pre.append(end - start)
            print("CNN prepare time", end - start)

            start = timer()
            pred_cell_class, pred_prob = CNN_classify(
                cnn_classifier, resized_img_crops, trans, device)
            end = timer()
            time_cnn.append(end - start)
            print("CNN inference time",  end - start)

            print(end - start_ini)  # Time in seconds, e.g. 5.38091952400282

            cell_IDs = k * np.ones([pred_cell_class.shape[0], 1])
            pred_classes = pred_cell_class.cpu().numpy().reshape(-1, 1)
            pred_scores = pred_prob.cpu().numpy().reshape(-1, 1)
            result_of_one_view = np.concatenate(
                [cell_IDs, cell_locations, pred_classes, pred_scores], 1)

            results.append(result_of_one_view)

            df = pd.DataFrame(result_of_one_view, columns = ['Cell_ID','X', "Y", "W", "H", "Predicted_Cell_Type", "Probability"])

            boxfig = display_boxes(
                im, cell_locations, pred_classes, pred_scores)

            if is_save_to_local:
                boxfig.savefig(f"0911old_{os.listdir(img_dir)[k]}.png")

            # if is_save_to_local: df.to_csv(f"result_{task}_{os.listdir(img_dir)[k]}.csv")
    results_arr = np.concatenate(results, 0)
    df_total = pd.DataFrame(results_arr, columns=[
                            'Cell_ID', 'X', "Y", "W", "H", "Predicted_Cell_Type", "Probability"])
    
    # df_total.to_csv("ML_results_0922_Lin+_Cycle01.csv")
    display_time_cost(time_yolo, time_crop, time_cnn_pre, time_cnn)

    pass
    return results, df_total, results_arr


def detect_crop(yolo_path, img_dir, out_dir):

    yolo_model = load_yolo(yolo_path)
    time_yolo, time_crop, time_cnn_pre, time_cnn = [], [], [], []
    input_size = 64



    # imgs= [plt.imread(os.path.join(img_dir, fileName))[:, :] for fileName in (os.listdir(img_dir))
    #         if fileName.endswith(".tif")]
    img_filenames = [ fileName for fileName in (os.listdir(img_dir)) if fileName.endswith(".tif")]
    imgs= [plt.imread(os.path.join(img_dir, fileName))[:, :] for fileName in img_filenames]
    if not imgs:
        return 0
    else:
        image_height = imgs[0].shape[0]
        image_width = imgs[0].shape[1]
        imgs = [(im * 255).astype(np.uint8)
                for im in imgs] if imgs[0].dtype != np.uint8 else imgs

        for k, im in enumerate(imgs):
            start = timer()
            start_ini = timer()
            cell_locations = yolo_detect(yolo_model, im)
            end = timer()
            time_yolo.append(end - start)
            print("YOLO time", end - start)

            start = timer()
            img_crops = crop_image(
                im, cell_locations, image_width, image_height)

            # This is for saving
            for crop_id, single_crop in enumerate(img_crops):
                fileName = f'{img_filenames[k][:-4]}_{crop_id}.tif'
                filePath = os.path.join(out_dir, fileName)
                cv2.imwrite(filePath, single_crop)



if __name__ == "__main__":

    
    yolo_path = "./src/Preprocessing/Yolo_model/yolov5.pt"

    root_dir = "./Input/"
    out_dir = "./Output"
    batch_ID = "2022-11-01_processed"
    
    img_dir = os.path.join(root_dir, batch_ID)
    out_dir = os.path.join(out_dir, batch_ID)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    detect_crop(yolo_path, img_dir, out_dir) # do detect crop instead of detect and classify!
    # detect_and_classify(cnn_model_path, yolo_path, img_dir)
    # df = pd.read_csv("ML_results_NonHSC.csv")