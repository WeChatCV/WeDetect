import os
import time
import json
import random
import argparse
import itertools
import subprocess
import torch
import copy
import torchvision

from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from d_cube import D3
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

from recall import bbox_overlaps, eval_recalls
from recall_precision_densityf1 import evaluate_dataset, print_comparative_metrics


ds_collections = {
    'coco': {
        'ann_path': 'data/coco/annotations/instances_val2017.json',
        'query': """Please detect the "%s" in the image""",
        'img_path': 'data/coco/val2017/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/coco_proposals_all.json',
        'classes_en': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    },
    'refcoco': {
        'ann_path': [
            'eval_refcoco/refcoco_validation.json',
            'eval_refcoco/refcoco_test.json',
            'eval_refcoco/refcoco_testB.json',
            'eval_refcoco/refcocoplus_validation.json',
            'eval_refcoco/refcocoplus_test.json',
            'eval_refcoco/refcocoplus_testB.json',
            'eval_refcoco/refcocog_validation.json',
            'eval_refcoco/refcocog_test.json',
        ],
        'query': """Please detect the "%s" in the image""",
        'img_path': 'data/coco2014/',
        'proposals': 'eval_refcoco/refcoco_proposals_all.json',
    },
    'grefcoco': {
        'ann_path': [
            'data/grefcoco/finetune_grefcoco_val.json',
            'data/grefcoco/finetune_grefcoco_testA.json',
            'data/grefcoco/finetune_grefcoco_testB.json',
        ],
        'query': """Please detect the "%s" in the image""",
        'img_path': 'data/coco2014/train2014/',
        'proposals': 'data/wedetect_ref/eval_proposals/grefcoco_proposals_all.json',
    },
    'humanref': {
        'ann_path': 'data/HumanRef/annotations.jsonl',
        'query': """Please detect the "%s" in the image""",
        'img_path': 'data/HumanRef/images/',
    },
    'd3': {
        'ann_path': [
            'data/d3/d3_json/d3_full_annotations.json',
            'data/d3/d3_json/d3_pres_annotations.json',
            'data/d3/d3_json/d3_abs_annotations.json',
        ],
        'query': """Please detect the "%s" in the image""",
        'img_path': 'data/d3/d3_images/',
        'proposals': 'data/wedetect_ref/eval_proposals/d3_proposals_all.json',
    },
    'odinw35': {
        'prposals': 'data/wedetect_ref/eval_proposals/odinw35_proposals_all.json',
        'query': """Please detect the "%s" in the image""",
        'datasets': {
            # 1
            'AerialMaritimeDrone_large': {
                'ann_path': 'data/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/large/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/large/valid/',
                'classes_en': ['boat', 'car', 'dock', 'jetski', 'lift'],
            },
            # 2
            'AerialMaritimeDrone_tiled': {
                'ann_path': 'data/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/tiled/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/tiled/valid/',
                'classes_en': ['boat', 'car', 'dock', 'jetski', 'lift'],
            },
            # 3
            'AmericanSignLanguageLetters': {
                'ann_path': 'data/ODinW35/AmericanSignLanguageLetters/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/AmericanSignLanguageLetters/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/valid/',
                'classes_en': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
            },
            # 4
            'Aquarium': {
                'ann_path': 'data/ODinW35/Aquarium/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/Aquarium/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/',
                'classes_en': ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'],
            },
            # 5
            'BCCD': {
                'ann_path': 'data/ODinW35/BCCD/BCCD/BCCD.v3-raw.coco/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/BCCD/BCCD/BCCD.v3-raw.coco/valid/',
                'classes_en': ['Platelets', 'RBC', 'WBC'],
            },
            # 6
            'boggleBoards': {
                'ann_path': 'data/ODinW35/boggleBoards/boggleBoards/416x416AutoOrient/export/val_annotations_without_background.json',
                'img_path': 'data/ODinW35/boggleBoards/boggleBoards/416x416AutoOrient/export/',
                'classes_en': ['Q', 'a', 'an', 'b', 'c', 'd', 'e', 'er', 'f', 'g', 'h', 'he', 'i', 'in', 'j', 'k', 'l', 'm', 'n', 'o', 'o ', 'p', 'q', 'qu', 'r', 's', 't', 't\\', 'th', 'u', 'v', 'w', 'wild', 'x', 'y', 'z'],
            },
            # 7
            'brackishUnderwater': {
                'ann_path': 'data/ODinW35/brackishUnderwater/brackishUnderwater/960x540/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/brackishUnderwater/brackishUnderwater/960x540/valid/',
                'classes_en': ['crab', 'fish', 'jellyfish', 'shrimp', 'small_fish', 'starfish'],
            },
            # 8
            'ChessPieces': {
                'ann_path': 'data/ODinW35/ChessPieces/ChessPieces/Chess_Pieces.v23-raw.coco/valid/new_annotations_without_background.json',
                'img_path': 'data/ODinW35/ChessPieces/ChessPieces/Chess_Pieces.v23-raw.coco/valid/',
                'classes_en': ['  ', 'black bishop', 'black king', 'black knight', 'black pawn', 'black queen', 'black rook', 'white bishop', 'white king', 'white knight', 'white pawn', 'white queen', 'white rook'],
            },
            # 9
            'CottontailRabbits': {
                'ann_path': 'data/ODinW35/CottontailRabbits/CottontailRabbits/valid/new_annotations_without_background.json',
                'img_path': 'data/ODinW35/CottontailRabbits/CottontailRabbits/valid/',
                'classes_en': ['rabbit'],
            },
            # 10
            'dice': {
                'ann_path': 'data/ODinW35/dice/dice/mediumColor/export/val_annotations_without_background.json',
                'img_path': 'data/ODinW35/dice/dice/mediumColor/export/',
                'classes_en': ['1', '2', '3', '4', '5', '6'],
            },
            # 11
            'DroneControl': {
                'ann_path': 'data/ODinW35/DroneControl/DroneControl/Drone Control.v3-raw.coco/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/DroneControl/DroneControl/Drone Control.v3-raw.coco/valid/',
                'classes_en': ['follow', 'follow_hand', 'land', 'land_hand', 'null', 'object', 'takeoff', 'takeoff-hand'],
            },
            # 12
            'EgoHands_generic': {
                'ann_path': 'data/ODinW35/EgoHands/EgoHands/generic/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/EgoHands/EgoHands/generic/valid/',
                'classes_en': ['hand'],
            },
            # 13
            'EgoHands_specific': {
                'ann_path': 'data/ODinW35/EgoHands/EgoHands/specific/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/EgoHands/EgoHands/specific/valid/',
                'classes_en': ['myleft', 'myright', 'yourleft', 'yourright'],
            },
            # 14
            'HardHatWorkers': {
                'ann_path': 'data/ODinW35/HardHatWorkers/HardHatWorkers/raw/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/HardHatWorkers/HardHatWorkers/raw/valid/',
                'classes_en': ['head', 'helmet', 'person'],
            },
            # 15
            'MaskWearing': {
                'ann_path': 'data/ODinW35/MaskWearing/MaskWearing/raw/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/MaskWearing/MaskWearing/raw/valid/',
                'classes_en': ['mask', 'no-mask'],
            },
            # 16
            'MountainDewCommercial': {
                'ann_path': 'data/ODinW35/MountainDewCommercial/MountainDewCommercial/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/MountainDewCommercial/MountainDewCommercial/valid/',
                'classes_en': ['bottle'],
            },
            # 17
            'NorthAmericaMushrooms': {
                'ann_path': 'data/ODinW35/NorthAmericaMushrooms/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/new_annotations_without_background.json',
                'img_path': 'data/ODinW35/NorthAmericaMushrooms/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/',
                'classes_en': ['flat mushroom', 'yellow mushroom'],
            },
            # 18
            'openPoetryVision': {
                'ann_path': 'data/ODinW35/openPoetryVision/openPoetryVision/512x512/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/openPoetryVision/openPoetryVision/512x512/valid/',
                'classes_en': ['American Typewriter', 'Andale Mono', 'Apple Chancery', 'Arial', 'Avenir', 'Baskerville', 'Big Caslon', 'Bradley Hand', 'Brush Script MT', 'Chalkboard', 'Comic Sans MS', 'Copperplate', 'Courier', 'Didot', 'Futura', 'Geneva', 'Georgia', 'Gill Sans', 'Helvetica', 'Herculanum', 'Impact', 'Kefa', 'Lucida Grande', 'Luminari', 'Marker Felt', 'Menlo', 'Monaco', 'Noteworthy', 'Optima', 'PT Sans', 'PT Serif', 'Palatino', 'Papyrus', 'Phosphate', 'Rockwell', 'SF Pro', 'SignPainter', 'Skia', 'Snell Roundhand', 'Tahoma', 'Times New Roman', 'Trebuchet MS', 'Verdana'],
            },
            # 19
            'OxfordPets_by_breed': {
                'ann_path': 'data/ODinW35/OxfordPets/OxfordPets/by-breed/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/OxfordPets/OxfordPets/by-breed/valid/',
                'classes_en': ['cat-Abyssinian', 'cat-Bengal', 'cat-Birman', 'cat-Bombay', 'cat-British_Shorthair', 'cat-Egyptian_Mau', 'cat-Maine_Coon', 'cat-Persian', 'cat-Ragdoll', 'cat-Russian_Blue', 'cat-Siamese', 'cat-Sphynx', 'dog-american_bulldog', 'dog-american_pit_bull_terrier', 'dog-basset_hound', 'dog-beagle', 'dog-boxer', 'dog-chihuahua', 'dog-english_cocker_spaniel', 'dog-english_setter', 'dog-german_shorthaired', 'dog-great_pyrenees', 'dog-havanese', 'dog-japanese_chin', 'dog-keeshond', 'dog-leonberger', 'dog-miniature_pinscher', 'dog-newfoundland', 'dog-pomeranian', 'dog-pug', 'dog-saint_bernard', 'dog-samoyed', 'dog-scottish_terrier', 'dog-shiba_inu', 'dog-staffordshire_bull_terrier', 'dog-wheaten_terrier', 'dog-yorkshire_terrier'],
            },
            # 20
            'OxfordPets_by_species': {
                'ann_path': 'data/ODinW35/OxfordPets/OxfordPets/by-species/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/OxfordPets/OxfordPets/by-species/valid/',
                'classes_en': ['cat', 'dog'],
            },
            # 21
            'PKLot': {
                'ann_path': 'data/ODinW35/PKLot/PKLot/640/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/PKLot/PKLot/640/valid/',
                'classes_en': ['space-empty', 'space-occupied'],
            },
            # 22
            'Packages': {
                'ann_path': 'data/ODinW35/Packages/Packages/Raw/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/Packages/Packages/Raw/valid/',
                'classes_en': ['package'],
            },
            # 23
            'PascalVOC': {
                'ann_path': 'data/ODinW35/PascalVOC/PascalVOC/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/PascalVOC/PascalVOC/valid/',
                'classes_en': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
            },
            # 24
            'pistols': {
                'ann_path': 'data/ODinW35/pistols/pistols/export/val_annotations_without_background.json',
                'img_path': 'data/ODinW35/pistols/pistols/export/',
                'classes_en': ['pistol'],
            },
            # 25
            'plantdoc': {
                'ann_path': 'data/ODinW35/plantdoc/plantdoc/416x416/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/plantdoc/plantdoc/416x416/valid/',
                'classes_en': ['Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf', 'Bell_pepper leaf spot', 'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf', 'Potato leaf early blight', 'Potato leaf late blight', 'Raspberry leaf', 'Soyabean leaf', 'Soybean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf', 'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot', 'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf', 'Tomato two spotted spider mites leaf', 'grape leaf', 'grape leaf black rot'],
            },
            # 26
            'pothole': {
                'ann_path': 'data/ODinW35/pothole/pothole/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/pothole/pothole/valid/',
                'classes_en': ['pothole'],
            },
            # 27
            'Raccoon': {
                'ann_path': 'data/ODinW35/Raccoon/Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/Raccoon/Raccoon/Raccoon.v2-raw.coco/valid/',
                'classes_en': ['raccoon'],
            },
            # 28
            'selfdrivingCar': {
                'ann_path': 'data/ODinW35/selfdrivingCar/selfdrivingCar/fixedLarge/export/val_annotations_without_background.json',
                'img_path': 'data/ODinW35/selfdrivingCar/selfdrivingCar/fixedLarge/export/',
                'classes_en': ['biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft', 'trafficLight-Red', 'trafficLight-RedLeft', 'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck'],
            },
            # 29
            'ShellfishOpenImages': {
                'ann_path': 'data/ODinW35/ShellfishOpenImages/ShellfishOpenImages/raw/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/ShellfishOpenImages/ShellfishOpenImages/raw/valid/',
                'classes_en': ['Crab', 'Lobster', 'Shrimp'],
            },
            # 30
            'ThermalCheetah': {
                'ann_path': 'data/ODinW35/ThermalCheetah/ThermalCheetah/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/ThermalCheetah/ThermalCheetah/valid/',
                'classes_en': ['cheetah', 'human'],
            },
            # 31
            'thermalDogsAndPeople': {
                'ann_path': 'data/ODinW35/thermalDogsAndPeople/thermalDogsAndPeople/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/thermalDogsAndPeople/thermalDogsAndPeople/valid/',
                'classes_en': ['dog', 'person'],
            },
            # 32
            'UnoCards': {
                'ann_path': 'data/ODinW35/UnoCards/UnoCards/raw/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/UnoCards/UnoCards/raw/valid/',
                'classes_en': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
            },
            # 33
            'VehiclesOpenImages': {
                'ann_path': 'data/ODinW35/VehiclesOpenImages/VehiclesOpenImages/416x416/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/VehiclesOpenImages/VehiclesOpenImages/416x416/valid/',
                'classes_en': ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck'],
            },
            # 34
            'WildfireSmoke': {
                'ann_path': 'data/ODinW35/WildfireSmoke/WildfireSmoke/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/WildfireSmoke/WildfireSmoke/valid/',
                'classes_en': ['smoke'],
            },
            # 35
            'websiteScreenshots': {
                'ann_path': 'data/ODinW35/websiteScreenshots/websiteScreenshots/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/websiteScreenshots/websiteScreenshots/valid/',
                'classes_en': ['button', 'field', 'heading', 'iframe', 'image', 'label', 'link', 'text'],
            },
        }
    },
    'odinw13': {
        'prposals': 'data/wedetect_ref/eval_proposals/odinw35_proposals_all.json',
        'query': """Please detect the "%s" in the image""",
        'datasets': {
            # 1
            'AerialMaritimeDrone_large': {
                'ann_path': 'data/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/large/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/large/valid/',
                'classes_en': ['boat', 'car', 'dock', 'jetski', 'lift'],
            },
            # 4
            'Aquarium': {
                'ann_path': 'data/ODinW35/Aquarium/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/Aquarium/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/',
                'classes_en': ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'],
            },
            # 9
            'CottontailRabbits': {
                'ann_path': 'data/ODinW35/CottontailRabbits/CottontailRabbits/valid/new_annotations_without_background.json',
                'img_path': 'data/ODinW35/CottontailRabbits/CottontailRabbits/valid/',
                'classes_en': ['rabbit'],
            },
            # 12
            'EgoHands_generic': {
                'ann_path': 'data/ODinW35/EgoHands/EgoHands/generic/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/EgoHands/EgoHands/generic/valid/',
                'classes_en': ['hand'],
            },
            # 17
            'NorthAmericaMushrooms': {
                'ann_path': 'data/ODinW35/NorthAmericaMushrooms/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/new_annotations_without_background.json',
                'img_path': 'data/ODinW35/NorthAmericaMushrooms/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/',
                'classes_en': ['flat mushroom', 'yellow mushroom'],
            },
            # 22
            'Packages': {
                'ann_path': 'data/ODinW35/Packages/Packages/Raw/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/Packages/Packages/Raw/valid/',
                'classes_en': ['package'],
            },
            # 23
            'PascalVOC': {
                'ann_path': 'data/ODinW35/PascalVOC/PascalVOC/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/PascalVOC/PascalVOC/valid/',
                'classes_en': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
            },
            # 24
            'pistols': {
                'ann_path': 'data/ODinW35/pistols/pistols/export/val_annotations_without_background.json',
                'img_path': 'data/ODinW35/pistols/pistols/export/',
                'classes_en': ['pistol'],
            },
            # 26
            'pothole': {
                'ann_path': 'data/ODinW35/pothole/pothole/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/pothole/pothole/valid/',
                'classes_en': ['pothole'],
            },
            # 27
            'Raccoon': {
                'ann_path': 'data/ODinW35/Raccoon/Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/Raccoon/Raccoon/Raccoon.v2-raw.coco/valid/',
                'classes_en': ['raccoon'],
            },
            # 29
            'ShellfishOpenImages': {
                'ann_path': 'data/ODinW35/ShellfishOpenImages/ShellfishOpenImages/raw/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/ShellfishOpenImages/ShellfishOpenImages/raw/valid/',
                'classes_en': ['Crab', 'Lobster', 'Shrimp'],
            },
            # 31
            'thermalDogsAndPeople': {
                'ann_path': 'data/ODinW35/thermalDogsAndPeople/thermalDogsAndPeople/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/thermalDogsAndPeople/thermalDogsAndPeople/valid/',
                'classes_en': ['dog', 'person'],
            },
            # 33
            'VehiclesOpenImages': {
                'ann_path': 'data/ODinW35/VehiclesOpenImages/VehiclesOpenImages/416x416/valid/annotations_without_background.json',
                'img_path': 'data/ODinW35/VehiclesOpenImages/VehiclesOpenImages/416x416/valid/',
                'classes_en': ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck'],
            },
        }
    },
}


class GroundingDataset(torch.utils.data.Dataset):
    

    def __init__(
        self,
        dataset: str,
    ):
        super().__init__()
        self.dataset = dataset
        self.ann = []
        self.query = ds_collections[dataset]['query']

        if dataset == 'coco':
            self.proposals = json.load(open(ds_collections[dataset]['proposals']))
            inverse_id_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
            images = json.load(open(ds_collections[dataset]['ann_path']))['images']
            coco = COCO(ds_collections[dataset]['ann_path'])
            for ann in images:
                gt_bboxes = []
                gt_labels = []
                ann_ids = coco.getAnnIds(imgIds=ann['id'])
                ann_infos = coco.loadAnns(ann_ids)
                for ann_info in ann_infos:
                    if ann_info.get('ignore', False) or ann_info['iscrowd']:
                        continue
                    x1, y1, w, h = ann_info['bbox']
                    gt_bboxes.append([x1, y1, x1 + w, y1 + h])
                    gt_labels.append(str(inverse_id_map[ann_info['category_id']]))
                item = {
                    'id': ann['id'],
                    'image': ann['file_name'],
                    'img_path': ds_collections[dataset]['img_path'],
                    'dataset': self.dataset,
                    'gt_labels': gt_labels,
                    'gt_bboxes': gt_bboxes,
                }
                self.ann.append(item)
            self.query = [copy.deepcopy(self.query) % (classname) for classname in ds_collections[dataset]['classes_en']]
        elif dataset == 'refcoco':
            self.proposals = json.load(open(ds_collections[dataset]['proposals']))
            for ann_path in ds_collections[dataset]['ann_path']:
                data = json.load(open(ann_path))
                sub_dataset = ann_path.split('/')[-1].split('.')[0]
                for ann in data:
                    item = {
                        'id': ann['id'],
                        'image': ann['image'],
                        'img_path': ds_collections[dataset]['img_path'],
                        'dataset': sub_dataset,
                        'query': [copy.deepcopy(self.query) % (ann['conversations'][1]['value'])],
                        'gt_labels': [ann['conversations'][1]['value']],
                        'gt_bboxes': ann['bounding_boxes'],
                    }
                    self.ann.append(item)
        elif dataset == 'grefcoco':
            self.proposals = json.load(open(ds_collections[dataset]['proposals']))
            for j, sub_dataset in enumerate(['val', 'testA', 'testB']):
                coco = COCO(ds_collections[dataset]['ann_path'][j])
                img_ids = coco.getImgIds()
                for i in range(len(img_ids)):
                    img_info = coco.loadImgs([img_ids[i]])[0]
                    ann_ids = coco.getAnnIds(imgIds=img_ids[i])
                    ann_info = coco.loadAnns(ann_ids)
                    bboxes = []
                    for ann in ann_info:
                        if ann.get('ignore', False) or ann['iscrowd']:
                            continue
                        x1, y1, w, h = ann['bbox']
                        bboxes.append([x1, y1, x1 + w, y1 + h])

                    item = {
                        'id': img_info['id'],
                        'image': img_info['file_name'],
                        'img_path': ds_collections[dataset]['img_path'],
                        'dataset': sub_dataset,
                        'query': [copy.deepcopy(self.query) % (img_info['caption'])],
                        'gt_labels': [img_info['caption'] for _ in range(len(bboxes))],
                        'gt_bboxes': bboxes,
                    }
                    self.ann.append(item)
        elif dataset == 'd3':
            self.proposals = json.load(open(ds_collections[dataset]['proposals']))
            for j, sub_dataset in enumerate(['FULL', 'PRES', 'ABS']):
                d3 = D3('data/d3/d3_images', 'data/d3/d3_pkl')
                img_ids = d3.get_img_ids()
                for i in range(len(img_ids)):
                    img_info = d3.load_imgs(img_ids[i])[0]
                    group_ids = d3.get_group_ids(img_ids=[img_ids[i]])
                    sent_ids = d3.get_sent_ids(group_ids=group_ids)
                    sent_list = d3.load_sents(sent_ids=sent_ids)
                    queries = [sent['raw_sent'] for sent in sent_list]
                    query_ids = [sent['id'] for sent in sent_list]
                    item = {
                        'id': img_info['id'],
                        'image': img_info['file_name'],
                        'img_path': ds_collections[dataset]['img_path'],
                        'dataset': sub_dataset,
                        'query': [copy.deepcopy(self.query) % query for query in queries],
                        'gt_labels': query_ids,
                        'gt_bboxes': [],
                    }
                    self.ann.append(item)
        elif dataset == 'humanref':
            self.proposals = {}
            with open(ds_collections[dataset]['ann_path'], 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            for ann in data:
                file_name = ann['image_name'].replace('é', 'é')
                file_name = file_name.replace('ü', 'ü')
                file_name = file_name.replace('łę', 'łę')
                file_name = file_name.replace('å', 'å')
                item = {
                    'id': ann['id'],
                    'image': file_name,
                    'img_path': ds_collections[dataset]['img_path'],
                    'dataset': self.dataset,
                    'query': [copy.deepcopy(self.query) % (ann["referring"])],
                    'gt_labels': [ann["referring"]] * len(ann["answer_boxes"]),
                    'gt_bboxes': ann["answer_boxes"],
                }
                self.proposals[item['image']] = ann['candidate_boxes']
                self.ann.append(item)
        elif dataset == 'odinw35' or dataset == 'odinw13':
            self.proposals = json.load(open(ds_collections[dataset]['prposals']))
            for sub_dataset_name, sub_dataset in ds_collections[dataset]['datasets'].items():
                images = json.load(open(sub_dataset['ann_path']))['images']
                coco = COCO(sub_dataset['ann_path'])
                for ann in images:
                    gt_bboxes = []
                    gt_labels = []
                    ann_ids = coco.getAnnIds(imgIds=ann['id'])
                    ann_infos = coco.loadAnns(ann_ids)
                    for ann_info in ann_infos:
                        if ann_info.get('ignore', False) or ann_info['iscrowd']:
                            continue
                        x1, y1, w, h = ann_info['bbox']
                        gt_bboxes.append([x1, y1, x1 + w, y1 + h])
                        gt_labels.append(str(ann_info['category_id']))
                    item = {
                        'id': ann['id'],
                        'image': ann['file_name'],
                        'img_path': sub_dataset['img_path'],
                        'dataset': sub_dataset_name,
                        'query': [copy.deepcopy(self.query) % (classname) for classname in sub_dataset['classes_en']],
                        'gt_labels': gt_labels,
                        'gt_bboxes': gt_bboxes,
                    }
                    self.ann.append(item)
                

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        ann = self.ann[idx]

        data = {}
        data['id'] = ann['id']
        data['image'] = Image.open(os.path.join(ann['img_path'], ann['image'])).convert('RGB')
        w, h = data['image'].size
        if len(self.proposals[ann['image']]) == 2 and self.dataset != 'humanref':
            data['proposals'] = self.proposals[ann['image']][0][:100]
            data['propsoals_score'] = self.proposals[ann['image']][1][:100]
        else:
            data['proposals'] = self.proposals[ann['image']][:100]
        for i in range(len(data['proposals'])):
            data['proposals'][i][0] = max(0, min(w, data['proposals'][i][0]))
            data['proposals'][i][1] = max(0, min(h, data['proposals'][i][1]))
            data['proposals'][i][2] = max(0, min(w, data['proposals'][i][2]))
            data['proposals'][i][3] = max(0, min(h, data['proposals'][i][3]))
        num_proposals = len(data['proposals'])
        proposal_str = "<object>" * num_proposals
        data['query'] = []
        if self.dataset == 'coco':
            for query in self.query:
                data['query'].append(
                    [{
                        "role": "user",
                        "content": [{"type": "text", "text": query}]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": proposal_str,}]
                    }]
                )
        elif self.dataset == 'refcoco' or self.dataset == 'humanref' or self.dataset == 'odinw35' or self.dataset == 'odinw13' or self.dataset == 'grefcoco' or self.dataset == 'd3':
            for query in ann['query']:
                data['query'].append(
                    [{
                        "role": "user",
                        "content": [{"type": "text", "text": query}]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": proposal_str,}]
                    }]
                )
        data['dataset'] = ann['dataset']
        data['gt_labels'] = ann['gt_labels']
        data['gt_bboxes'] = ann['gt_bboxes']
        for i in range(len(data['gt_bboxes'])):
            data['gt_bboxes'][i][0] = max(0, min(w, data['gt_bboxes'][i][0]))
            data['gt_bboxes'][i][1] = max(0, min(h, data['gt_bboxes'][i][1]))
            data['gt_bboxes'][i][2] = max(0, min(w, data['gt_bboxes'][i][2]))
            data['gt_bboxes'][i][3] = max(0, min(h, data['gt_bboxes'][i][3]))

        return data


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def collate_fn(inputs):
    return inputs



def fast_eval_recall(dataset):
    """Evaluate proposal recall with COCO's fast_eval_recall.

    Args:
        results (List[dict]): Results of the dataset.
        proposal_nums (Sequence[int]): Proposal numbers used for
            evaluation.
        iou_thrs (Sequence[float]): IoU thresholds used for evaluation.
        logger (MMLogger, optional): Logger used for logging the recall
            summary.
    Returns:
        np.ndarray: Averaged recall results.
    """
    
    iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

    proposals = json.load(open(ds_collections[dataset]['proposals']))
    for k, v in proposals.items():
        if len(v) == 2:
            proposals[k] = v[0][:100]
        else:
            proposals[k] = v[:100]
    if dataset == 'coco':
        coco = COCO(ds_collections[dataset]['ann_path'])
        img_ids = coco.getImgIds()
        gt_bboxes = []
        pred_bboxes = []
        for i in range(len(img_ids)):
            file_name = os.path.join(ds_collections['coco']['img_path'], coco.loadImgs([img_ids[i]])[0]['file_name'])
            single_proposals = proposals[file_name]
            pred_bboxes.append(np.array(single_proposals, dtype=np.float32))
            ann_ids = coco.getAnnIds(imgIds=img_ids[i])
            ann_info = coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], iou_thrs)
    
    elif dataset == 'odinw35' or dataset == 'odinw13':
        gt_bboxes = []
        pred_bboxes = []
        for sub_dataset_name, sub_dataset in ds_collections[dataset]['datasets'].items():
            coco = COCO(sub_dataset['ann_path'])
            images = json.load(open(sub_dataset['ann_path']))['images']
            for ann in images:
                file_name = sub_dataset['img_path'] + ann['file_name']
                single_proposals = proposals[file_name]
                pred_bboxes.append(np.array(single_proposals, dtype=np.float32))
                ann_ids = coco.getAnnIds(imgIds=ann['id'])
                ann_info = coco.loadAnns(ann_ids)
                if len(ann_info) == 0:
                    gt_bboxes.append(np.zeros((0, 4)))
                    continue
                bboxes = []
                for ann in ann_info:
                    if ann.get('ignore', False) or ann['iscrowd']:
                        continue
                    x1, y1, w, h = ann['bbox']
                    bboxes.append([x1, y1, x1 + w, y1 + h])
                bboxes = np.array(bboxes, dtype=np.float32)
                if bboxes.shape[0] == 0:
                    bboxes = np.zeros((0, 4))
                gt_bboxes.append(bboxes)

        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], iou_thrs)

    elif dataset == 'grefcoco':
        # validation
        gt_bboxes = []
        pred_bboxes = []
        coco = COCO(ds_collections[dataset]['ann_path'][0])
        img_ids = coco.getImgIds()
        for i in range(len(img_ids)):
            img_info = coco.loadImgs([img_ids[i]])[0]
            ann_ids = coco.getAnnIds(imgIds=img_ids[i])
            ann_info = coco.loadAnns(ann_ids)
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            file_name = ds_collections[dataset]['img_path'] + img_info['file_name']
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("grefcoco_validation")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

        # testA
        gt_bboxes = []
        pred_bboxes = []
        coco = COCO(ds_collections[dataset]['ann_path'][1])
        img_ids = coco.getImgIds()
        for i in range(len(img_ids)):
            img_info = coco.loadImgs([img_ids[i]])[0]
            ann_ids = coco.getAnnIds(imgIds=img_ids[i])
            ann_info = coco.loadAnns(ann_ids)
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            file_name = ds_collections[dataset]['img_path'] + img_info['file_name']
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("grefcoco_testA")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

        # testB
        gt_bboxes = []
        pred_bboxes = []
        coco = COCO(ds_collections[dataset]['ann_path'][2])
        img_ids = coco.getImgIds()
        for i in range(len(img_ids)):
            img_info = coco.loadImgs([img_ids[i]])[0]
            ann_ids = coco.getAnnIds(imgIds=img_ids[i])
            ann_info = coco.loadAnns(ann_ids)
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            file_name = ds_collections[dataset]['img_path'] + img_info['file_name']
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("grefcoco_testB")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)
    

    elif dataset == 'd3':
        # FULL
        gt_bboxes = []
        pred_bboxes = []
        coco = COCO(ds_collections[dataset]['ann_path'][0])
        img_ids = coco.getImgIds()
        for i in range(len(img_ids)):
            img_info = coco.loadImgs([img_ids[i]])[0]
            ann_ids = coco.getAnnIds(imgIds=img_ids[i])
            ann_info = coco.loadAnns(ann_ids)
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            file_name = ds_collections[dataset]['img_path'] + img_info['file_name']
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("d3_FULL")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

        # PRES
        gt_bboxes = []
        pred_bboxes = []
        coco = COCO(ds_collections[dataset]['ann_path'][1])
        img_ids = coco.getImgIds()
        for i in range(len(img_ids)):
            img_info = coco.loadImgs([img_ids[i]])[0]
            ann_ids = coco.getAnnIds(imgIds=img_ids[i])
            ann_info = coco.loadAnns(ann_ids)
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            file_name = ds_collections[dataset]['img_path'] + img_info['file_name']
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("d3_PRES")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

        # ABS
        gt_bboxes = []
        pred_bboxes = []
        coco = COCO(ds_collections[dataset]['ann_path'][2])
        img_ids = coco.getImgIds()
        for i in range(len(img_ids)):
            img_info = coco.loadImgs([img_ids[i]])[0]
            ann_ids = coco.getAnnIds(imgIds=img_ids[i])
            ann_info = coco.loadAnns(ann_ids)
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            file_name = ds_collections[dataset]['img_path'] + img_info['file_name']
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("d3_ABS")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

    
    elif dataset == 'refcoco':
        # refcoco_validation
        with open(ds_collections['refcoco']['ann_path'][1]) as f:
            data = json.load(f)
        
        gt_bboxes = []
        pred_bboxes = []
        for da in data:
            file_name = os.path.join(ds_collections['refcoco']['img_path'], da["image"])
            bboxes = np.array(da['bounding_boxes'], dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("refcoco_validation")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

        # refcoco_test
        with open(ds_collections['refcoco']['ann_path'][1]) as f:
            data = json.load(f)
        
        gt_bboxes = []
        pred_bboxes = []
        for da in data:
            file_name = os.path.join(ds_collections['refcoco']['img_path'], da["image"])
            bboxes = np.array(da['bounding_boxes'], dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("refcoco_test")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

        
        # refcoco_testB
        with open(ds_collections['refcoco']['ann_path'][2]) as f:
            data = json.load(f)
        
        gt_bboxes = []
        pred_bboxes = []
        for da in data:
            file_name = os.path.join(ds_collections['refcoco']['img_path'], da["image"])
            bboxes = np.array(da['bounding_boxes'], dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("refcoco_testB")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

        # refcocoplus_validation
        with open(ds_collections['refcoco']['ann_path'][3]) as f:
            data = json.load(f)
        
        gt_bboxes = []
        pred_bboxes = []
        for da in data:
            file_name = os.path.join(ds_collections['refcoco']['img_path'], da["image"])
            bboxes = np.array(da['bounding_boxes'], dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("refcocoplus_validation")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

        # refcocoplus_test
        with open(ds_collections['refcoco']['ann_path'][4]) as f:
            data = json.load(f)
        
        gt_bboxes = []
        pred_bboxes = []
        for da in data:
            file_name = os.path.join(ds_collections['refcoco']['img_path'], da["image"])
            bboxes = np.array(da['bounding_boxes'], dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("refcocoplus_test")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

        
        # refcocoplus_testB
        with open(ds_collections['refcoco']['ann_path'][5]) as f:
            data = json.load(f)
        
        gt_bboxes = []
        pred_bboxes = []
        for da in data:
            file_name = os.path.join(ds_collections['refcoco']['img_path'], da["image"])
            bboxes = np.array(da['bounding_boxes'], dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("refcocoplus_testB")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

        
        # refcocog_validation
        with open(ds_collections['refcoco']['ann_path'][6]) as f:
            data = json.load(f)
        
        gt_bboxes = []
        pred_bboxes = []
        for da in data:
            file_name = os.path.join(ds_collections['refcoco']['img_path'], da["image"])
            bboxes = np.array(da['bounding_boxes'], dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("refcocog_validation")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)


        # refcocog_test
        with open(ds_collections['refcoco']['ann_path'][7]) as f:
            data = json.load(f)
        
        gt_bboxes = []
        pred_bboxes = []
        for da in data:
            file_name = os.path.join(ds_collections['refcoco']['img_path'], da["image"])
            bboxes = np.array(da['bounding_boxes'], dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
            pred_bboxes.append(np.array(proposals[file_name], dtype=np.float32))

        print("refcocog_test")
        recalls = eval_recalls(gt_bboxes, pred_bboxes, [100], 0.5)

    
    


def eval_coco(ids, pred_bboxes, pred_labels, pred_scores):
    # 加载COCO标注
    coco_gt = COCO(ds_collections['coco']['ann_path'])

    id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                  41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

    # 转换为COCO结果格式
    results = []
    for img_id, pred_bbox, pred_label, pred_score in zip(ids, pred_bboxes, pred_labels, pred_scores):
        for box, label, score in zip(pred_bbox, pred_label, pred_score):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            w = xmax - xmin
            h = ymax - ymin
            
            results.append({
                "image_id": img_id,
                "category_id": id_map[label.item()],
                "bbox": [xmin, ymin, w, h],
                "score": score.item()
            })
    
    # 评估指标计算
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def eval_odinw35(ids, datasets, pred_bboxes, pred_labels, pred_scores):
    # 加载COCO标注
    dataset2coco = {}
    for sub_dataset_name, sub_dataset in ds_collections['odinw35']['datasets'].items():
        dataset2coco[sub_dataset_name] = COCO(sub_dataset['ann_path'])

    id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29, 29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 41, 41: 42, 42: 43, 43: 44, 44: 45, 45: 46, 46: 47, 47: 48, 48: 49, 49: 50, 50: 51, 51: 52, 52: 53, 53: 54, 54: 55, 55: 56, 56: 57, 57: 58, 58: 59, 59: 60, 60: 61, 61: 62, 62: 63, 63: 64, 64: 65, 65: 66, 66: 67, 67: 68, 68: 69, 69: 70, 70: 71, 71: 72, 72: 73, 73: 74, 74: 75, 75: 76, 76: 77, 77: 78, 78: 79}

    dataset2results = {sub_dataset_name: [] for sub_dataset_name in ds_collections['odinw35']['datasets'].keys()}

    # 转换为COCO结果格式
    for img_id, dataset, pred_bbox, pred_label, pred_score in zip(ids, datasets, pred_bboxes, pred_labels, pred_scores):
        for box, label, score in zip(pred_bbox, pred_label, pred_score):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            w = xmax - xmin
            h = ymax - ymin
            
            dataset2results[dataset].append({
                "image_id": img_id,
                "category_id": id_map[label.item()],
                "bbox": [xmin, ymin, w, h],
                "score": score.item()
            })
    
    avg_map = []
    for sub_dataset_name, results in dataset2results.items():
        print(f"Evaluating {sub_dataset_name}...")
        coco_gt = dataset2coco[sub_dataset_name]
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        avg_map.append(coco_eval.stats[0])
    
    print(f"Average mAP across all sub-datasets: {np.mean(avg_map)}")



def eval_odinw13(ids, datasets, pred_bboxes, pred_labels, pred_scores):
    # 加载COCO标注
    dataset2coco = {}
    for sub_dataset_name, sub_dataset in ds_collections['odinw13']['datasets'].items():
        dataset2coco[sub_dataset_name] = COCO(sub_dataset['ann_path'])

    id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29, 29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 41, 41: 42, 42: 43, 43: 44, 44: 45, 45: 46, 46: 47, 47: 48, 48: 49, 49: 50, 50: 51, 51: 52, 52: 53, 53: 54, 54: 55, 55: 56, 56: 57, 57: 58, 58: 59, 59: 60, 60: 61, 61: 62, 62: 63, 63: 64, 64: 65, 65: 66, 66: 67, 67: 68, 68: 69, 69: 70, 70: 71, 71: 72, 72: 73, 73: 74, 74: 75, 75: 76, 76: 77, 77: 78, 78: 79}

    dataset2results = {sub_dataset_name: [] for sub_dataset_name in ds_collections['odinw13']['datasets'].keys()}

    # 转换为COCO结果格式
    for img_id, dataset, pred_bbox, pred_label, pred_score in zip(ids, datasets, pred_bboxes, pred_labels, pred_scores):
        for box, label, score in zip(pred_bbox, pred_label, pred_score):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            w = xmax - xmin
            h = ymax - ymin
            
            dataset2results[dataset].append({
                "image_id": img_id,
                "category_id": id_map[label.item()],
                "bbox": [xmin, ymin, w, h],
                "score": score.item()
            })
    
    avg_map = []
    for sub_dataset_name, results in dataset2results.items():
        print(f"Evaluating {sub_dataset_name}...")
        coco_gt = dataset2coco[sub_dataset_name]
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        avg_map.append(coco_eval.stats[0])
    
    print(f"Average mAP across all sub-datasets: {np.mean(avg_map)}")


def eval_refcoco(ids, datasets, pred_bboxes, pred_labels, pred_scores):
    topk = (1, 5, 10)
    iou_thrs = 0.5
    dataset2score = {
        'refcoco_validation': {k: 0.0 for k in topk},
        'refcoco_test': {k: 0.0 for k in topk},
        'refcoco_testB': {k: 0.0 for k in topk},
        'refcocoplus_validation': {k: 0.0 for k in topk},
        'refcocoplus_test': {k: 0.0 for k in topk},
        'refcocoplus_testB': {k: 0.0 for k in topk},
        'refcocog_validation': {k: 0.0 for k in topk},
        'refcocog_test': {k: 0.0 for k in topk},
    }
    dataset2count = {
        'refcoco_validation': 0.0, 
        'refcoco_test': 0.0, 
        'refcoco_testB': 0.0, 
        'refcocoplus_validation': 0.0, 
        'refcocoplus_test': 0.0, 
        'refcocoplus_testB': 0.0, 
        'refcocog_validation': 0.0, 
        'refcocog_test': 0.0, 
    }

    # refcoco_validation
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcoco_validation':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open(ds_collections['refcoco']['ann_path'][0]) as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = da['bounding_boxes']
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcoco_validation'][k] += 1.0
        dataset2count['refcoco_validation'] += 1.0


    # refcoco_test
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcoco_test':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open(ds_collections['refcoco']['ann_path'][1]) as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = da['bounding_boxes']
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcoco_test'][k] += 1.0
        dataset2count['refcoco_test'] += 1.0

    
    # refcoco_testB
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcoco_testB':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open(ds_collections['refcoco']['ann_path'][2]) as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = da['bounding_boxes']
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcoco_testB'][k] += 1.0
        dataset2count['refcoco_testB'] += 1.0


    # refcocoplus_validation
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcocoplus_validation':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open(ds_collections['refcoco']['ann_path'][3]) as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = da['bounding_boxes']
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcocoplus_validation'][k] += 1.0
        dataset2count['refcocoplus_validation'] += 1.0


    # refcocoplus_test
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcocoplus_test':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open(ds_collections['refcoco']['ann_path'][4]) as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = da['bounding_boxes']
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcocoplus_test'][k] += 1.0
        dataset2count['refcocoplus_test'] += 1.0

    
    # refcocoplus_testB
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcocoplus_testB':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open(ds_collections['refcoco']['ann_path'][5]) as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = da['bounding_boxes']
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcocoplus_testB'][k] += 1.0
        dataset2count['refcocoplus_testB'] += 1.0

    
    # refcocog_validation
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcocog_validation':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open(ds_collections['refcoco']['ann_path'][6]) as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = da['bounding_boxes']
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcocog_validation'][k] += 1.0
        dataset2count['refcocog_validation'] += 1.0


    # refcocog_test
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcocog_test':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open(ds_collections['refcoco']['ann_path'][7]) as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = da['bounding_boxes']
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcocog_test'][k] += 1.0
        dataset2count['refcocog_test'] += 1.0

    
    # summary
    for key, value in dataset2score.items():
        for k in topk:
            try:
                value[k] /= dataset2count[key]
            except Exception as e:
                print(e)

    for key, value in dataset2score.items():
        print(f' Dataset: {key} - Precision @ 1, 5, 10: {sorted([v for k, v in value.items()])}')


def eval_humanref(ids, datasets, pred_bboxes, pred_labels, pred_scores):

    results = [{'id': img_id, "extracted_predictions": pred_bbox.cpu().tolist()} for img_id, pred_bbox in zip(ids, pred_bboxes)]
    gt_data = [json.loads(line) for line in open(ds_collections['humanref']['ann_path'], "r", encoding='utf-8')]

    # Process prediction files
    all_metrics = {}
    all_metrics["Model"] = evaluate_dataset(gt_data, results)

    # Print results with all models in same tables and optionally dump to file
    print_comparative_metrics(all_metrics, gt_data, None)
    

def eval_grefcoco(ids, datasets, pred_bboxes, pred_labels, pred_scores):

    results_val = [{'img_id': img_id, "bboxes": pred_bbox.cpu().numpy(), "scores": pred_score.cpu().numpy()} for img_id, pred_bbox, pred_score, sub_dataset in zip(ids, pred_bboxes, pred_scores, datasets) if sub_dataset == 'val']

    results_testA = [{'img_id': img_id, "bboxes": pred_bbox.cpu().numpy(), "scores": pred_score.cpu().numpy()} for img_id, pred_bbox, pred_score, sub_dataset in zip(ids, pred_bboxes, pred_scores, datasets) if sub_dataset == 'testA']

    results_testB = [{'img_id': img_id, "bboxes": pred_bbox.cpu().numpy(), "scores": pred_score.cpu().numpy()} for img_id, pred_bbox, pred_score, sub_dataset in zip(ids, pred_bboxes, pred_scores, datasets) if sub_dataset == 'testB']
    
    from grefcoco_metric import gRefCOCOMetric
    evaluator_val = gRefCOCOMetric(ds_collections['grefcoco']['ann_path'][0])
    results = evaluator_val.compute_metrics(results_val)
    print("grefcoco_val")
    print(results)

    evaluator_testA = gRefCOCOMetric(ds_collections['grefcoco']['ann_path'][1])
    results = evaluator_testA.compute_metrics(results_testA)
    print("grefcoco_testA")
    print(results)

    evaluator_testB = gRefCOCOMetric(ds_collections['grefcoco']['ann_path'][2])
    results = evaluator_testB.compute_metrics(results_testB)
    print("grefcoco_testB")
    print(results)
    


def eval_d3(ids, datasets, pred_bboxes, pred_labels, pred_scores):

    results_val = [{'img_id': img_id, "bboxes": pred_bbox.cpu().numpy(), "scores": pred_score.cpu(), 'labels': pred_label.cpu().numpy()} for img_id, pred_bbox, pred_score, pred_label, sub_dataset in zip(ids, pred_bboxes, pred_scores, pred_labels, datasets) if sub_dataset == 'FULL']

    results_testA = [{'img_id': img_id, "bboxes": pred_bbox.cpu(), "scores": pred_score.cpu(), 'labels': pred_label.cpu().numpy()} for img_id, pred_bbox, pred_score, pred_label, sub_dataset in zip(ids, pred_bboxes, pred_scores, pred_labels, datasets) if sub_dataset == 'PRES']

    results_testB = [{'img_id': img_id, "bboxes": pred_bbox.cpu(), "scores": pred_score.cpu(), 'labels': pred_label.cpu().numpy()} for img_id, pred_bbox, pred_score, pred_label, sub_dataset in zip(ids, pred_bboxes, pred_scores, pred_labels, datasets) if sub_dataset == 'ABS']
    
    from dod_metric import DODCocoMetric
    evaluator_val = DODCocoMetric(ds_collections['d3']['ann_path'][0])
    results = evaluator_val.compute_metrics(results_val)
    print("d3_FULL")
    print(results)

    evaluator_testA = DODCocoMetric(ds_collections['d3']['ann_path'][1])
    results = evaluator_testA.compute_metrics(results_testA)
    print("d3_PRES")
    print(results)

    evaluator_testB = DODCocoMetric(ds_collections['d3']['ann_path'][2])
    results = evaluator_testB.compute_metrics(results_testB)
    print("d3_ABS")
    print(results)
    



def plot_bounding_boxes(im, bounding_boxes, labels):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(bounding_boxes):
        # Select a color from the list
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box[1])
        abs_x1 = int(bounding_box[0])
        abs_y2 = int(bounding_box[3])
        abs_x2 = int(bounding_box[2])

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
        )

        # Draw the text
        draw.text((abs_x1 + 8, abs_y1 + 6), labels[i], fill=color)

    # Display the image
    return img
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_select', type=int, default=300)
    parser.add_argument('--nms', action='store_true')
    parser.add_argument('--score_thre', type=float, default=-1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--single_box', action='store_true')
    parser.add_argument('--recall', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    # results = torch.load('pred1.pth', 'cpu')
    # eval_d3(results['merged_ids'], results['merged_image_datasets'], results['merged_pred_bboxes'], results['merged_pred_labels'], results['merged_pred_scores'])
    # assert False
    if args.recall:
        fast_eval_recall(args.dataset)

    else:
        from datetime import timedelta
        timeout = timedelta(seconds=7200)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            rank=int(os.getenv('RANK', '0')),
            timeout=timeout,
        )
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

        from models.vision_process import process_vision_info
        from transformers import AutoProcessor

        # Model initialization
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        from models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration
        model = Qwen3VLGroundingForConditionalGeneration.from_pretrained(args.checkpoint, **model_kwargs)


        processor = AutoProcessor.from_pretrained(args.checkpoint)
        object_token_index = processor.tokenizer.convert_tokens_to_ids("<object>")
        model.model.object_token_id = object_token_index

        model = model.cuda().eval()


        random.seed(args.seed)
        dataset = GroundingDataset(args.dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        image_ids = []
        image_datasets = []
        all_pred_bboxes = []
        all_pred_labels = []
        all_pred_scores = []
        visualize_idx = 0
        for inputs in tqdm(dataloader, disable=torch.distributed.get_rank() != 0):
            pred_scores = []
            pred_bboxes = []
            pred_labels = []
            image_ids.append(inputs[0]['id'])
            image_datasets.append(inputs[0]['dataset'])
            image = inputs[0]['image']
            ori_shape = [image.size]
            proposals = copy.deepcopy(inputs[0]['proposals'])
            # random.shuffle(proposals)
            proposals = [torch.tensor(proposals).cuda().to(model.dtype)]
            if 'propsoals_score' in inputs[0]:
                propsoals_score = torch.tensor(inputs[0]['propsoals_score'])

            for i, prompt in enumerate(inputs[0]['query']):
                messages = copy.deepcopy(prompt)
                messages[0]['content'] = [{"type": "image", "image": copy.deepcopy(image)}] + messages[0]['content']
                image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

                texts = [processor.apply_chat_template(messages, tokenize=False)]
                model_inputs = processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=True,
                    do_resize=False,
                )
                model_inputs = model_inputs.to(model.device)
                
                with torch.inference_mode():
                    pred = model(
                        **model_inputs,
                        bboxes=copy.deepcopy(proposals),
                        ori_shapes=ori_shape,
                        bboxes_id=object_token_index,
                        image_inputs=image_inputs,
                    )
                
                proposal_positions = model_inputs['input_ids'] == object_token_index
                logits = pred.logits.sigmoid()[proposal_positions].view(-1)
                # if 'propsoals_score' in inputs[0]:
                #     logits *= propsoals_score
                pred_scores.append(logits.float())
                pred_labels.append(torch.full_like(logits, i, dtype=torch.int))
                pred_bboxes.append(proposals[0].clone().float())
            
            if not args.single_box:
                pred_bboxes = torch.cat(pred_bboxes, dim=0)
                pred_labels = torch.cat(pred_labels, dim=0)
                pred_scores = torch.cat(pred_scores, dim=0)
                if len(pred_bboxes) > 1000:
                    topk_values, topk_indexes = torch.topk(
                        pred_scores.view(-1), 1000, dim=0)
                    pred_scores = topk_values
                    pred_bboxes = pred_bboxes[topk_indexes]
                    pred_labels = pred_labels[topk_indexes]
                if args.nms:
                    selected_indices = torchvision.ops.batched_nms(pred_bboxes, pred_scores, pred_labels, iou_threshold=0.7)
                    pred_bboxes = pred_bboxes[selected_indices]
                    pred_labels = pred_labels[selected_indices]
                    pred_scores = pred_scores[selected_indices]
                if args.score_thre > 0:
                    mask = pred_scores > args.score_thre
                    pred_bboxes = pred_bboxes[mask]
                    pred_labels = pred_labels[mask]
                    pred_scores = pred_scores[mask]
                else:
                    topk = min(args.num_select, len(pred_scores))
                    topk_values, topk_indexes = torch.topk(
                        pred_scores.view(-1), topk, dim=0)
                    pred_scores = topk_values
                    pred_bboxes = pred_bboxes[topk_indexes]
                    pred_labels = pred_labels[topk_indexes]
            
            else:
                pred_scores = torch.stack(pred_scores, dim=1)
                pred_bboxes = pred_bboxes[0]
                pred_scores, pred_labels = torch.max(pred_scores, dim=1)
                if args.nms:
                    selected_indices = torchvision.ops.batched_nms(pred_bboxes, pred_scores, pred_labels, iou_threshold=0.7)
                    pred_bboxes = pred_bboxes[selected_indices]
                    pred_labels = pred_labels[selected_indices]
                    pred_scores = pred_scores[selected_indices]

            if args.dataset == 'd3':
                gt_labeles = torch.tensor(inputs[0]['gt_labels'])
                pred_labels = gt_labeles[pred_labels.cpu()]
            all_pred_bboxes.append(pred_bboxes)
            all_pred_labels.append(pred_labels)
            all_pred_scores.append(pred_scores)
            
            if args.visualize:
                if visualize_idx < 10:
                    if inputs[0]['dataset'] == 'coco':
                        gt_image = plot_bounding_boxes(copy.deepcopy(inputs[0]['image']), inputs[0]['gt_bboxes'], inputs[0]['gt_labels'])
                        gt_image.save("%d_gt.png" % (visualize_idx))  # 你可以自定义保存路径和文件名
                        # mask = pred_scores > 0.1
                        mask = torch.topk(pred_scores, 100)[1]
                        visualize_bboxes = pred_bboxes[mask].cpu().tolist()
                        visualize_labels = pred_labels[mask].cpu().tolist()
                        visualize_labels = [str(label) for label in visualize_labels]
                        print(pred_scores[mask])
                        pred_image = plot_bounding_boxes(copy.deepcopy(inputs[0]['image']), visualize_bboxes, visualize_labels)
                        pred_image.save("%d_pred.png" % (visualize_idx))  # 你可以自定义保存路径和文件名
                        proposal_image = plot_bounding_boxes(copy.deepcopy(inputs[0]['image']), proposals[0].cpu().tolist(), ["0" for _ in range(len(proposals[0]))])
                        proposal_image.save("%d_proposal.png" % (visualize_idx))  # 你可以自定义保存路径和文件名
                        visualize_idx += 1
                    elif inputs[0]['dataset'] in 'humanref':
                        gt_image = plot_bounding_boxes(copy.deepcopy(inputs[0]['image']), inputs[0]['gt_bboxes'], inputs[0]['gt_labels'])
                        gt_image.save("%d_gt.png" % (visualize_idx))  # 你可以自定义保存路径和文件名
                        visualize_bboxes = all_pred_bboxes[-1].cpu().tolist()
                        visualize_labels = all_pred_labels[-1].cpu().tolist()
                        visualize_labels = [str(label) for label in visualize_labels]
                        print(pred_scores)
                        pred_image = plot_bounding_boxes(copy.deepcopy(inputs[0]['image']), visualize_bboxes, visualize_labels)
                        pred_image.save("%d_pred.png" % (visualize_idx))  # 你可以自定义保存路径和文件名
                        visualize_idx += 1
                    else: # refcoco
                        gt_image = plot_bounding_boxes(copy.deepcopy(inputs[0]['image']), inputs[0]['gt_bboxes'], inputs[0]['gt_labels'])
                        gt_image.save("%d_gt.png" % (visualize_idx))  # 你可以自定义保存路径和文件名
                        visualize_bboxes = all_pred_bboxes[-1][:1].cpu().tolist()
                        visualize_labels = all_pred_labels[-1][:1].cpu().tolist()
                        visualize_labels = [str(label) for label in visualize_labels]
                        print(pred_scores[:1])
                        pred_image = plot_bounding_boxes(copy.deepcopy(inputs[0]['image']), visualize_bboxes, visualize_labels)
                        pred_image.save("%d_pred.png" % (visualize_idx))  # 你可以自定义保存路径和文件名
                        visualize_idx += 1

                

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_ids = [None for _ in range(world_size)]
        merged_image_datasets = [None for _ in range(world_size)]
        merged_pred_bboxes = [None for _ in range(world_size)]
        merged_pred_labels = [None for _ in range(world_size)]
        merged_pred_scores = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_ids, image_ids)
        torch.distributed.all_gather_object(merged_image_datasets, image_datasets)
        torch.distributed.all_gather_object(merged_pred_bboxes, all_pred_bboxes)
        torch.distributed.all_gather_object(merged_pred_labels, all_pred_labels)
        torch.distributed.all_gather_object(merged_pred_scores, all_pred_scores)

        merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
        merged_image_datasets = [_ for _ in itertools.chain.from_iterable(merged_image_datasets)]
        merged_pred_bboxes = [_ for _ in itertools.chain.from_iterable(merged_pred_bboxes)]
        merged_pred_labels = [_ for _ in itertools.chain.from_iterable(merged_pred_labels)]
        merged_pred_scores = [_ for _ in itertools.chain.from_iterable(merged_pred_scores)]

        if torch.distributed.get_rank() == 0:
            print(f"Evaluating {args.dataset} ...")

            if args.dataset == 'coco':
                eval_coco(merged_ids, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
            elif args.dataset == 'refcoco':
                eval_refcoco(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
            elif args.dataset == 'humanref':
                eval_humanref(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
            elif args.dataset == 'odinw35':
                eval_odinw35(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
            elif args.dataset == 'odinw13':
                eval_odinw13(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
            elif args.dataset == 'grefcoco':
                eval_grefcoco(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
            elif args.dataset == 'd3':
                eval_d3(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
            
        torch.distributed.barrier()


