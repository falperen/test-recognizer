import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import models

import ResumableTimer as rt

MODEL_PATH = 'trained_model/tr_model.pth'
FOLDER_PATH = 'AllFotos/'


def arg_parse():
    parser = argparse.ArgumentParser(description='Image recognizer for bins')
    parser.add_argument('-i', '--image', help='path to an image', type=str)
    return parser.parse_args()


def pick_random_image():
    _randFolder = random.choice(os.listdir(FOLDER_PATH)) + "/"
    _randImg = _randFolder + random.choice(os.listdir(FOLDER_PATH + _randFolder))
    print('Choosing a {}'.format(FOLDER_PATH + _randImg))
    return FOLDER_PATH + _randImg


def recognize(_image, _model):
    x = TF.to_tensor(Image.open(_image))
    x.unsqueeze_(0)
    x = x.to(DEVICE)
    output = _model(x)
    _, pred = torch.max(output, 1)
    return class_names[pred]


def start_recognizer(image, model):
    elapsed_time.start()
    result = recognize(image, model)

    print('It\'s a piece of {}'.format(result))
    elapsed_time.pause()
    total_time = elapsed_time.get_actual_time()
    print('Test completed in {:.3f}s'.format(total_time))


def main(args):
    print('Running smart bin brain...')
    if args.image is None:
        args.image = pick_random_image()

    model_ts = models.resnet18(pretrained=True)
    num_ftrs = model_ts.fc.in_features
    model_ts.fc = nn.Linear(num_ftrs, 20)
    model_ts.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model_ts.eval()
    model_ts = model_ts.to(DEVICE)

    start_recognizer(args.image, model_ts)


if __name__ == '__main__':
    elapsed_time = rt.ResumableTimer()

    DEVICE = torch.device("cpu")

    with open('class_names.txt', 'r') as f:
        class_names = [line.rstrip('\n') for line in f]

    args = arg_parse()

    main(args)
