# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:27:29 2020

@author: Hp
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import ResumableTimer as tm

from PIL import Image
import torchvision.transforms.functional as TF


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = 'AllFotos'
image_datasets = datasets.ImageFolder(data_dir, data_transforms['test'])

dataset_size = len(image_datasets)

dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4,
                                             shuffle=True, num_workers=0)

#class_names = image_datasets.classes

with open('class_names.txt', 'r') as f:
    class_names = [line.rstrip('\n') for line in f]
     
final_class_names = [i.split("_")[0] for i in class_names]
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
test_time = tm.ResumableTimer()

y_true_test = []
y_pred_test = []

def main():    
        
    np.set_printoptions(precision=2)
    MODEL_PATH = 'trained_model/tr_model.pth'
    criterion = nn.CrossEntropyLoss()
    model_ts = models.resnet18(pretrained=True)
    num_ftrs = model_ts.fc.in_features
    model_ts.fc = nn.Linear(num_ftrs, 20)
    model_ts.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model_ts.eval()
    model_ts = model_ts.to(device)
    
#    test_model(model_ts, criterion)
#    test_single_batch_image(model_ts, criterion)
    test_single_image(model_ts, criterion)
    

def test_single_image(model, criterion):

    foto_dir = '1552300023.21.jpg'
    image = Image.open(foto_dir)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    x = x.to(device)
    output = model(x)
    _, pred = torch.max(output, 1)
    print(class_names[pred])

def test_single_batch_image(model, criterion):

    data_dir2 = 'jar_photos'
    image_dataset = datasets.ImageFolder(data_dir2, data_transforms['test'])
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1,
                                             shuffle=True, num_workers=0)
    i = 1
    for inputs1, labels1 in dataloader:
        inputs1 = inputs1.to(device)
        output1 = model(inputs1)
        _, preds1 = torch.max(output1, 1)
        print(i , class_names[preds1])
        i += 1
        
def test_model(model, criterion):
    test_time.start()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
                
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        y_true_test.extend(labels.data.squeeze().tolist())
        y_pred_test.extend(preds.squeeze().tolist())

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'test', epoch_loss, epoch_acc))
    test_time.pause()
    time_elapsed_test = test_time.get_actual_time()
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed_test // 60, time_elapsed_test % 60))



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    else:
        pass

    #print(cm)

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    main()