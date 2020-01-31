# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:58:52 2020

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

class_names = image_datasets.classes
final_class_names = [i.split("_")[0] for i in class_names]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
test_time = tm.ResumableTimer()

y_true_test = []
y_pred_test = []

def main():    
        
    MODEL_PATH = 'trained_model/tr_model.pth'
    criterion = nn.CrossEntropyLoss()
    model_ts = models.resnet18(pretrained=True)
    num_ftrs = model_ts.fc.in_features
    model_ts.fc = nn.Linear(num_ftrs, 20)
    model_ts.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model_ts.eval()
    model_ts = model_ts.to(device)
    test_model(model_ts, criterion)
    #visualize_model(model_ft)
    #yt_train = np.reshape(np.array(y_true_train),(num_epochs,-1))
    #yp_train = np.reshape(np.array(y_pred_train),(num_epochs,-1))
    
    yt_test = np.reshape(np.array(y_true_test),(1,-1))
    yp_test = np.reshape(np.array(y_pred_test),(1,-1))
    
    
    np.set_printoptions(precision=2)
    
    
#    ids = list(range(20))
    final_class_names2 = final_class_names.copy()
    l=0
    while l<len(final_class_names):
        try:
            final_class_names2[final_class_names2.index('aluminium')] = 'aluminium/plastic'
        except:
            pass
        try:
            final_class_names2[final_class_names2.index('plastic')] = 'aluminium/plastic'
        except:
            pass
        l+=1
#    final_classes = list(set(final_class_names))
    final_classes2 = list(set(final_class_names2))
    
    # Plot non-normalized confusion matrix
    p7 = plot_confusion_matrix(yt_test[-1], yp_test[-1], classes=class_names,
                          title='Confusion matrix, without normalization')
    p7.savefig('testAll_resnet18_ep25.png')
    # Plot normalized confusion matrix
    p8 = plot_confusion_matrix(yt_test[-1], yp_test[-1], classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    p8.savefig('normalized_testAll_resnet18_ep25.png')
    
    final_class_names2 = final_class_names.copy()
    l=0
    while l<len(final_class_names):
        try:
            final_class_names2[final_class_names2.index('aluminium')] = 'aluminium/plastic'
        except:
            pass
        try:
            final_class_names2[final_class_names2.index('plastic')] = 'aluminium/plastic'
        except:
            pass
        l+=1
    
    yp_test_final2 = []
    yt_test_final2 = []
    for i in yp_test[-1]:
        c = final_class_names[i]
        if c!='aluminium' and c!='plastic':
            yp_test_final2.append(final_classes2.index(c))
        else:
            yp_test_final2.append(final_classes2.index('aluminium/plastic'))
        
    for i in yt_test[-1]:
        c = final_class_names[i]
        if c!='aluminium' and c!='plastic':
            yt_test_final2.append(final_classes2.index(c))
        else:
            yt_test_final2.append(final_classes2.index('aluminium/plastic'))
        
    p11 = plot_confusion_matrix(yt_test_final2, yp_test_final2, classes=final_classes2,
                          title='Confusion matrix, without normalization')
    p11.savefig('final_testAll_resnet18_ep25.png')
    # Plot normalized confusion matrix
    p12 = plot_confusion_matrix(yt_test_final2, yp_test_final2, classes=final_classes2, normalize=True,
                          title='Normalized confusion matrix')
    p12.savefig('normalized_final_testAll_resnet18_ep25.png')

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