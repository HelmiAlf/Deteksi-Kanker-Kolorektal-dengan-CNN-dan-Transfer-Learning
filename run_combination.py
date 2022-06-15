from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.optim import SGD, RMSprop, NAdam, Adam, Adadelta
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from collections import OrderedDict
import time
import os
import json
import copy
import gc

cudnn.benchmark = True
plt.ion()   # interactive mode

print("Using torch", torch.__version__)
torch.manual_seed(42) # Setting the seed

data_transforms = {
    'Training': transforms.Compose([
        transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.4),
        transforms.ToTensor()
    ]),
    'Validation': transforms.Compose([
        transforms.ToTensor()
    ]),
    'Testing': transforms.Compose([
        transforms.ToTensor()
    ]),
}
batch_size = 32


# data_dir = '/content/gdrive/MyDrive/Tugas Akhir/Pytorch Notebook/Warwick 300_50'
# combination_dir = '/content/gdrive/MyDrive/Tugas Akhir/Pytorch Notebook/combination_demo.txt'
# root_path= '/content/gdrive/MyDrive/Tugas Akhir/Pytorch Notebook/'

data_dir = 'Warwick 300_50'
combination_dir = 'combination2.txt'
root_path = ''

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['Training', 'Validation', 'Testing']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['Training', 'Validation', 'Testing']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['Training', 'Validation', 'Testing']}
class_names = image_datasets['Training'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('USING', device)

def freeze_model(model, freeze_rate):
    parameter_count = sum([1 for _ in model.parameters()])
    stop_freeze = int(parameter_count*freeze_rate)
    
    i = 0
    for param in model.parameters():
      i += 1
      if i >= stop_freeze:
          break
      param.requires_grad = False
    return model

def mobilenet_v2(freeze_rate, classifier):
        model = models.mobilenet_v2(pretrained=True)
        model = freeze_model(model, freeze_rate)
        model.classifier = classifier
        return model

def resnet50(freeze_rate, classifier):
        model = models.resnet50(pretrained=True)
        model = freeze_model(model, freeze_rate)
        model.fc = classifier
        return model

def resnet18(freeze_rate, classifier):
        model = models.resnet18(pretrained=True)
        model = freeze_model(model, freeze_rate)
        model.fc = classifier
        return model

def densenet161(freeze_rate, classifier):
        model = models.densenet161(pretrained=True)
        model = freeze_model(model, freeze_rate)
        model.classifier = classifier
        return model

def densenet121(freeze_rate, classifier):
        model = models.densenet121(pretrained=True)
        model = freeze_model(model, freeze_rate)
        model.classifier = classifier
        return model

def inception_v3(freeze_rate, classifier):
        model = models.inception_v3(pretrained=True)
        model.aux_logits=False
        model = freeze_model(model, freeze_rate)
        model.fc = classifier
        return model

def resnext50_32x4d(freeze_rate, classifier):
        model = models.resnext50_32x4d(pretrained=True)
        model = freeze_model(model, freeze_rate)
        model.fc = classifier
        return model

def c_model(base, freeze_rate, hidden_layer_count):
    
    classifier = create_classifier(base, hidden_layer_count)
    model = globals()[base](freeze_rate, classifier)
        
    return model

# todo 
def create_classifier(base, hidden_layer_count):
    num_ftrs = {
        'mobilenet_v2' : 1280,
        'resnet50': 2048,
        'resnet18': 512,
        'densenet161': 2208,
        'densenet121': 1024,
        'inception_v3': 2048,
        'resnext50_32x4d': 2048
    }

    neuron_counts = [num_ftrs[base]]
    curr = num_ftrs[base]

    for i in range(hidden_layer_count):
        curr = int(curr/2)
        neuron_counts.append(curr)
    neuron_counts.append(2)

    layers = []
    for i in range(1, len(neuron_counts)):
        layers.append( ('Lin'+str(i), nn.Linear(neuron_counts[i-1], neuron_counts[i])) )
        if i < len(neuron_counts)-1:
          layers.append( ('Rel'+str(i), nn.ReLU(inplace=True)) )

    classifier = nn.Sequential(OrderedDict(layers))

    return classifier


def c_optimizer(model, opt, lr):
    if opt == 'SGD':
        optimizer = globals()[opt](model.parameters(), lr=lr, momentum=0.8, nesterov=True)
    else:
        optimizer = globals()[opt](model.parameters(), lr=lr)
    
    return optimizer


def evaluate(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            if y_true[i] == 0:
                TN += 1
            else:
                FN += 1
        else:
            if y_true[i] == 0:
                FP += 1 
            else:
                TP += 1

    return np.array([TP, TN, FP, FN])


def calculate_metric(evals):
    TP = evals[0]
    TN = evals[1]
    FP = evals[2]
    FN = evals[3]

    accuracy = (TP + TN) / evals.sum()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return (TP, TN, FP, FN, accuracy, sensitivity, specificity)



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    training_time = []
    training_loss = []
    validation_loss = []

    early_stopper = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        since = None
        time_elapsed = None

        if early_stopper == 0:
          print("\nEARLY STOPPING")
          break

        # Each epoch has a training and validation phase
        print(early_stopper)
        for phase in ['Training', 'Validation']:
            if phase == 'Training':
                since = time.time()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_eval = np.array([0, 0, 0, 0])

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_eval += evaluate(labels.data, preds)

            epoch_loss = running_loss / dataset_sizes[phase]
            TP, TN, FP, FN, acc, sensi, speci = calculate_metric(running_eval)

            if phase == 'Training':
                training_loss.append(epoch_loss)

                time_elapsed = time.time() - since
                training_time.append(time_elapsed)
                print(f'Epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

            if phase == 'Validation':
                validation_loss.append(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {acc:.4f} Sensi: {sensi:.4f} Speci: {speci:.4f}')

            # deep copy the model
            if phase == 'Validation' and acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'Validation' and sensi > 0.95 and speci > 0.95 and early_stopper is None:
                early_stopper = 10
                print('Early Stopping in '+str(early_stopper)+' epoch')
                
            
            if phase == 'Validation' and early_stopper:
                early_stopper -= 1
      
        print()
        # if best_acc > 0.95:
        #   break


    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_time, {'Training Loss': training_loss, 'Validation Loss':validation_loss}

def test_model(model, criterion):
    model.eval()
    phase = 'Testing'

    running_loss = 0.0
    running_eval = np.array([0, 0, 0, 0])
    y_true = np.array([])
    scores = np.array([])

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            mistery, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            softmax = nn.Softmax(dim=1)(outputs)
            probs = softmax[:, 1]

            y_true = np.append(y_true, labels.cpu().numpy())
            scores = np.append(scores, probs.cpu().numpy())

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_eval += evaluate(labels.data, preds)

    epoch_loss = running_loss / dataset_sizes[phase]
    TP, TN, FP, FN, acc, sensi, speci = calculate_metric(running_eval)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    nfpr, ntpr, _ = roc_curve(y_true, [0 for _ in range(len(y_true))])

    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {acc:.4f} Sensi: {sensi:.4f} Speci: {speci:.4f}')

    return {
        'Loss': float(epoch_loss),
        'True Pos': int(TP),
        'True Neg': int(TN),
        'False Pos': int(FP),
        'False Neg': int(FN),
        'Accuracy':acc,
        'Sensitivity':sensi,
        'Specificity':speci,
        'FPR': fpr.tolist(),
        'TPR': tpr.tolist(),
        'threshold': thresholds.tolist(),
        'Nfpr': nfpr.tolist(),
        'Ntpr': ntpr.tolist()
        }


def create_search_list():
    f = open(combination_dir).read().splitlines()
    f.sort()
    result = []
    for i, query in enumerate(f):
        if "DONE" not in query:
            result.append((i, query))
    
    return result


def update_search_list(index):
    f = open(combination_dir).read().splitlines()
    f.sort()
    f[index] = f[index] +' DONE'
    new_file = '\n'.join(f)

    with open(combination_dir, 'w') as new:
        new.write(new_file)



def count_trainable_param(model):
    trainable_count = 0

    for param in model.parameters():
      if param.requires_grad:
        count = torch.numel(param)
        trainable_count += count

    return trainable_count 

def get_model_summary(model):
    texts = 'Trainable Params:'+str(count_trainable_param(model))+'\n'
    texts += str(model)
    return texts

def write_combination_summary(root_path, query, model, training_eval, training_time, testing_eval):
    q = query.split(' ')
    folder_name = q[-1]+' '+query

    path = root_path+'Models_Warwick_vol2/'
    os.makedirs(path, mode=511, exist_ok=True)
    
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(path+'Weights/'+folder_name+'.pt')
  
    model_summary = get_model_summary(model)
    with open(path+'Summary/'+folder_name+'.txt', 'w') as s:
      s.write(model_summary)
    
    with open(path+'History/'+folder_name+'.json', 'w') as history:
      json.dump(training_eval, history)

    with open(path+'Evaluation/'+folder_name+'.json', 'w') as eval:
      json.dump(testing_eval, eval)
    
    with open(path+'TrainingTime/'+folder_name+'.txt', 'w') as train_time:
      train_time.write(str(training_time))

    plt.clf()
    plt.plot(testing_eval['Nfpr'], testing_eval['Ntpr'], linestyle='--')
    plt.plot(testing_eval['FPR'], testing_eval['TPR'])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(query)
    plt.savefig(path+'ROC_Curve/'+folder_name+'.png')
    plt.show()
    

# base, freeze_rate, hidden_layer_count, dropout_rate, optimizer, learning_rate, l2_rate



def main():
    
    models_combination = create_search_list()

    for index, query in models_combination:
        q = query.split(' ')
        print(q)
        print()

        model_ft = c_model(q[0], float(q[1]), int(q[2]))
        optimizer_ft = c_optimizer(model_ft, q[3], float(q[4]))
        criterion = nn.CrossEntropyLoss()

        model_ft.to(device)

        model_ft, training_time, loss = train_model(model_ft, criterion, optimizer_ft, None,
                           num_epochs=30)
        
        test_eval = test_model(model_ft, criterion)

        write_combination_summary(root_path, query, model_ft, loss, training_time, test_eval)
        
        print(training_time)
        print(loss)
        print(test_eval)

        torch.cuda.empty_cache()

        del model_ft
        del training_time
        del loss
        del test_eval
        gc.collect()

        update_search_list(index)

        print()
        print('-'*5,'end','-'*5)
        print()

if __name__ == "__main__":
    main()