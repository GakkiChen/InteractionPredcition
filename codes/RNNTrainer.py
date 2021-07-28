from utils import *
from data_loading import ProSampler
from ModelFinal import ModalRnn
from pathlib import Path
from CustomDatasets import FeatureDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import sys
import os
import torch
import torch.nn as nn
import argparse
'''
# cmd interface
parser = argparse.ArgumentParser(description='Get Features of image sequences')
parser.add_argument('-mod', '--modality', type=str, metavar='', help='Path to pretrained model', required=True)
parser.add_argument('-b', '--batch_size', type=int, metavar='', help='Batch size', required=True)
parser.add_argument('-lr', '--learning_rate', type=float, metavar='', help='Learning rate', required=True)
parser.add_argument('-dr', '--decay', type=float, metavar='', help='Decay rate', required=True)
args = parser.parse_args()

mod = args.modality  # <-- IMPORTANT Change mod for different modality
'''
#model_file = os.path.join(os.getcwd(), "generated_models", "RNN_LSTM_Face", "Face_RNN.pth.tar")
model_file = ''  # <-- IMPORTANT Change Model file for past saved models


# Parameters  CHANGE HERE
mod = 'EmotionDeepemotion'
#torch.manual_seed(10)
csv_path = 'D:/Interacction/Interaction_Dataset/Completed_Annotation/processed_csv'
batch_size = 64
learning_rate = 3e-05
decay_rate = 0.9999

graph_path = './graphs/runs/' + mod + '_RNN'
output_file = mod + '_RNN_latest_25July.pth.tar'
path = os.path.join(os.getcwd(), 'extracted_features')

# Hyperparameters
num_classes = 2
#learning_rate = args.learning_rate
min_learning_rate = 0.0009
num_epochs = 100
#batch_size = args.batch_size
#decay_rate = args.decay
patience_limit = 60

if mod == 'Object':
    input_size = 1000
elif mod == 'Scene':
    input_size = 365
elif mod == 'Face':
    input_size = 2048
elif mod == 'Keypoint':
    input_size = 140
elif mod == 'EmotionFER':
    input_size = 512
elif mod == 'EmotionEmonet':
    input_size = 256
elif mod == 'EmotionResnet':
    input_size = 512
elif mod == 'EmotionDeepemotion':
    input_size = 50
elif mod == 'EngagementCNN':
    input_size = 512
else:
    input_size = 1000



# Define Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModalRnn(input_size)  # Change model here
model = model.to(device)
model_name = str(model._get_name() + "_" + mod)
model_save_path = os.path.join('./generated_models', model_name)
Path(model_save_path).mkdir(parents=True, exist_ok=True)
model_save_path = os.path.join(model_save_path, output_file)

# Load Dataset
dataset_train = FeatureDataset(mod=mod, csv_path=csv_path, train=True)
dataset_test = FeatureDataset(mod=mod, csv_path=csv_path, train=False)


train_loader = DataLoader(dataset_train, batch_size=batch_size,
                          sampler=ProSampler(dataset_train, mode='train', function='oversample', num_class=2,
                                             train_split=1.0, val_split=0.),
                          collate_fn=RNN_collate)
print("train loader initialized")
val_loader = DataLoader(dataset_test, batch_size=batch_size,
                        sampler=ProSampler(dataset_test, mode='test', function='original',
                                           num_class=2, train_split=1.0, val_split=0.),
                        collate_fn=RNN_collate)
print("val loader initialized")

# Learning Rate decay function inside utils
lr_init = lr_decay(0, learning_rate, min_learning_rate, decay_rate)

# Optimiser
optimiser = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.LambdaLR(optimiser, lambda step: lr_decay(step, learning_rate, min_learning_rate,
                                                                         decay_rate)/lr_init)
#class_weight = 1.0 / torch.tensor([100, 150], dtype=torch.float).to(device)

#criteria = nn.BCEWithLogitsLoss()
criteria = nn.CrossEntropyLoss()
print(mod)

if model_file != '':
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state'])
    optimiser.load_state_dict(checkpoint['optimiser_state'])
    epoch = checkpoint['epoch']
    max_accuracy = checkpoint['val_accuracy']
    print("Model File Loaded", "Epochs Trained: ", epoch, 'Last Accuracy:', max_accuracy)

else:
    print("NO MODEL FILE")
    max_accuracy = 0


# Training loop
patience_counter = 0  # counts how many times val acc was not >= max acc
#p1 = os.path.join(graph_path, f'Epoch {epoch}')
writer = SummaryWriter(graph_path)

for epoch in range(num_epochs):
    print("Epoch:", epoch)
    if epoch != 0:
        scheduler.step()
        print("Scheduler has stepped for epoch:", epoch, "Learning Rate:", scheduler.get_last_lr())

    pred_list = []
    total_correct = 0
    total_sample = 0
    correct_per_class = list(0 for i in range(num_classes))
    total_per_class = list(0 for i in range(num_classes))
    total_per_pred = list(0 for i in range(num_classes))
    recall_list = []
    precision_list = []
    f1_list = []
    loss_list = []

    start = time.time()
    for batch_index, (data, targets) in enumerate(train_loader):
        model.train()
        #model.zero_grad()  # commented this line! Check if this is causing performance drop!!!!!
        targets = torch.tensor(targets)
        data, targets = data.to(device), targets.to(device)

        # Forward Pass
        scores, prob = model(data)
        #print("prob", prob)
        if criteria.__class__ == nn.CrossEntropyLoss:
            loss = criteria(scores, targets)

        else:
            loss = criteria(scores, targets.type_as(scores))

        loss_list.append(float(loss.data))

        # Backward Propagation

        optimiser.zero_grad()  #<-- For Version >=1.7
        #for param in model.parameters():
            #param.grad = None

        loss.backward()
        optimiser.step()

        # Check Prediction
        #print(scores, scores.shape)

        if criteria.__class__ == nn.CrossEntropyLoss:
            _, predictions = torch.max(scores, 1)

        elif criteria.__class__ == nn.BCEWithLogitsLoss or criteria.__class__ == nn.BCELoss \
                or criteria.__class__ == FocalLoss:
            predictions = (scores > 0.5).long()

        #print(predictions)
        pred_list.append(predictions.data.tolist())
        c = (predictions == targets)
        num_correct = int(c.sum())
        total_correct += num_correct
        total_sample += len(targets)

        for i in range(len(targets)):
            label = targets[i]
            pred = predictions[i]
            correct_per_class[label] += c.squeeze()[i].item()
            total_per_class[label] += 1
            total_per_pred[pred] += 1

        if batch_index % 5 == 0:
            # Calculate mean loss of n number of batches
            mean_loss = 0.
            for l in loss_list:
                mean_loss += l

            mean_loss /= len(loss_list)
            loss_list.clear()
            writer.add_scalar("Training Loss", mean_loss, global_step=batch_index)
            #print(f'Batch: {batch_index} Loss:{mean_loss}')
        print(f"Target Distribution: {total_per_class}, Predicted Distribution: {total_per_pred}")

    # Accuracy for Training
    mean_acc = float(total_correct / total_sample)  # Overall Accuracy

    for i in range(num_classes):
        re = float(correct_per_class[i] / total_per_class[i])  # Recall per class
        if total_per_pred[i] > 0:
            pr = float(correct_per_class[i] / total_per_pred[i])  # Precision per class
            recall_list.append((i, re))
            precision_list.append((i, pr))
            if pr + re <= 0:
                pr = 1
            f1 = 2 * ((pr * re) / (pr + re))  # F1 Score per class
            f1_list.append((i, f1))
        else:
            print("No predictions for class", i)

    writer.add_scalar("Training Accuracy", mean_acc, global_step=epoch)  #
    end = time.time()
    training_time = end - start

    print(" --------------------------TRAINING-------------------------------------- ")
    print(f'Epoch: {epoch} | Training Metrics |')
    print(f'Avg Accuracy: {mean_acc}, Time: {training_time}')
    print(f'Precision: {precision_list}\nRecall: {recall_list}\nF1: {f1_list}')
    print(" ------------------------------------------------------------------------ ")

    val_acc = 0
    if mean_acc > 0.5:  # training acc > 0.7

    # Check Validation Accuracy
        start1 = time.time()
        val_acc, val_pr, val_re, val_f1, w_f1 = get_accuracy(loader=val_loader, model=model, num_classes=num_classes, loss_fn='CrossEntropy')
        writer.add_scalar("Validation Weighted F1", w_f1, global_step=epoch)
        end1 = time.time()
        validation_time = end1 - start1

        print(" --------------------------VALIDATION-------------------------------------- ")
        print(f'Epoch: {epoch} | Validation Metrics |')
        print(f'Avg Accuracy: {val_acc}, Time: {validation_time}')
        print(f'Precision: {val_pr}\nRecall: {val_re}\nF1: {val_f1}\nWeighted F1: {w_f1}')
        print(" -------------------------------------------------------------------------- ")

    # Save the Model
    if w_f1 >= max_accuracy:

        max_accuracy = w_f1

        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimiser_state': optimiser.state_dict(),
            'loss': mean_loss,
            'val_accuracy': val_acc,
            'train_accuracy': mean_acc
        }

        torch.save(checkpoint, model_save_path)
        patience_counter = 0

    else:
        patience_counter += 1

        if patience_counter == patience_limit + 1:
            break

writer.flush()
writer.close()
