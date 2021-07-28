import numpy as np
import os
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch.nn.functional as F


def sortby_seqid(list):
    def second_sorter(x):
        sid = int(x.split("_", 1)[1].split(".", 1)[0])
        return sid

    ls = [l.sort(key=second_sorter)for l in list]
    return ls


def file_list_generator(path):
    masterfilelist = []
    allmods = os.listdir(path)
    modlist = [foldername for foldername in allmods]

    for i in range(len(modlist)):
        modpath = os.path.join(path, modlist[i])
        allfiles = os.listdir(modpath)
        filelist = [filename for filename in allfiles]
        masterfilelist.append(filelist)
    return masterfilelist


def find_class(path, mod):
    filelist = file_list_generator(path)
    sortby_seqid(filelist)
    allmods = os.listdir(path)
    modlist = np.array([foldername for foldername in allmods])
    mod_index = np.where(modlist == mod)[0].item()
    mod_file = filelist[mod_index]
    class_list = [int(file[:1]) for file in mod_file]

    return class_list


def random_oversampler_index(list, mode, shuffle=True, seed=1, distribution_list=None, train_split=0.8, val_split=0.1):
    tar = np.asarray(list)
    print(len(tar))
    H_id = np.where(tar == 0)[0]
    L_id = np.where(tar == 1)[0]
    N_id = np.where(tar == 2)[0]

    print("H", H_id, len(H_id))
    print("L", L_id, len(L_id))
    print("N", N_id, len(N_id))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(H_id), np.random.shuffle(L_id), np.random.shuffle(N_id)

    if mode == 'train':
        H_id = H_id[:int(np.floor(len(H_id) * train_split))]
        L_id = L_id[:int(np.floor(len(L_id) * train_split))]
        N_id = N_id[:int(np.floor(len(N_id) * train_split))]

        print("1", len(H_id))
        print("2", len(L_id))
        print("3", len(N_id))


        if distribution_list is None:
            max_samples = max(len(H_id), len(L_id), len(N_id))

            # TO DO: figure out a way to put True or False Automatically based on Max samples
            high = np.random.choice(H_id, max_samples, replace=False)
            low = np.random.choice(L_id, max_samples, replace=True)
            no = np.random.choice(N_id, max_samples, replace=True)


            print("4", len(high))
            print("5", len(low))
            print("6", len(no), no)


        else:
            number_samples = len(H_id) + len(L_id) + len(N_id)

            high = np.random.choice(H_id, int(np.floor(number_samples * distribution_list[0])), replace=False)
            low = np.random.choice(L_id, int(np.floor(number_samples * distribution_list[1])), replace=True)
            no = np.random.choice(N_id, int(np.floor(number_samples * distribution_list[2])), replace=False)

        ids = np.hstack([high, low, no])
        np.random.seed(seed)
        np.random.shuffle(ids)
        print("Training Indices Length: ", len(ids))
        return iter(ids)

    if mode == 'val':
        H_id = H_id[int(np.floor(len(H_id) * train_split)): int(np.floor(len(H_id) * val_split))]
        L_id = L_id[int(np.floor(len(L_id) * train_split)): int(np.floor(len(L_id) * val_split))]
        N_id = N_id[int(np.floor(len(N_id) * train_split)): int(np.floor(len(N_id) * val_split))]

        '''
        print("8", len(H_id))
        print("9", len(L_id))
        print("10", len(N_id))
        '''

        ids = np.hstack([H_id, L_id, N_id])
        np.random.seed(seed)
        np.random.shuffle(ids)
        print("Validation Indices Length:", len(ids))
        return iter(ids)

    if mode == 'test':
        H_id = H_id[int(np.floor(len(H_id) * val_split)):]
        L_id = L_id[int(np.floor(len(L_id) * val_split)):]
        N_id = N_id[int(np.floor(len(N_id) * val_split)):]

        ids = np.hstack([H_id, L_id, N_id])
        np.random.seed(seed)
        np.random.shuffle(ids)
        print("Testing Indices Length:", len(ids))
        return iter(ids)


def lr_decay(global_step, init_learning_rate, min_learning_rate, decay_rate):
    lr = ((init_learning_rate - min_learning_rate) * pow(decay_rate, global_step) + min_learning_rate)
    return lr


def get_accuracy(loader, model, num_classes, loss_fn='CrossEntropy'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        correct_per_class = list(0 for i in range(num_classes))
        total_per_class = list(0 for i in range(num_classes))
        total_per_pred = list(0 for i in range(num_classes))
        recall_list = []
        precision_list = []
        f1_list = []

        for images, labels in loader:

            labels = torch.tensor(labels)

            images, labels = images.to(device), labels.to(device)
            scores, prob = model(images)  #RNN model training
            #scores = model(images)

            if loss_fn == 'CrossEntropy':
                _, predictions = torch.max(scores, 1)

            else:
                predictions = (scores > 0.5).long()
            #print('Pred', predictions.data, "Label", labels.data)
            c = (predictions == labels)

            for i in range(len(labels)):
                label = labels[i]
                pred = predictions[i]
                correct_per_class[label] += c.squeeze()[i].item()
                total_per_class[label] += 1
                total_per_pred[pred] += 1

        correct = 0
        total = 0
        for i in range(num_classes):
            if total_per_pred[i] > 0:
                re = float(correct_per_class[i] / total_per_class[i])  # Recall per class
                pr = float(correct_per_class[i] / total_per_pred[i])  # Precision per class
                recall_list.append((i, re))
                precision_list.append((i, pr))
                if pr + re <= 0:
                    pr = 1
                f1 = 2 * ((pr * re) / (pr + re))  # F1 per class
                f1_list.append((i, f1))
                correct += correct_per_class[i]
                total += total_per_class[i]
            else:
                print("No predictions for class index of", i)
                f1 = 0.
                f1_list.append((i, f1))

        overall_acc = float(correct / total)
        print(f"Target Distribution: {total_per_class}, Predicted Distribution: {total_per_pred}")
        weighted_f1 = 0
        for i in range(num_classes):
            weighted_f1 += (total_per_class[i] / total) * f1_list[i][1]

    return overall_acc, precision_list, recall_list, f1_list, weighted_f1


def RNN_collate(batch):
    batch = list(filter(lambda img: img is not None, batch))
    seq_length_list = []

    for i in range(len(batch)):
        seq_len = batch[i][0].shape[0]
        seq_length_list.append(seq_len)

    if len(batch[0]) == 2:
        feature, target = zip(*batch)
        feature = pad_sequence(feature, batch_first=True)
        seq_tensor = torch.tensor(seq_length_list)
        feature = pack_padded_sequence(feature, seq_tensor, batch_first=True, enforce_sorted=False)

        return feature, target

    elif len(batch[0]) == 3:
        feature, target, idx = zip(*batch)
        feature = pad_sequence(feature, batch_first=True)
        seq_tensor = torch.tensor(seq_length_list)
        feature = pack_padded_sequence(feature, seq_tensor, batch_first=True, enforce_sorted=False)

        return feature, target, idx


def get_length():
    cwd = os.getcwd()
    neighbour_pt = os.path.join(cwd, 'neighbourhood', 'neighbourhood.pt')

    with open(neighbour_pt, 'rb') as handle:
        my_dict = torch.load(handle)
        seq = my_dict['seq']
        length = len(seq)

        return length


def get_neighbour(input_seqid):
    cwd = os.getcwd()
    neighbour_pt = os.path.join(cwd, 'neighbourhood', 'neighbourhood.pt')

    with open(neighbour_pt, 'rb') as handle:
        my_dict = torch.load(handle)
        nb_seq = (my_dict['seq'][input_seqid]).tolist()
        nb_cos = my_dict['score'][input_seqid]
        nb_class_index = my_dict['class_index'][input_seqid]

        return nb_seq, nb_class_index, nb_cos


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=None)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=None)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
