from torch.utils.data import DataLoader, Sampler
import numpy as np
import CustomDatasets
from torch.utils.data import dataloader
import pandas as pd
from random import shuffle

'''
oversampler class allows to do Train-Validation-Test split of data and oversample imbalanced classes
Has 3 modes: 'train', 'val', 'test'
Only oversamples on train mode
DONT USE THE oversampler class since it is deprecated! Use ProSampler instead (my custom implementation)
'''
class oversampler(Sampler):
    def __init__(self, dataset, mode, class_list=None, distribution_list=None, train_split=0.8, val_split=0.1, shuffle=True, seed=1):
        self.dataset = dataset
        self.distribution = distribution_list
        self.train_split = train_split
        self.shuffle = shuffle
        self.seed = seed
        self.mode = mode
        self.val_split = val_split
        self.class_list = class_list

        if self.mode == 'train':
            self.n = int(len(self.dataset) * self.train_split)
        if self.mode == 'val':
            self.n = int((len(self.dataset) * self.val_split))
        if self.mode == 'test':
            self.n = int((len(self.dataset) * (1 - self.train_split - self.val_split)))

        self.val_split = val_split + train_split

    def __iter__(self):
        if self.class_list is None:
            tar = np.asarray(self.dataset.targets)
        else:
            tar = np.asarray(self.class_list)
        H_id = np.where(tar == 0)[0]
        L_id = np.where(tar == 1)[0]
        N_id = np.where(tar == 2)[0]

        if self.shuffle is True:
            np.random.seed(self.seed)
            np.random.shuffle(H_id), np.random.shuffle(L_id), np.random.shuffle(N_id)

        if self.mode == 'train':
            H_id = H_id[:int(np.floor(len(H_id) * self.train_split))]
            L_id = L_id[:int(np.floor(len(L_id) * self.train_split))]
            N_id = N_id[:int(np.floor(len(N_id) * self.train_split))]

            if self.distribution is None:
                max_samples = max(len(H_id), len(L_id), len(N_id))

                def replace_gen(combine_class):
                    blist = []
                    for el in combine_class:
                        if len(el) == max_samples:
                            blist.append(False)
                        else:
                            blist.append(True)
                    return blist

                combine = list([H_id, L_id, N_id])
                blist = replace_gen(combine)

                for i in range(len(combine)):
                    x = combine[i]
                    y = blist[i]
                    np.random.seed(self.seed + 1)

                    if len(x) == len(H_id):
                        high = np.random.choice(x, max_samples, replace=y)
                    elif len(x) == len(L_id):
                        low = np.random.choice(x, max_samples, replace=y)
                    elif len(x) == len(N_id):
                        no = np.random.choice(x, max_samples, replace=y)

            else:
                number_samples = len(H_id) + len(L_id) + len(N_id)
                np.random.seed(self.seed + 1)
                high = np.random.choice(H_id, int(np.floor(number_samples * self.distribution[0])), replace=False)
                low = np.random.choice(L_id, int(np.floor(number_samples * self.distribution[1])), replace=True)
                no = np.random.choice(N_id, int(np.floor(number_samples * self.distribution[2])), replace=False)

            ids = np.hstack([high, low, no])
            np.random.seed(self.seed + 2)
            np.random.shuffle(ids)
            print("Training Indices Length: ", len(ids))
            return iter(ids)

        if self.mode == 'val':
            H_id = H_id[int(np.floor(len(H_id) * self.train_split)): int(np.floor(len(H_id) * self.val_split))]
            L_id = L_id[int(np.floor(len(L_id) * self.train_split)): int(np.floor(len(L_id) * self.val_split))]
            N_id = N_id[int(np.floor(len(N_id) * self.train_split)): int(np.floor(len(N_id) * self.val_split))]

            np.random.seed(self.seed + 2)
            np.random.shuffle(H_id)
            np.random.shuffle(L_id)
            np.random.shuffle(N_id)
            ids = np.hstack([H_id, L_id, N_id])
            np.random.seed(self.seed + 3)
            np.random.shuffle(ids)

            self.n = len(ids)
            print("Validation Indices Length:", len(ids))
            return iter(ids)

        if self.mode == 'test':
            H_id = H_id[int(np.floor(len(H_id) * self.val_split)):]
            L_id = L_id[int(np.floor(len(L_id) * self.val_split)):]
            N_id = N_id[int(np.floor(len(N_id) * self.val_split)):]

            ids = np.hstack([H_id, L_id, N_id])
            np.random.seed(self.seed + 4)
            np.random.shuffle(ids)
            self.n = len(ids)
            print("Testing Indices Length:", len(ids))
            return iter(ids)

    def __len__(self):
        return self.n


'''
ProSampler class allows to do Train-Validation-Test split of data, oversampling, undersampling, custom sampling
Has 3 modes: 'train', 'val', 'test'
Only oversamples or undersamples or custom samples on train mode
Has 3 functions: 'oversample', 'undersample', 'original'  | original keeps the original data distribution
'''


class ProSampler(Sampler):

    def __init__(self, dataset, mode, function, num_class, distribution_list=None, train_split=0.8, val_split=0.1, shuffle=True, seed=1):
        self.dataset = dataset
        self.distribution = distribution_list
        self.train_split = train_split
        self.shuffle = shuffle
        self.seed = seed
        self.mode = mode
        self.val_split = val_split
        self.num_class = num_class
        self.function = function

        if self.function == 'split' or self.function == 'both':
            if self.mode == 'train':
                self.n = int(len(self.dataset) * self.train_split)
            if self.mode == 'val':
                self.n = int((len(self.dataset) * self.val_split))
            if self.mode == 'test':
                self.n = int((len(self.dataset) * (1 - self.train_split - self.val_split)))

            self.val_split = val_split + train_split

        elif self.function == 'oversample' or self.function == 'undersample':
            self.n = len(self.dataset)

        self.index_by_class_list = [[] for cls in range(self.num_class)]

        for i in range(len(self.dataset)):
            _, tar = self.dataset.__getitem__(i)
            if type(tar) is not int:
                tar = tar[0]
            self.index_by_class_list[tar].append(i)

        self.index_by_class_array = np.array([np.array(xi) for xi in self.index_by_class_list], dtype=object)

    def __iter__(self):

        if self.shuffle is True:  # shuffle in each class seperately
            np.random.seed(self.seed)
            for array in self.index_by_class_array:
                np.random.shuffle(array)

        if self.mode == 'train':
            for num, array in enumerate(self.index_by_class_array):
                self.index_by_class_array[num] = array[:int(np.floor(len(array) * self.train_split))]

            if self.function == 'oversample':
                if self.distribution is None:  # make all classes same number
                    max_samples = 0
                    for array in self.index_by_class_array:
                        if len(array) > max_samples:
                            max_samples = len(array)

                    def replace_gen(combine_class):
                        blist = []
                        for el in combine_class:
                            if len(el) == max_samples:
                                blist.append(False)
                            else:
                                blist.append(True)
                        return blist

                    blist = replace_gen(self.index_by_class_array)

                    for i in range(len(self.index_by_class_array)):
                        x = self.index_by_class_array[i]
                        y = blist[i]
                        np.random.seed(self.seed + 1)

                        self.index_by_class_array[i] = np.random.choice(x, max_samples, replace=y)

                else:  # use a custom distribution provided as distribution_list
                    number_samples = 0
                    np.random.seed(self.seed + 1)
                    for i in range(len(self.index_by_class_array)):
                        number_samples += len(self.index_by_class_array[i])

                    for i in range(len(self.index_by_class_array)):
                        x = self.index_by_class_array[i]
                        if int(np.floor(number_samples * self.distribution[i])) > len(x):
                            replace = True
                        else:
                            replace = False

                        self.index_by_class_array[i] = np.random.choice(
                            x, int(np.floor(number_samples * self.distribution[i])), replace=replace)

            elif self.function == 'undersample':
                if self.distribution is None:  # make all classes same number
                    min_samples = None

                    for array in self.index_by_class_array:
                        if min_samples is not None:
                            if len(array) < min_samples:
                                min_samples = len(array)
                        else:
                            min_samples = len(array)

                    for i in range(len(self.index_by_class_array)):
                        x = self.index_by_class_array[i]
                        np.random.seed(self.seed + 1)

                        self.index_by_class_array[i] = np.random.choice(x, min_samples, replace=False)

                else:  # use a custom distribution provided as distribution_list
                    number_samples = None
                    np.random.seed(self.seed + 1)
                    div_dist = 1
                    for i in range(len(self.index_by_class_array)):
                        if number_samples is not None:
                            if len(self.index_by_class_array[i]) < number_samples:
                                number_samples = len(self.index_by_class_array[i])
                        else:
                            number_samples = len(self.index_by_class_array[i])

                    for i in range(len(self.index_by_class_array)):
                        x = self.index_by_class_array[i]
                        if len(x) == number_samples:
                            div_dist = self.distribution[i]

                    for i in range(len(self.index_by_class_array)):
                        x = self.index_by_class_array[i]
                        if not len(x) == number_samples:
                            self.index_by_class_array[i] = np.random.choice(
                                x, int(np.floor((len(x)/div_dist) * self.distribution[i])), replace=False)

            elif self.function == 'original':
                assert self.distribution is None, print("Cannot have custom distribution and original distribution!")
                pass

            ids = np.hstack(self.index_by_class_array)
            np.random.seed(self.seed + 2)
            np.random.shuffle(ids)
            print("Training Indices Length: ", len(ids))
            return iter(ids)

        if self.mode == 'val':
            for i in range(len(self.index_by_class_array)):
                x = self.index_by_class_array[i]
                self.index_by_class_array[i] = x[int(
                    np.floor(len(x) * self.train_split)): int(np.floor(len(x) * self.val_split))]

            np.random.seed(self.seed + 2)
            for array in self.index_by_class_array:
                np.random.shuffle(array)

            ids = np.hstack(self.index_by_class_array)
            np.random.seed(self.seed + 3)
            np.random.shuffle(ids)

            self.n = len(ids)
            print("Validation Indices Length:", len(ids))
            return iter(ids)

        if self.mode == 'test':
            for i in range(len(self.index_by_class_array)):
                x = self.index_by_class_array[i]
                self.index_by_class_array[i] = x[int(np.floor(len(x) * self.val_split)):]

            ids = np.hstack(self.index_by_class_array)
            np.random.seed(self.seed + 4)
            np.random.shuffle(ids)
            self.n = len(ids)
            print("Testing Indices Length:", len(ids))
            return iter(ids)

    def __len__(self):
        return self.n


class SequenceIDSampler(Sampler):
    def __init__(self, dataset, csv_file):
        super(SequenceIDSampler, self).__init__(dataset)
        self.dataset = dataset
        self.csv = csv_file
        self.df = pd.read_csv(self.csv, sep=',')
        self.seq_id_unique = sorted(list(set(self.df.sequence_id)))
        shuffle(self.seq_id_unique)

    def __iter__(self):
        sequence_id_dict = {}
        for seq_id in self.seq_id_unique:
            ids = [i for i, x in enumerate(self.df.sequence_id) if x == seq_id]
            sequence_id_dict.update({seq_id: ids})

        return iter(sequence_id_dict.values())

    def __len__(self):
        pass

'''
csv_file = 'D:/Interacction/Interaction_Dataset/Completed_Annotation/processed_csv/zip10_processed.csv'
root_image = 'D:/Interacction/Interaction_Dataset/zip10'
dataset_our = CustomDatasets.InteractionImageDataset(csv_file, root_image, mode='train')
SequenceIDSampler(dataset_our, csv_file)
SequenceIDSampler.__iter__(self=SequenceIDSampler(dataset_our, csv_file))
'''

def my_collate(batch):
    batch = filter(lambda img: img is not None, batch)
    return dataloader.default_collate(list(batch))


def exp1(csv_file, root, mode, transform, batch_size=50):

    #dataset = datasets.ImageFolder(root=root, transform=transform)
    dataset = CustomDatasets.InteractionImageDataset(csv_file=csv_file, root_image=root, mode=mode, transform=transform)

    train_dataloader = DataLoader(dataset, num_workers=0, pin_memory=False, batch_size=batch_size, shuffle=False)  # sampler=oversampler(dataset, mode='train')

    val_dataloader = DataLoader(dataset, num_workers=0, pin_memory=False, batch_size=batch_size)  # sampler=oversampler(dataset, mode='val')

    test_dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size)  # sampler=oversampler(dataset, mode='test')

    return train_dataloader, val_dataloader, test_dataloader, dataset


def extract(csv_file, root, mode, transform, batch_size=50):
    dataset = CustomDatasets.InteractionImageDataset(csv_file=csv_file, root_image=root, mode=mode, transform=transform)
    dataloader = DataLoader(dataset, pin_memory=False, batch_size=batch_size, shuffle=False, collate_fn=my_collate)

    return dataset, dataloader


def keypoint_extract(csv_file, transform=None, batch_size=64):
    dataset = CustomDatasets.KeypointDataset(csv_file, transform=transform)
    dataloader = DataLoader(dataset, pin_memory=False, batch_size=batch_size, shuffle=False, collate_fn=my_collate)

    return dataset, dataloader