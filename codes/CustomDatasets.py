import csv
from utils import file_list_generator
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import ast
import sys
from ModelFinal import ModalRnn


class InteractionImageDataset(Dataset):  # Used to get Big image or face crop or both
    def __init__(self, csv_file, root_image, mode, transform=None, convo=False):

        """
        :param csv_file: location of csv file
        :param root_image: location of all the user data(all zip folders)
        :param transform: transformation
        :param mode: full or face or both
        """

        self.dataframe = pd.read_csv(csv_file)
        self.rootimage = root_image
        self.transform = transform
        self.mode = mode
        self.convo = convo

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        def getsample(idx):
            face_crop_path = os.path.join(self.rootimage, self.dataframe.iloc[idx, 2])

            big_image_path = os.path.join(self.rootimage, self.dataframe.iloc[idx, 0])

            interaction_class = str(self.dataframe.iloc[idx, 6])

            seq_id = self.dataframe.iloc[idx, 5]

            attributes = list(ast.literal_eval(self.dataframe.iloc[idx, 8]))
            convo = [attributes.pop(i) for i in [0, 11]]

            if sum(convo) > 0.999:
                pass
            else:
                convo[1] += round((1.0 - convo[0] - convo[1]), ndigits=3)

            convo_label = convo.index(max(convo))

            samples = {'face_image': face_crop_path, 'full_image': big_image_path, 'sequence': seq_id}
            targets = {'class': interaction_class, 'class_index': interaction_class, 'convo_label': convo_label}

            if self.mode == 'full':
                try:
                    sample = Image.open(samples['full_image'])
                    sample.convert('RGB')

                except FileNotFoundError:
                    print('Invalid Big Image: ', big_image_path)
                    return None, None, None

            elif self.mode == 'face':
                try:
                    sample = Image.open(samples['face_image'])
                    sample.convert('RGB')

                except FileNotFoundError:
                    print('Invalid Face crop: ', face_crop_path)
                    return None, None, None

            elif self.mode == 'both':
                try:
                    sample_full = Image.open(samples['full_image'])
                    sample_full.convert('RGB')
                    sample_face = Image.open(samples['face_image'])
                    sample_face.convert('RGB')

                except FileNotFoundError:
                    print('Invalid:', face_crop_path, 'or', big_image_path)
                    return None, None, None
            else:
                print("MODE DOES NOT EXIST! Check mode")
                sys.exit()

            if self.convo is False:
                target = targets['class_index']
            else:
                target = targets['convo_label']

            sequence_id = samples['sequence']

            if (self.mode == 'face' or self.mode == 'full') and self.transform is not None:
                if sample is not None:
                    sample = self.transform(sample)

                    return sample, target, sequence_id

            elif self.mode == 'both' and self.transform is not None:
                if sample_full is not None and sample_face is not None:
                    sample_full = self.transform(sample_full)
                    sample_face = self.transform(sample_face)

                    return sample_full, sample_face, target, sequence_id

        if self.mode == 'face' or self.mode == 'full':
            if type(idx) == int:
                sample, target, sequence_id = getsample(idx)

                return sample, target, sequence_id

            else:
                target_list = []
                sequence_id_list = []
                sample_tensor = []
                for i, index in enumerate(idx):
                    sample, target, sequence_id = getsample(index)
                    sample = sample.unsqueeze(dim=0)

                    if i == 0:
                        sample_tensor = sample
                    else:
                        sample_tensor = torch.cat((sample_tensor, sample), dim=0)

                    sequence_id_list.append(sequence_id), target_list.append(target)

                assert len(set(target_list)) == 1, "Target list ERROR!"
                assert len(set(sequence_id_list)) == 1, "Sequence ID list ERROR!"

                if sample_tensor.shape[0] > 30:  # truncating longer sequences due to GPU memory issue
                    sample_tensor = sample_tensor[:30]

                return sample_tensor, int(target), sequence_id

        elif self.mode == 'both':
            if type(idx) == int:
                sample_full, sample_face, target, sequence_id = getsample(idx)

                return sample_full, sample_face, target, sequence_id

            else:
                target_list = []
                sequence_id_list = []
                sample_full_tensor = []
                sample_face_tensor = []
                for i, index in enumerate(idx):
                    sample_full, sample_face, target, sequence_id = getsample(index)
                    sample_full = sample_full.unsqueeze(dim=0)
                    sample_face = sample_face.unsqueeze(0)

                    if i == 0:
                        sample_full_tensor = sample_full
                        sample_face_tensor = sample_face
                    else:
                        sample_full_tensor = torch.cat((sample_full_tensor, sample_full), dim=0)
                        sample_face_tensor = torch.cat((sample_face_tensor, sample_face), dim=0)

                    sequence_id_list.append(sequence_id), target_list.append(target)

                assert len(set(target_list)) == 1, "Target list ERROR!"
                assert len(set(sequence_id_list)) == 1, "Sequence ID list ERROR!"

                if sample_full_tensor.shape[0] > 30:  # truncating longer sequences due to GPU memory issue
                    sample_full_tensor = sample_full_tensor[:30]

                if sample_face_tensor.shape[0] > 30:  # truncating longer sequences due to GPU memory issue
                    sample_face_tensor = sample_face_tensor[:30]

                return sample_full_tensor, sample_face_tensor, int(target), sequence_id


class FeatureDataset(Dataset):  # for Emotion, Convsersation, Engagement
    def __init__(self, mod, csv_path, feature_path=None, cos=False, train=True, interaction=False):
        from utils import file_list_generator

        if feature_path is None:
            if train is True:
                self.path = os.path.join(os.getcwd(), 'extracted_features')

            else:
                self.path = os.path.join(os.getcwd(), 'extracted_features_test')
        else:
            self.path = feature_path

        if cos is True:
            self.cos = True

        else:
            self.cos = False

        self.modality = mod
        self.masterlist = file_list_generator(self.path)
        self.modlist = [foldername for foldername in os.listdir(self.path)]
        self.mod_index = self.modlist.index(self.modality)
        self.csv_path = csv_path  # path to folder where all processed csvs are stored
        self.interaction = interaction

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = str(self.masterlist[self.mod_index][idx])
        ptpath = os.path.join(self.path, str(self.modlist[self.mod_index]), file)

        with open(ptpath, 'rb') as handle:
            my_dict = torch.load(handle)
            feature = my_dict['full_feature']
            if self.modality == "EmotionResnet":
               feature = feature.squeeze(3).squeeze(2)
            zipfolder = file.split("__")[0]
            csv_file = os.path.join(self.csv_path, f'{zipfolder}_processed.csv')

            with open(csv_file, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:  # finding for the sequence id
                    attributes_annotation = []
                    if file.split("__")[-1][:-3] == row[5]:
                        attributes_annotation = ast.literal_eval(row[8])
                        interaction_label = int(row[6])
                        break
                if 'Emotion' in self.modality:
                    index_list = [4, 9]

                elif 'Engagement' in self.modality:
                    index_list = [1, 12]

                assert len(attributes_annotation) > 0, f"ERROR Sequence ID: {file[:-3]} not found in csv"
                mod_label = [attributes_annotation[i] for i in index_list]

                if np.mean(mod_label) == np.max(mod_label):  # both N and Y are the same

                    if np.mean(mod_label) > 0.5:
                        label = 0
                    else:
                        label = 1  # assume no

                else:
                    assert np.mean(mod_label) < np.max(mod_label), "ERROR in emotion label!"
                    label = mod_label.index(max(mod_label))

            if self.interaction is True:
                return feature, interaction_label

            elif self.cos is True:
                return feature, label, idx  # label: 0 - Present, 1 - Not present

            else:
                return feature, label

    def __len__(self):
        return len(self.masterlist[self.mod_index])


class ContextFeatureDataset(Dataset):  # for Face, Object, Scene
    def __init__(self, feature_path=None, mode='train'):  # mode is train or test

        if feature_path is None and mode == 'train':
            self.path = os.path.join(os.getcwd(), 'extracted_features')

        elif feature_path is None and mode == 'test':
            self.path = os.path.join(os.getcwd(), 'extracted_features_test')
        else:
            self.path = feature_path

        self.cos = True

        self.masterlist = file_list_generator(self.path)
        self.modlist = [foldername for foldername in os.listdir(self.path)]
        self.mod_index_list = [self.modlist.index(mod) for mod in ['Object', 'Scene']]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        concat_feature = None
        for index in self.mod_index_list:
            file = str(self.masterlist[index][idx])
            #print("file", file)
            if index == max(self.mod_index_list):
                assert file == str(self.masterlist[index-1][idx]), "ERROR"  # check between Object and Scene
            ptpath = os.path.join(self.path, str(self.modlist[index]), file)

            with open(ptpath, 'rb') as handle:
                my_dict = torch.load(handle)
                feature = my_dict['full_feature']
                label = file

                if concat_feature is None:
                    concat_feature = feature

                else:
                    concat_feature = torch.cat((concat_feature, feature), dim=-1)

        return concat_feature, label, idx

    def __len__(self):
        length = 0
        for num, index in enumerate(self.mod_index_list):
            if length == 0 and num == 0:
                length = len(self.masterlist[index])
            else:
                assert length == len(self.masterlist[index]), "Context Modalities dataset size mismatch!"
        return length


class CompetenceDataset(Dataset):
    def __init__(self, processed_csv_folder_path, mode, mod, model_file, input_size, feature_path=None):
        # Get file
        cwd = os.getcwd()
        neigh_path = os.path.join(cwd, 'neighbourhood', 'neighbourhood.pt')

        if feature_path is None and mode == 'train':
            self.path = os.path.join(os.getcwd(), 'extracted_features')

        elif feature_path is None and mode == 'test':
            self.path = os.path.join(os.getcwd(), 'extracted_features_test')
        else:
            self.path = feature_path

        with open(neigh_path, 'rb') as f:
            mydict = torch.load(f)
            neigh_seq = mydict['seq']
            neigh_similarity = mydict['score']

        self.neigh_seq = neigh_seq
        self.neigh_similarity = neigh_similarity
        self.processed_path = processed_csv_folder_path
        self.modlist = [foldername for foldername in os.listdir(self.path)]
        self.mod_index_list = [self.modlist.index(mod) for mod in ['Object', 'Scene']]
        self.index = self.mod_index_list[0]
        self.masterlist = file_list_generator(self.path)
        self.mod = mod
        self.mode = mode
        self.input_size = input_size

        # Getting all the sequence id for each index
        self.idx2seqid = []
        for i in range(neigh_seq.shape[0]):
            seqid = str(self.masterlist[self.index][i])[:-3]
            self.idx2seqid.append(seqid)

        # Getting all the latent vector
        self.latent_folder = os.path.join(os.getcwd(), "latent_vector")
        latent_file_list = os.listdir(self.latent_folder)
        latent_file = [l for l in latent_file_list if l.__contains__(self.mode)]
        assert len(latent_file) == 1, "LATENT VECTOR FILE ERROR"
        self.latent_file = latent_file

        with open(os.path.join(self.latent_folder, *self.latent_file), 'rb') as f:
            self.all_vectors = torch.load(f)

        # Initalize Model
        input_size = input_size   # size of output cnn feature
        self.model = ModalRnn(feature_size=input_size)
        checkpoint = torch.load(model_file)
        self.model.load_state_dict(checkpoint['model_state'])

        if self.mod == 'emo':
            self.feature_folder = os.path.join(self.path, 'EmotionEmonet')
        elif self.mod == 'eng':
            self.feature_folder = os.path.join(self.path, 'EngagementCNN')

        self.feature_file_list = os.listdir(self.feature_folder)

    def __getitem__(self, idx):  # idx is index of original sequence
        if torch.is_tensor(idx):
            idx = idx.tolist()

        neighbours_index_list = self.neigh_seq[idx]

        gt_neigh = 0
        latent_neigh = 0
        feature_neigh = 0

        for index in neighbours_index_list:
            seq_id = self.idx2seqid[index]
            folder = seq_id.split("__")[0]

            latent_feature = self.all_vectors[index]
            img_feature_file = os.path.join(self.feature_folder, self.feature_file_list[index])

            with open(img_feature_file, 'rb') as ftfile:
                imgdict = torch.load(ftfile)
                img_feature = imgdict['full_feature']

            csv_file = os.path.join(self.processed_path, f'{folder}_processed.csv')

            with open(csv_file, 'r') as handle:
                reader = csv.reader(handle, delimiter=',')
                for idx, row in enumerate(reader):
                    if idx != 0:
                        reason_att = ast.literal_eval(row[7])
                        all_att = ast.literal_eval(row[8])

                        if self.mod == 'emo':
                            if row[5] == seq_id.split("__")[-1]:
                                total_prob = round((all_att[4] + all_att[9]), 1)
                                # adjust so that total attributes add up to 1
                                if total_prob < 1.0:
                                    all_att[9] = round((1.0 - all_att[4]), 3)

                                assert reason_att[3] <= all_att[4] and reason_att[7] <= all_att[9], "ERROR in processing!"

                                gt_list = [None for i in range(2)]

                                if all_att[4] == 0.:
                                    gt_list[0] = 0.0
                                else:
                                    gt_list[0] = reason_att[3] / all_att[4]

                                if all_att[9] == 0.:
                                    gt_list[1] = 0.0
                                else:
                                    gt_list[1] = reason_att[7] / all_att[9]

                                if type(gt_neigh) is int:
                                    gt_neigh = torch.tensor(gt_list).unsqueeze(0)
                                    break

                                else:
                                    cat_tensor = torch.tensor(gt_list).unsqueeze(0)
                                    gt_neigh = torch.cat((gt_neigh, cat_tensor), 0)
                                    break

                        elif self.mod == 'eng':
                            if row[5] == seq_id.split("__")[-1]:
                                total_prob = round((all_att[1] + all_att[12]), 1)
                                # adjust so that total attributes add up to 1
                                if total_prob < 1.0:
                                    all_att[12] = round((1.0 - all_att[1]), 3)

                                assert reason_att[1] <= all_att[1] and reason_att[10] <= all_att[12], "ERROR in processing!"

                                gt_list = [None for i in range(2)]

                                if all_att[1] == 0.:
                                    gt_list[0] = 0.0
                                else:
                                    gt_list[0] = reason_att[1] / all_att[1]

                                if all_att[12] == 0.:
                                    gt_list[1] = 0.0
                                else:
                                    gt_list[1] = reason_att[10] / all_att[12]

                                if type(gt_neigh) is int:
                                    gt_neigh = torch.tensor(gt_list).unsqueeze(0)
                                    break

                                else:
                                    cat_tensor = torch.tensor(gt_list).unsqueeze(0)
                                    gt_neigh = torch.cat((gt_neigh, cat_tensor), 0)
                                    break

            if type(latent_neigh) is int:
                latent_neigh = latent_feature.unsqueeze(0)

            else:
                cat_tensor = latent_feature.unsqueeze(0)
                latent_neigh = torch.cat((latent_neigh, cat_tensor), 0)

            # DO RNN MODEL INFERENCE
            one_hot_feature = torch.zeros((3))
            with torch.no_grad():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.eval()
                self.model.to(device)
                scores, prob = self.model(img_feature.unsqueeze(0))
                _, predictions = torch.max(scores, 1)
                one_hot_feature[int(predictions)] = 1.0
                one_hot_feature[one_hot_feature.shape[0] - 1] = prob.squeeze(0)[int(predictions)].data

            if type(feature_neigh) is int:
                feature_neigh = one_hot_feature.unsqueeze(0)

            else:
                cat_tensor = one_hot_feature.unsqueeze(0)
                feature_neigh = torch.cat((feature_neigh, cat_tensor), 0)

        comp_feature = torch.cat((latent_neigh, feature_neigh), 1)
        return neighbours_index_list, gt_neigh, latent_neigh, feature_neigh, comp_feature

    def __len__(self):

        return self.neigh_seq.shape[0]


class KeypointDataset(Dataset):
    def __init__(self, csv_file, transform=None):

        """
        :param csv_file: location of csv file
        :param transform: transformation
        """

        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        keypoints = self.dataframe.iloc[idx, 9]
        li = ast.literal_eval(keypoints)
        keypoints = [l for l in li]
        keypoints = torch.tensor(keypoints).flatten()  # [x1, y1, x2, y2, x3, y3 ...] ,len = 140

        if len(keypoints) == 0:
            return None, None, None

        else:

            if self.transform is not None:
                keypoints = self.transform(keypoints)

            interaction_class = str(self.dataframe.iloc[idx, 7])
            class_mapping = {'h': 0, 'l': 1, 'n': 2}       # Maps class to index (CHANGE FOR MORE CLASSES)
            class_to_index = class_mapping[interaction_class]

            seq_id = self.dataframe.iloc[idx, 6]

            result = {'kp':keypoints, 'ci':class_to_index, 'seq':seq_id}

            return keypoints, class_to_index, seq_id

    def __getclasslist__(self):
        interact_list = []
        for i in range(len(self.dataframe)):
            keypoints = self.dataframe.iloc[i, 9]
            li = ast.literal_eval(keypoints)
            keypoints = [l for l in li]

            if len(keypoints) > 0:
                interaction_class = str(self.dataframe.iloc[i, 7])
                class_mapping = {'h': 0, 'l': 1, 'n': 2}  # Maps class to index (CHANGE FOR MORE CLASSES)
                class_to_index = class_mapping[interaction_class]
                interact_list.append(class_to_index)

            else:
                interact_list.append(None)

        return interact_list


'''
image_folder = 'F:\Interacction\Interaction_Dataset'
csv_folder = os.path.join(image_folder, 'completed_Annotation\processed_csv')
folder = 'zip10'
csv_file = os.path.join(csv_folder, f'{folder}_processed.csv')
dataset = InteractionImageDataset(csv_file, image_folder, 'face')

for l in range(10000):
    sample, target, seq = dataset.__getitem__(l)
    print(sample, target, seq)
'''

#dataset = ContextFeatureDataset(mode='test')
#f, l, idx = dataset.__getitem__(0)
#print(f.shape, l, idx)

'''
for idx in range(1980):
    modality = 'EmotionFER'
    csv_path = 'D:/Interacction/Interaction_Dataset/Completed_Annotation/processed_csv'
    dataset = FeatureDataset(modality, csv_path)
    f, l = dataset.__getitem__(idx)
    print(f.shape, l)
'''

'''
path = 'D:/Interacction/Interaction_Dataset/Completed_Annotation/processed_csv'
model_file = os.path.join(os.getcwd(), 'generated_models', 'ModalRnn_EmotionEmonet', 'EmotionEmonet_RNNemo.pth.tar')
dataset = CompetenceDataset(path, mode='train', mod='emo', model_file=model_file)
i, x, z, y, zy = dataset.__getitem__(0)
print(i.shape, x.shape, z.shape, y.shape, zy.shape)
pass
'''





