from CustomDatasets import InteractionImageDataset
from data_loading import SequenceIDSampler
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import torch
from ModelFinal import InteractionPipeline
import torch.nn as nn
import torch.optim as optim
from random import shuffle
from torch.utils.data import Dataset

root_dataset = '/data/chen/work/memory_aug/final_data'  # ONLY NEED TO CHANGE THIS (path to where all the zip folders are)
mode = 'both'
num_epoch = 30
num_classes = 3
learning_rate = 3e-04

model_dir = os.path.join(os.getcwd(), "pretrained_models")


class CombinedCompetence(Dataset):
    def __init__(self):
        super(CombinedCompetence, self).__init__()

        self.path = os.path.join(os.getcwd(), 'extracted_features')

        self.emo_feature_folder = os.path.join(self.path, 'EmotionEmonet')
        self.eng_feature_folder = os.path.join(self.path, 'EngagementCNN')
        self.convo_feature_folder = os.path.join(self.path, 'ConvoCNN')

        self.emo_feature_file_list = os.listdir(self.emo_feature_folder)
        self.eng_feature_file_list = os.listdir(self.eng_feature_folder)
        self.convo_feature_file_list = os.listdir(self.convo_feature_folder)

    def __getitem__(self, idx):  # index of item to be retrieved
        emo_feature_file = os.path.join(self.emo_feature_folder, self.emo_feature_file_list[idx])
        with open(emo_feature_file, 'rb') as ftfile:
            imgdict = torch.load(ftfile)
            emo_feature = imgdict['full_feature']

        eng_feature_file = os.path.join(self.eng_feature_folder, self.eng_feature_file_list[idx])
        with open(eng_feature_file, 'rb') as ftfile:
            imgdict = torch.load(ftfile)
            eng_feature = imgdict['full_feature']

        convo_feature_file = os.path.join(self.convo_feature_folder, self.convo_feature_file_list[idx])
        with open(convo_feature_file, 'rb') as ftfile:
            imgdict = torch.load(ftfile)
            convo_feature = imgdict['full_feature']

        return emo_feature, eng_feature, convo_feature

    def __len__(self):
        #assert self.emo_comp.neigh_seq.shape[0] == self.eng_comp.neigh_seq.shape[0], "Two Datasets size mismatch! ERROR"
        return len(self.emo_feature_file_list)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

combined_dataset = CombinedCompetence()
model = InteractionPipeline(combined_dataset).to(device)
optimiser = optim.Adam(model.parameters(), lr=learning_rate)
weight = torch.tensor([0.488, 0.411, 0.101]).to(device)
criteria = nn.CrossEntropyLoss()

transform_resnet = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop((256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

# Load in all pretrained weights as seperate modules

checkpoint = torch.load(os.path.join(model_dir, 'emonet_8.pth'))  # emonet CNN
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.attexr.emotion_arm.emonet.load_state_dict(state_dict, strict=False)

checkpoint = torch.load(os.path.join(model_dir, 'EmotionEmonet_RNN_latest_25July.pth.tar'))  # emotion rnn
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.attexr.emotion_arm.lstm.load_state_dict(state_dict, strict=False)

checkpoint = torch.load(os.path.join(model_dir, 'engagement_model.pth'))  # engagement cnn
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.attexr.engagement_arm.engage.load_state_dict(state_dict, strict=False)

checkpoint = torch.load(os.path.join(model_dir, 'EngagementCNN_RNN_latest_20July.pth.tar'))  # enagement rnn
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.attexr.engagement_arm.lstm.load_state_dict(state_dict, strict=False)

checkpoint = torch.load(os.path.join(model_dir, 'resnet50convo.pth.tar'))  # convo cnn + rnn
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.attexr.convo_arm.load_state_dict(state_dict, strict=False)

checkpoint = torch.load(os.path.join(model_dir, 'resnet50_places365.pth.tar'))  # scene cnn
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.weightgen.neighgen.scenecnn.load_state_dict(state_dict, strict=False)

checkpoint = torch.load(os.path.join(model_dir, 'lstm_autoeconder_dim512.pth.tar'))  # LSTM Autoencoder
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.weightgen.neighgen.encoder.load_state_dict(state_dict, strict=False)

#checkpoint = torch.load(os.path.join(model_dir, 'mlp_weight_gen.pth.tar'))  # competence classifier
#state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
#model.weightgen.cc.load_state_dict(state_dict, strict=False)

checkpoint = torch.load(os.path.join(model_dir, 'logreg.pth.tar'))  # final interaction classifier
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.interact.load_state_dict(state_dict, strict=False)


csv_list = ['zip10', 'zip15', 'zip21', 'zip31']
#csv_list = ['zip29']
max_acc = 0.0
for epoch in range(num_epoch):
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

    for zipfile in csv_list:
        csv_file = os.path.join(root_dataset, 'Completed_Annotation', 'processed_csv', f'{zipfile}_processed.csv')

        train_dataset = InteractionImageDataset(csv_file=csv_file, root_image=root_dataset, mode=mode,
                                          transform=transform_resnet, convo=False)

        train_loader = DataLoader(train_dataset, pin_memory=False, shuffle=False,
                                  sampler=SequenceIDSampler(train_dataset, csv_file))

        for full_images, face_images, label, sequence_id in train_loader:
            full_images = full_images.squeeze(0).to(device)
            face_images = face_images.squeeze(0).to(device)
            print(full_images.shape)
            labels = label.to(device)
            sequences = sequence_id

            scores = model(fulls=full_images, crops=face_images)
            scores = scores.unsqueeze(0)

            if criteria.__class__ == nn.CrossEntropyLoss:
                loss = criteria(scores, labels)
                _, predictions = torch.max(scores, -1)

            else:
                loss = criteria(scores, labels.type_as(scores))
                predictions = (scores > 0.5).long()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            predictions = predictions.detach()
            print(f"Target:{labels.item()}, Predicted:{predictions.item()}, Loss:{loss.item()}")
            del full_images, face_images, loss

            pred_list.append(predictions.data.tolist())
            c = (predictions == labels)
            num_correct = int(c.sum())
            total_correct += num_correct
            total_sample += len(labels)

            for i in range(len(labels)):
                label = int(labels[i])
                pred = predictions[i]
                correct_per_class[label] += c[i].item()
                total_per_class[label] += 1.0
                total_per_pred[pred] += 1.0

            #print(f"Target Distribution: {total_per_class}, Predicted Distribution: {total_per_pred}")

    print(f"Target Distribution: {total_per_class}, Predicted Distribution: {total_per_pred}")
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

    #writer.add_scalar("Training Accuracy", mean_acc, global_step=epoch)  #
    #end = time.time()
    #training_time = end - start

    print(" --------------------------TRAINING-------------------------------------- ")
    print(f'Epoch: {epoch} | Training Metrics |')
    print(f'Avg Accuracy: {mean_acc}, Time:')
    print(f'Precision: {precision_list}\nRecall: {recall_list}\nF1: {f1_list}')
    print(" ------------------------------------------------------------------------ ")

    '''
    if mean_acc >= max_acc:
        max_acc = mean_acc

        checkpoint = {
            'epoch': epoch,
            'model_state': model.interact.state_dict(),
            'optimiser_state': optimiser.state_dict()
        }
        os.makedirs(os.path.join(os.getcwd(), 'generated_models', 'Int_mlp'), exist_ok=True)
        model_save_path = os.path.join(os.getcwd(), 'generated_models', 'Int_mlp', 'mlp.pth.tar')
        torch.save(checkpoint, model_save_path)
    '''
