import torch
from models_all_old import ObjectCnn, EmoNet, DeepEmotion, EngagementCNN, ConvoModel
import os
import torchvision.transforms as transforms
from data_loading import extract
import torchvision.models as models
import pickle
import argparse

#from img2pose_main import img2pose  # IF YOU WANT TO USE img2pose ACTIVATE THIS

# cmd interface

parser = argparse.ArgumentParser(description='Get Features of image sequences')
parser.add_argument('-ml', '--model', type=str, metavar='', help='Path to pretrained model')
parser.add_argument('-d', '--dataset', type=str, metavar='', help='Path to dataset')
parser.add_argument('-mo', '--mode', type=str, required=True, metavar='',
                    help='F: Face, O: Object, S: Scene or E1: Emonet or EDE: DeepEmotion or Ecnn: Emotional '
                         'Engagement or Convo: Conversation')
parser.add_argument('-b', '--batch_size', type=int, metavar='', help='Batch size')
args = parser.parse_args()

# Parameters
model_file = args.model
root_dataset = args.dataset  # 'F:/Interacction/Interaction_Dataset'  #path to where all the zip folders are
mode = args.mode
batch_size = args.batch_size

''' USE THIS FOR DEBUGGING OR IF YOU DONT LIKE CMD INTERFACE
model_file = os.path.join(os.getcwd(), 'pretrained_models', 'resnet50convo.pth.tar')
root_dataset = 'D:/Interacction/Interaction_Dataset'  #path to where all the zip folders are
mode = 'Convo'
batch_size = 64
'''
# Define Network
if mode == 'F':
    crop = 'face'
    arch = 'resnet50'
    model = models.__dict__[arch](num_classes=3)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state'])
    folder_name = os.path.join(os.getcwd(), 'extracted_features', 'Face')
    model.eval()

elif mode == 'O':
    crop = 'full'
    model = ObjectCnn()
    folder_name = os.path.join(os.getcwd(), 'extracted_features', 'Object')
    model.eval()

elif mode == 'S':
    crop = 'full'
    arch = 'resnet50'
    model = models.__dict__[arch](num_classes=365)
    folder_name = os.path.join(os.getcwd(), 'extracted_features', 'Scene')

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

elif mode == 'E1':  # only can use batch size 32 with emonet
    crop = 'face'
    model = EmoNet()
    folder_name = os.path.join(os.getcwd(), 'extracted_features', 'EmotionEmonet')

    state_dict_path = model_file
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

elif mode == 'EDE':
    crop = 'face'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepEmotion().to(device)
    folder_name = os.path.join(os.getcwd(), 'extracted_features', 'EmotionDeepemotion')

    state_dict_path = model_file
    state_dict = torch.load(str(state_dict_path))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

elif mode == 'Ecnn':
    crop = 'face'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_name = os.path.join(os.getcwd(), 'extracted_features', 'EngagementCNN')
    model = EngagementCNN().to(device)

    state_dict_path = model_file
    state_dict = torch.load(str(state_dict_path))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

elif mode == 'Convo':
    crop = 'face'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_name = os.path.join(os.getcwd(), 'extracted_features', 'ConvoCNN')
    model = ConvoModel().to(device)

    state_dict_path = model_file
    state_dict = torch.load(str(state_dict_path))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()


elif mode == 'Engagement':
    crop = 'face'
    ''' IF YOU WANT TO USE img2pose ACTIVATE THIS
    threed_points = np.load('./img2pose_main/pose_references/reference_3d_68_points_trans.npy')

    DEPTH = 18
    MAX_SIZE = 1400
    MIN_SIZE = 600

    POSE_MEAN = "./img2pose_main/models/WIDER_train_pose_mean_v1.npy"
    POSE_STDDEV = "./img2pose_main/models/WIDER_train_pose_stddev_v1.npy"
    MODEL_PATH = "./img2pose_main/models/img2pose_v1.pth"

    pose_mean = np.load(POSE_MEAN)
    pose_stddev = np.load(POSE_STDDEV)

    model = img2pose.img2poseModel(DEPTH, MIN_SIZE, MAX_SIZE, pose_mean=pose_mean, pose_stddev=pose_stddev,
                                   threed_68_points=threed_points)

    checkpoint = torch.load(MODEL_PATH)
    model.fpn_model.load_state_dict(checkpoint["fpn_model"], strict=False)
    model.evaluate()

    folder_name = os.path.join(os.getcwd(), 'extracted_features', 'EngagementImg2pose')
'''

else:
    print("INVALID ENTRY")

if batch_size is None:
    batch_size = int(64)
else:
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transformation
if crop == 'face':
    if mode == 'EDE':
        transform_resnet = transforms.Compose([
            transforms.Resize(48),
            transforms.CenterCrop((48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    elif mode == 'Ecnn':
        transform_resnet = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    else:
        transform_resnet = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop((256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
else:
    transform_resnet = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Load Dataset
csv_list = ['zip10', 'zip15', 'zip21', 'zip31', 'zip45']  # CHANGE zips to extract feature from
#csv_list = ['zip29']
for zipfile in csv_list:
    csv_file = os.path.join(root_dataset, 'Completed_Annotation', 'processed_csv', f'{zipfile}_processed.csv')
    dataset, loader = extract(root=root_dataset, transform=transform_resnet, batch_size=batch_size, csv_file=csv_file,
                              mode=crop)

    with torch.no_grad():
        store_tensor = None
        past_seq_list = []
        for images, _, sequence_id in loader:
            images = images.to(device)
            print(images.shape)
            sequences = sequence_id

            if mode == 'Engagement':
                feature = model.predict(images)
                print(len(feature))

            elif mode == 'Ecnn':
                _, feature = model.to(device)(images)
                feature = feature.squeeze(3).squeeze(2)

            elif mode == 'Convo':
                feature = model.to(device)(images)
                feature = feature.squeeze(0)

            else:
                feature = model.to(device)(images)

            for i, x in enumerate(feature):
                if mode == 'Engagement':
                    x = x["dofs"]
                    print(x.shape)
                    if x.shape[0] == 0:
                        x = torch.zeros((1, x.shape[1])).to(device)
                curr_seq = sequences[i]

                if len(past_seq_list) == 0:
                    past_seq_list.append(curr_seq)

                if past_seq_list.__contains__(curr_seq):  # seen sequence
                    if store_tensor is None:  # only one time
                        x = x.unsqueeze(0)
                        store_tensor = x

                    else:
                        idx = past_seq_list.index(curr_seq)

                        if idx == len(past_seq_list) - 1:  # latest sequence
                            assert past_seq_list[idx] == curr_seq, "Wrong Sequence ID! Not the latest seq"
                            x = x.unsqueeze(0)
                            store_tensor = torch.cat((store_tensor, x), dim=0)

                        else:  # some previous sequence
                            seq = past_seq_list[idx]
                            assert seq == curr_seq, "Wrong Sequence ID! Check!"
                            filename = str(zipfile) + '__' + str(seq) + '.pt'
                            ptpath = os.path.join(folder_name, filename)
                            my_dict = {}
                            ft = 0
                            with open(ptpath, 'rb') as handle:  # opens previously saved pt file
                                my_dict = torch.load(handle)
                                assert my_dict['seq_id'] == seq, "Sequence ID does not match in .pt file"
                                ft = my_dict['full_feature']
                                x = x.unsqueeze(0)
                                ft = torch.cat((ft, x), dim=0)

                            with open(ptpath, 'wb') as handle:
                                my_dict['full_feature'] = ft
                                torch.save(my_dict, handle, pickle_protocol=pickle.HIGHEST_PROTOCOL)

                else:  # some unseen sequence
                    mydict = {'seq_id': past_seq_list[-1], 'full_feature': store_tensor}

                    os.makedirs(folder_name, exist_ok=True)
                    filename = str(zipfile) + '__' + str(past_seq_list[-1]) + '.pt'
                    # filename format is "zip__seqid"

                    with open((os.path.join(folder_name, filename)), 'wb') as handle:
                        torch.save(mydict, handle, pickle_protocol=pickle.HIGHEST_PROTOCOL)

                    x = x.unsqueeze(0)
                    store_tensor = x
                    past_seq_list.append(curr_seq)

