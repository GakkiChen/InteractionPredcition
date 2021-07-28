import os
import sys
import CustomDatasets
from ModelFinal import LstmAutoEncoder
from torch.utils.data import DataLoader
import torch
import numpy as np
import json
import pickle

def validation(model_path, mode='test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LstmAutoEncoder(1365, 512).to(device)
    criteron = torch.nn.MSELoss(reduction='mean').to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    dataset = CustomDatasets.ContextFeatureDataset(mode=mode)

    with torch.no_grad():
        latent_tensor = torch.zeros((dataset.__len__(), 512))
        model.eval()
        loss_list = []
        for i in range(dataset.__len__()):
            feature, _, _ = dataset.__getitem__(i)
            feature = feature.unsqueeze(0).to(device)
            encoded, decoded = model(feature)
            latent_tensor[i] = encoded
            loss = criteron(decoded, feature)
            loss_list.append(loss.item())
            #if loss.item() < 0.1:
             #   print(feature, decoded, loss.item())
        #print('Loss| mean:', np.mean(loss_list), 'min:', np.min(loss_list), 'max:', np.max(loss_list), 'std:', np.std(loss_list))
        print(mode, dataset.__len__())

        folder_name = os.path.join(os.getcwd(), 'latent_vector')
        os.makedirs(folder_name, exist_ok=True)
        with open(os.path.join(folder_name, f'{mode}_{dataset.__len__()}_latent_vector.pt'), 'wb') as handle:
            torch.save(latent_tensor, handle, pickle_protocol=pickle.HIGHEST_PROTOCOL)

        return latent_tensor

def training(model, train_dataloader, num_epochs, optimiser, loss_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criteron = torch.nn.MSELoss(reduction='mean').to(device)

    for epoch in range(num_epochs):
        model.train()

        losses_list = []
        for feature, file, index in train_dataloader:
            optimiser.zero_grad()
            feature.to(device)
            encoded, decoded = model(feature)

            loss = criteron(decoded, feature)
            #print(file, loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimiser.step()
            losses_list.append(loss.item())

        print(epoch, np.mean(losses_list), losses_list)
        loss_list.append(np.mean(losses_list))

'''
model_save_path = os.path.join(os.getcwd(), f"new_lstm_autoeconder_dim512.pth.tar")
ntensor = validation(model_save_path, mode='test')
sys.exit()
'''

'''
for latent_size in [512, 256]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LstmAutoEncoder(1365, latent_size).to(device)
    learning_rate = 0.001
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = CustomDatasets.ContextFeatureDataset()
    dataloader = DataLoader(dataset, pin_memory=False, batch_size=1, shuffle=True)
    print("Size", latent_size)
    loss_list = []
    training(model, dataloader, 200, optimiser, loss_list)

    model_save_path = os.path.join(os.getcwd(), f"new_lstm_autoeconder_dim{latent_size}.pth.tar")
    checkpoint = {
        'model_state': model.state_dict(),
        'optimiser_state': optimiser.state_dict()
    }

    torch.save(checkpoint, model_save_path)
    list_save_path = os.path.join(os.getcwd(), "LSTM_auto_loss.txt")

    with open(list_save_path, 'w') as f:
        f.write(json.dumps(loss_list))
'''
