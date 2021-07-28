import os
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, xavier_uniform_

nn.InstanceNorm2d = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.InstanceNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.InstanceNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.InstanceNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class EmoNet(nn.Module):
    def __init__(self, num_modules=2, n_expression=8, n_reg=2, n_blocks=4, attention=True, temporal_smoothing=False):
        super(EmoNet, self).__init__()
        self.num_modules = num_modules
        self.n_expression = n_expression
        self.n_reg = n_reg
        self.attention = attention
        self.temporal_smoothing = temporal_smoothing
        self.init_smoothing = False

        if self.temporal_smoothing:
            self.n_temporal_states = 5
            self.init_smoothing = True
            self.temporal_weights = torch.Tensor([0.1, 0.1, 0.15, 0.25, 0.4]).unsqueeze(0).unsqueeze(
                2).cuda()  # Size (1,5,1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.InstanceNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.InstanceNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            68, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(68,
                                                                 256, kernel_size=1, stride=1, padding=0))
        # Do not optimize the FAN
        for p in self.parameters():
            p.requires_grad = False

        if self.attention:
            n_in_features = 256 * (num_modules + 1)  # Heatmap is applied hence no need to have it
        else:
            n_in_features = 256 * (num_modules + 1) + 68  # 68 for the heatmap

        n_features = [(256, 256)] * (n_blocks)

        self.emo_convs = []
        self.conv1x1_input_emo_2 = nn.Conv2d(n_in_features, 256, kernel_size=1, stride=1, padding=0)
        for in_f, out_f in n_features:
            self.emo_convs.append(ConvBlock(in_f, out_f))
            self.emo_convs.append(nn.MaxPool2d(2, 2))
        self.emo_net_2 = nn.Sequential(*self.emo_convs)
        self.avg_pool_2 = nn.AvgPool2d(4)
        self.emo_fc_2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                      nn.Linear(128, self.n_expression + n_reg))

    def forward(self, x, reset_smoothing=False):

        # Resets the temporal smoothing
        if self.init_smoothing:
            self.init_smoothing = False
            self.temporal_state = torch.zeros(x.size(0), self.n_temporal_states, self.n_expression + self.n_reg).cuda()
        if reset_smoothing:
            self.temporal_state = self.temporal_state.zeros_()

        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.max_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        hg_features = []

        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            tmp_out = self._modules['l' + str(i)](ll)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            hg_features.append(ll)

        hg_features_cat = torch.cat(tuple(hg_features), dim=1)

        if self.attention:
            mask = torch.sum(tmp_out, dim=1, keepdim=True)
            hg_features_cat *= mask
            emo_feat = torch.cat((x, hg_features_cat), dim=1)
        else:
            emo_feat = torch.cat([x, hg_features_cat, tmp_out], dim=1)

        emo_feat_conv1D = self.conv1x1_input_emo_2(emo_feat)
        final_features = self.emo_net_2(emo_feat_conv1D)
        final_features = self.avg_pool_2(final_features)

        batch_size = final_features.shape[0]
        final_features1 = final_features.squeeze(dim=3).squeeze(dim=2)
        final_features = self.emo_fc_2(final_features1)

        if self.temporal_smoothing:
            with torch.no_grad():
                self.temporal_state[:, :-1, :] = self.temporal_state[:, 1:, :]
                self.temporal_state[:, -1, :] = final_features
                final_features = torch.sum(self.temporal_weights * self.temporal_state, dim=1)

        return final_features1
        # {'heatmap': tmp_out, 'expression': final_features[:, :-2], 'valence': final_features[:, -2], 'arousal': final_features[:, -1]}

    def eval(self):

        for module in self.children():
            module.eval()


class EngagementCNN(nn.Module):
    def __init__(self):
        super(EngagementCNN, self).__init__()
        self.resnet = torchvision.models.resnet18()
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 1), nn.Sigmoid())
        self.activation = {}
        self.resnet.avgpool.register_forward_hook(self.get_activation('avg_pool'))

    def forward(self, x):
        output = self.resnet(x)

        return output, self.activation['avg_pool']

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook


class ModalRnn(nn.Module):
    def __init__(self, feature_size, dropout=0.3, out_features=2):  # input should be (batch_size, seq_len, features)
        super(ModalRnn, self).__init__()
        self.feature_size = feature_size
        self.dropout = dropout
        self.rnn = nn.LSTM(self.feature_size, dropout=dropout, hidden_size=self.feature_size, batch_first=True,
                           num_layers=3)
        self.fc = nn.Linear(self.feature_size, out_features)
        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x, (h1, _) = self.rnn(x)  # h1 shape: (num_layers, batch_size, feature_size)
        '''
        x1, input_sizes = pad_packed_sequence(x)

        last_output = torch.empty(len(input_sizes), self.feature_size, dtype=torch.float32)
        for i, seq_size in enumerate(input_sizes):
            last_output[i] = x1[seq_size-1, i]
        '''
        output = self.fc(h1[-1])  # h1[-1]: last hidden state of the second layer, out shape: (batch_size, out_features)
        # output = self.fc(last_output.to(self.device))
        prob = self.softmax(output)

        return output, prob  # x shape: (batch_size, 2), prob shape: (batch_size, 2)


class LstmEncoder(nn.Module):
    def __init__(self, input_features, embedding_dim):  # input should be (batch_size, seq_len, in_features)
        super(LstmEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = 2 * embedding_dim

        self.lstm1 = nn.LSTM(input_features, self.hidden_size, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(self.hidden_size, self.embedding_dim, batch_first=True, num_layers=1)

    def forward(self, x):
        x, (_, _) = self.lstm1(x)
        x, (h1, _) = self.lstm2(x)  # h1 of shape (1, batch_size, embedding_dim)

        return h1


class LstmDecoder(nn.Module):
    def __init__(self, input_features, output_features):  # input -> (batch_size, seq_len, embedding_dim)
        super(LstmDecoder, self).__init__()
        self.embedding_dim = input_features
        self.hidden_size = 2 * output_features
        self.output_features = output_features

        self.lstm1 = nn.LSTM(self.embedding_dim, self.embedding_dim, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True, num_layers=1)

        self.fc = nn.Linear(self.hidden_size, output_features)

    def forward(self, x, seq_len):
        x = x.squeeze(0).unsqueeze(1)  # output shape: (batch_size, 1, embedding_dim)
        x = x.repeat(1, seq_len, 1)  # output shape: (batch_size, seq_len, embedding_dim)
        x, (_, _) = self.lstm1(x)  # output shape: (batch_size, seq_len, embedding_dim)
        x, (_, _) = self.lstm2(x)  # output shape: (batch_size, seq_len, 2*output_features)
        x = self.fc(x)  # output shape: (batch_size, seq_len, output_features)

        return x


class LstmAutoEncoder(nn.Module):
    def __init__(self, features, embedding):  # input should be (batch_size, seq_len, features)
        super(LstmAutoEncoder, self).__init__()

        self.encoder = LstmEncoder(input_features=features, embedding_dim=embedding)
        self.decoder = LstmDecoder(input_features=embedding, output_features=features)

    def forward(self, x):
        seq_len = x.shape[1]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, seq_len)

        return encoded, decoded  # output shape: (batch_size, seq_len, features)


class ObjectCnn(nn.Module):
    def __init__(self):
        super(ObjectCnn, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=True)

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input):
        x = self.vgg(input)

        return x


class SceneCnn(nn.Module):
    def __init__(self):
        super(SceneCnn, self).__init__()
        arch = 'resnet50'
        self.resnet = torchvision.models.__dict__[arch](num_classes=365)

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.resnet(x)

        return output


class EmotionArm(nn.Module):
    def __init__(self):
        super(EmotionArm, self).__init__()
        self.emonet = EmoNet()
        self.emonet.eval()
        self.lstm = ModalRnn(256)  # change here for different feature size

        for params in self.emonet.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.emonet(x)
        x = x.unsqueeze(0)
        x1, prob = self.lstm(x)

        one_hot_feature = torch.zeros((x1.shape[0], 2))
        _, predictions = torch.max(x1, 1)

        for i in range(int(predictions.shape[0])):
            one_hot_feature[i][int(predictions[i])] = 1.0
            # one_hot_feature[i][2] = torch.max(prob.squeeze(0)[int(predictions[i])].data)  # FOR ADDING IN CONF

        return one_hot_feature


class EngagementArm(nn.Module):
    def __init__(self):
        super(EngagementArm, self).__init__()
        self.engage = EngagementCNN()
        self.lstm = ModalRnn(512)  # change here for different feature size

    def forward(self, x):
        _, x = self.engage(x)
        x = x.unsqueeze(0).squeeze(4).squeeze(3)
        x1, prob = self.lstm(x)

        one_hot_feature = torch.zeros((x1.shape[0], 2))
        _, predictions = torch.max(x1, 1)

        for i in range(int(predictions.shape[0])):
            one_hot_feature[i][int(predictions[i])] = 1.0
            # one_hot_feature[i][2] = torch.max(prob.squeeze(0)[int(predictions[i])].data) # FOR ADDING IN CONF

        return one_hot_feature


class ConvoCNN(nn.Module):
    def __init__(self, latent_dim):
        super(ConvoCNN, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.fc = nn.Linear(resnet.fc.in_features, latent_dim)
        self.cnn = resnet

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.cnn(x)
        return x


class ConvoArm(nn.Module):
    def __init__(self):
        super(ConvoArm, self).__init__()

        # cnn
        self.latent_dim = 512
        self.cnn = ConvoCNN(self.latent_dim)
        self.num_classes = 2

        # rnn
        self.rnn_hidden_size = 1024
        self.rnn = nn.LSTM(input_size=self.latent_dim,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=2,
                           dropout=0.3,
                           batch_first=True)

        self.output_layer = nn.Linear(self.rnn_hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.unsqueeze(0)
        x1, (h, c) = self.rnn(x)
        output = self.output_layer(h[-1])
        prob = self.softmax(output)

        one_hot_feature = torch.zeros((x1.shape[0], 2))
        _, predictions = torch.max(output, 1)

        for i in range(int(predictions.shape[0])):
            one_hot_feature[i][int(predictions[i])] = 1.0
            # one_hot_feature[i][2] = torch.max(prob.squeeze(0)[int(predictions[i])].data) # FOR ADDING IN CONF

        return one_hot_feature


class AttributeExtractor(nn.Module):  # Input: Sequence of images, Output: Attributes of sequences
    def __init__(self):
        super(AttributeExtractor, self).__init__()
        self.emotion_arm = EmotionArm()
        self.engagement_arm = EngagementArm()
        self.convo_arm = ConvoArm()

        for param in self.emotion_arm.parameters():
            param.requires_grad = False

        for param in self.engagement_arm.parameters():
            param.requires_grad = False

        for param in self.convo_arm.parameters():
            param.requires_grad = False

    def forward(self, x):
        x1 = self.emotion_arm(x)
        x2 = self.engagement_arm(x)
        x3 = self.convo_arm(x)

        output = torch.cat((x1, x2, x3))

        return output


class NeighbourGenerator(nn.Module):  # Input: Sequence of images, Output: List of neighbours
    def __init__(self, k=10):
        super(NeighbourGenerator, self).__init__()
        self.objectcnn = ObjectCnn()
        self.scenecnn = SceneCnn()
        self.encoder = LstmAutoEncoder(1365, 512)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.dblatentvec = self.get_neigh_latent()
        self.num_neigh = k

    def forward(self, x):
        x1 = self.objectcnn(x)
        x2 = self.scenecnn(x)

        z = torch.cat((x1, x2), dim=1)
        z = z.unsqueeze(0)
        l, _ = self.encoder(z)
        l = l.squeeze(0)
        diff = self.cos(l, self.dblatentvec)
        similarity = torch.sort(diff, descending=True)[0]
        similarity = torch.narrow(similarity, 0, 0, self.num_neigh)
        neigh_index = torch.sort(diff, descending=True)[1]
        neigh_index = torch.narrow(neigh_index, 0, 0, self.num_neigh)

        return neigh_index, similarity  # returns neighbour seq id indexes and similarity score

    def get_neigh_latent(self):
        latent_file = os.path.join(os.getcwd(), 'latent_vector', 'train_1275_latent_vector.pt')
        with open(latent_file, 'rb') as f:
            vector = torch.load(f, map_location=torch.device('cuda'))

        return vector


class CompetenceClassifier(nn.Module):
    def __init__(self):
        super(CompetenceClassifier, self).__init__()
        self.fc1 = nn.Linear(514, 100)
        self.fc2 = nn.Linear(100, 5)
        self.act = nn.Sigmoid()
        self.act1 = nn.ReLU()
        self.fc3 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        output = self.act(self.fc3(x))

        return output


class DynamicweightGenerator(nn.Module):
    def __init__(self, combined_dataset):
        super(DynamicweightGenerator, self).__init__()
        self.neighgen = NeighbourGenerator()
        self.neigh_vec = self.neighgen.get_neigh_latent()
        self.dataset = combined_dataset
        self.cc1 = CompetenceClassifier()
        self.cc2 = CompetenceClassifier()
        self.cc3 = CompetenceClassifier()
        self.emo_rnn = ModalRnn(256)
        self.eng_rnn = ModalRnn(512)
        self.covno_rnn = ModalRnn(512)

    def forward(self, x):
        x, _ = self.neighgen(x)
        cc_emo = None
        cc_eng = None
        cc_convo = None
        for index in x:
            emo_feature, eng_feature, convo_feature = self.dataset.__getitem__(index.item())
            emo_one_hot = self.one_hot_encode(self.emo_rnn, emo_feature.unsqueeze(0))
            eng_one_hot = self.one_hot_encode(self.eng_rnn, eng_feature.unsqueeze(0))
            convo_one_hot = self.one_hot_encode(self.covno_rnn, convo_feature.unsqueeze(0))

            latent_vec = self.neigh_vec[index.item()].unsqueeze(0).to('cpu')
            '''
            if cc_feature is None:
                cc_feature = torch.stack((torch.cat((latent_vec, emo_one_hot), dim=1),
                                          torch.cat((latent_vec, eng_one_hot), dim=1),
                                          torch.cat((latent_vec, convo_one_hot), dim=1)))
            else:
                cc_before_feature = torch.stack((torch.cat((latent_vec, emo_one_hot), dim=1),
                                                 torch.cat((latent_vec, eng_one_hot), dim=1),
                                                 torch.cat((latent_vec, convo_one_hot), dim=1)))

                cc_feature = torch.cat((cc_feature, cc_before_feature), dim=1)
            '''
            if cc_emo is None:
                cc_emo = torch.cat((latent_vec, emo_one_hot), dim=1)
            else:
                cc_emo = torch.cat((cc_emo, torch.cat((latent_vec, emo_one_hot), dim=1)), dim=0)

            if cc_eng is None:
                cc_eng = torch.cat((latent_vec, eng_one_hot), dim=1)
            else:
                cc_eng = torch.cat((cc_eng, torch.cat((latent_vec, eng_one_hot), dim=1)), dim=0)

            if cc_convo is None:
                cc_convo = torch.cat((latent_vec, convo_one_hot), dim=1)
            else:
                cc_convo = torch.cat((cc_convo, torch.cat((latent_vec, convo_one_hot), dim=1)), dim=0)

        weights = torch.stack((self.cc1(cc_emo.to('cuda')), self.cc2(cc_eng.to('cuda')), self.cc3(cc_convo.to('cuda'))), dim=0)

        return weights

    def one_hot_encode(self, rnn, feature):
        x1, prob = rnn(feature)

        one_hot_feature = torch.zeros((x1.shape[0], 2))
        _, predictions = torch.max(x1, 1)

        for i in range(int(predictions.shape[0])):
            one_hot_feature[i][int(predictions[i])] = 1.0
            # one_hot_feature[i][2] = torch.max(prob.squeeze(0)[int(predictions[i])].data) # FOR ADDING IN CONF

        return one_hot_feature


class InteractionMLP(nn.Module):
    def __init__(self):
        super(InteractionMLP, self).__init__()
        self.fc1 = nn.Linear(7, 3)
        #xavier_uniform_(self.fc1.weight)
        #self.dropout = nn.Dropout(0.2)
        #self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(5, 3)
        #kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        #self.fc3 = nn.Linear(8, 3)
        #xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.relu(x)
        # = self.dropout(x)
        #x = self.fc2(x)
        output = x

        return output


class InteractionPipeline(nn.Module):
    def __init__(self, dataset):
        super(InteractionPipeline, self).__init__()
        self.attexr = AttributeExtractor()
        self.weightgen = DynamicweightGenerator(dataset)
        self.interact = InteractionMLP()

        for params in self.attexr.parameters():
            params.requires_grad = False

        for params in self.weightgen.parameters():
            params.requires_grad = False

        ''' TO OPTIMIZE CERTAIN LAYERS
        for params in self.attexr.emotion_arm.lstm.parameters():
            params.requires_grad = True

        for params in self.attexr.engagement_arm.lstm.parameters():
            params.requires_grad = True

        for params in self.attexr.convo_arm.rnn.parameters():
            params.requires_grad = True

        for params in self.weightgen.cc1.parameters():
            params.requires_grad = True

        for params in self.weightgen.cc2.parameters():
            params.requires_grad = True

        for params in self.weightgen.cc3.parameters():
            params.requires_grad = True
        '''
        self.attexr.eval()
        self.weightgen.eval()

    def forward(self, crops, fulls):
        if crops.shape[0] > 2:
            brief = torch.tensor([0])
        else:
            brief = torch.tensor([1])

        xc = self.attexr(crops)
        xw = self.weightgen(fulls)

        xw = torch.mean(xw, dim=1).to('cpu')
        padding = torch.ones((xw.shape[0], 1))
        xw_padded = torch.cat((xw, padding), dim=1)
        final_feature = xc * xw
        final_feature = torch.flatten(final_feature)
        final_feature = torch.cat((final_feature, brief), dim=0).to("cuda")
        print("Feature:", final_feature.data)

        output = self.interact(final_feature)
        print(output)

        return output
