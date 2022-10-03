from torch import nn
import torch
from torch.nn import utils
from torch.nn import init


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.num_features = 5000
        self.n_diagnosis_codes = 3
        self.emb_size = 8
        self.emb = nn.Embedding(self.n_diagnosis_codes, self.emb_size, padding_idx=-1, max_norm=1.0)
        self.hidden_size = 16
        self.num_classes = 2
        self.num_new_features = 256
        self.fc = nn.Linear(self.num_features, self.num_new_features)
        self.batchnorm = nn.Sequential(
            nn.BatchNorm1d(self.num_new_features),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.num_new_features * self.emb_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        self.softmax = nn.Softmax(dim=1)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    def forward(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        x = self.batchnorm(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        x = self.softmax(x)
        return x

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None

    def with_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def no_grad(self):
        for param in self.parameters():
            param.requires_grad = False


class MLP_E(nn.Module):
    def __init__(self):
        super(MLP_E, self).__init__()
        self.num_features = 5000
        self.n_diagnosis_codes = 3
        self.emb_size = 8
        self.emb = nn.Embedding(self.n_diagnosis_codes, self.emb_size, padding_idx=-1, max_norm=1.0)
        self.hidden_size = 16
        self.num_classes = 2
        self.num_new_features = 256
        self.fc = nn.Linear(self.num_features, self.num_new_features)
        self.batchnorm = nn.Sequential(
            nn.BatchNorm1d(self.num_new_features),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.num_new_features * self.emb_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.softmax = nn.Softmax(dim=1)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    def forward(self, x):
        if self.attributions is not None:
            x = x * (1-self.attributions).unsqueeze(1).cuda()
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        x = self.batchnorm(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x


class MLP_Dc(nn.Module):
    def __init__(self):
        super(MLP_Dc, self).__init__()
        self.hidden_size = 16
        self.num_classes = 2
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x


class SNPDpedec(nn.Module):

  def __init__(self, encoder, num_features, num_classes=0):
    super(SNPDpedec, self).__init__()
    self.encoder = encoder
    self.num_features = num_features
    self.num_classes = num_classes

  def forward(self, *input):
    raise NotImplementedError


class SNPDFC3(SNPDpedec):

  def __init__(self, encoder, num_features=512, num_classes=0):
    super(SNPDFC3, self).__init__(
      encoder=encoder, num_features=num_features,
      num_classes=num_classes)

    self.linear1 = utils.spectral_norm(nn.Linear(num_features, num_features))
    self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    self.linear2 = utils.spectral_norm(nn.Linear(num_features, num_features))
    self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    self.l7 = utils.spectral_norm(nn.Linear(num_features, 1))
    if num_classes > 0:
      self.l_y = utils.spectral_norm(
        nn.Embedding(num_classes, num_features))

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.l7.weight.data)
    optional_l_y = getattr(self, 'l_y', None)
    if optional_l_y is not None:
      init.xavier_uniform_(optional_l_y.weight.data)

  def forward(self, x, y=None, decoder_only=False):
    if decoder_only:
      h = x
    else:
      h = self.encoder(x)
    h = self.linear1(h)
    h = self.relu1(h)
    h = self.linear2(h)
    h = self.relu2(h)

    output = self.l7(h)
    if y is not None:
      output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
    return output
