from torch import nn
import torch
from torch.nn import utils
from torch.nn import init

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.num_features = 41
        self.num_cate = 52
        self.embedding = nn.Embedding(
            num_embeddings=self.num_features * self.num_cate,
            embedding_dim=64,
            max_norm=1.0
        )
        empty_list = self.empty_emb()
        self.embedding.weight.data[empty_list] = 0
        self.hidden_size = 64
        self.num_classes = 2
        self.mlp = nn.Sequential(
            nn.Linear(self.num_features * self.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        self.softmax = nn.Softmax(dim=1)
        self.model_input = torch.LongTensor(range(self.num_features * self.num_cate))

    def forward(self, x):
        model_input = self.model_input.reshape(1, -1).cuda()
        weight = self.embedding(model_input)
        x = x.flatten(start_dim=1, end_dim=2)
        x = torch.unsqueeze(x, dim=2)
        x = x * weight
        x = x.reshape(x.size(0), self.num_features, self.num_cate, -1)
        x = torch.sum(x, dim=2)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        x = self.softmax(x)
        return x

    def empty_emb(self):
        census_category = [5, 9, 4, 4, 17, 4, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 4, 4, 4, 6, 6,
                           51, 38, 8, 3, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 4, 2]
        empty_list = []
        for i in range(len(census_category)):
            for j in range(52):
                if j >= census_category[i]:
                    empty_list.append(i * 52 + j)
        return empty_list

    def valid_matrix(self):
        census_category = [5, 9, 4, 4, 17, 4, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 4, 4, 4, 6, 6,
                           51, 38, 8, 3, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 4, 2]
        valid_list = []
        for i in range(len(census_category)):
            for j in range(52):
                if j >= census_category[i]:
                    valid_list.append(0)
                else:
                    valid_list.append(1)
        valid_mat = torch.LongTensor(valid_list).reshape(self.num_features, self.num_cate)
        return valid_mat

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
        self.num_features = 41
        self.num_cate = 52
        self.embedding = nn.Embedding(
            num_embeddings=self.num_features * self.num_cate,
            embedding_dim=64,
            max_norm=1.0
        )
        empty_list = self.empty_emb()
        self.embedding.weight.data[empty_list] = 0
        self.hidden_size = 64
        self.num_classes = 2
        self.mlp = nn.Sequential(
            nn.Linear(self.num_features * self.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.model_input = torch.LongTensor(range(self.num_features * self.num_cate))

    def forward(self, x):
        model_input = self.model_input.reshape(1, -1).cuda()
        weight = self.embedding(model_input)
        x = x.flatten(start_dim=1, end_dim=2)
        x = torch.unsqueeze(x, dim=2)
        x = x * weight
        x = x.reshape(x.size(0), self.num_features, self.num_cate, -1)
        x = torch.sum(x, dim=2)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x

    def empty_emb(self):
        census_category = [5, 9, 4, 4, 17, 4, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 4, 4, 4, 6, 6,
                           51, 38, 8, 3, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 4, 2]
        empty_list = []
        for i in range(len(census_category)):
            for j in range(52):
                if j >= census_category[i]:
                    empty_list.append(i * 52 + j)
        return empty_list


class MLP_Dc(nn.Module):
    def __init__(self):
        super(MLP_Dc, self).__init__()
        self.hidden_size = 64
        self.num_classes = 2
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x


class SNPDcensus(nn.Module):

  def __init__(self, encoder, num_features, num_classes=0):
    super(SNPDcensus, self).__init__()
    self.encoder = encoder
    self.num_features = num_features
    self.num_classes = num_classes

  def forward(self, *input):
    raise NotImplementedError


class SNPDFC3(SNPDcensus):

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

