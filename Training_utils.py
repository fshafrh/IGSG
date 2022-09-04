from utils import *
import torch.nn.functional as F
import random
import time

def smooth_sampling(batch_data, Dataset, n=3):
    p_change = budgets[Dataset] / batch_data.size(1)
    batch_data_perturbed = torch.tensor([])
    for i in range(n):
        valid = torch.LongTensor(batch_data.shape).bernoulli_(p_change)
        if Dataset_type[Dataset] == 'binary':
            batch_data_perturbed_t = batch_data * (1 - valid) + valid * (1 - batch_data)
        elif Dataset not in complex_categories.keys():
            changed_to = np.argmax(np.random.multinomial(1, [1 / num_avail_category[Dataset]] *
                                                         num_avail_category[Dataset], size=batch_data.shape), axis=2)
            changed_to = torch.tensor(changed_to)
            batch_data_perturbed_t = batch_data * (1 - valid) + valid * changed_to
        else:
            changed_to = torch.tensor([])
            for j in range(batch_data.size(1)):
                changed_to_t = np.argmax(np.random.multinomial(1, [1 / complex_categories[Dataset][j]] *
                                                               complex_categories[Dataset][j],
                                                               size=(batch_data.size(0), 1)), axis=2)
                changed_to_t = torch.tensor(changed_to_t)
                changed_to = torch.cat((changed_to, changed_to_t), dim=1)
            batch_data_perturbed_t = batch_data * (1 - valid) + valid * changed_to
        batch_data_perturbed = torch.cat((batch_data_perturbed, batch_data_perturbed_t), dim=0).long()
    return batch_data_perturbed


def IntegratedGradient(X_Train, y_Train, Dataset, n_batches_IG, invalid_sample, model):
    IG_matrix_all = torch.tensor(0)
    for k in range(32):
        IG_matrix = None
        for index in range(n_batches_IG):
            batch_diagnosis_codes = X_Train[512 * index: 512 * (index + 1)]
            t_labels = y_Train[512 * index: 512 * (index + 1)].cuda()
            t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)

            temp_codes = invalid_sample + (t_diagnosis_codes - invalid_sample) * (k / 32)
            temp_codes = torch.autograd.Variable(temp_codes.data, requires_grad=True)
            logit = model(temp_codes)
            loss = F.cross_entropy(logit, t_labels)
            loss.backward()
            temp_grad = temp_codes.grad.cpu().data
            if Dataset_type[Dataset] == 'multi':
                temp_IG_matrix = torch.sum(temp_grad * t_diagnosis_codes.cpu(), dim=2)
            elif Dataset_type[Dataset] == 'binary':
                temp_IG_matrix = temp_grad * t_diagnosis_codes.cpu()
            else:
                temp_IG_matrix = None
            if index == 0:
                IG_matrix = temp_IG_matrix
            else:
                IG_matrix = torch.cat((IG_matrix, temp_IG_matrix), dim=0)

        IG_matrix_all = IG_matrix + IG_matrix_all
    IG_matrix_all = IG_matrix_all / -32
    attributions_all = F.softmax(IG_matrix_all, dim=1)
    attributions = torch.mean(attributions_all, dim=0)
    return attributions, IG_matrix_all


# load the dataset
def preparation(dataset):
    x = pickle.load(open('./dataset/' + dataset + 'X.pickle', 'rb'))
    y = pickle.load(open('./dataset/' + dataset + 'Y.pickle', 'rb'))
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y


seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
betas = {'Splice': 1, 'pedec': 50, 'census': 1}
alphas = {'Splice': 0.3, 'pedec': 0.05, 'census': 10}
gammas = {'Splice': 1, 'pedec': 1, 'census': 0.1}
deltas = {'Splice': 1, 'pedec': 0.05, 'census': 1}
epsilons = {'Splice': 0.0001, 'pedec': 0.00000001, 'census': 0.001}
epochs = {'Splice': 20, 'pedec': 400, 'census': 200}
batch_sizes = {'Splice': 64, 'pedec': 128, 'census': 256}
