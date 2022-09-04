from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from Training_utils import *

# creating parser object
parser = argparse.ArgumentParser(description='gene')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--dataset', default='census', type=str, help='Dataset')
parser.add_argument('--model', default='MLP', type=str, help='LSTM, MLP')
args = parser.parse_args()

# There are three datasets, some models have the same name for the datasets, so we just load one depending on the detaset.
if args.dataset == 'Splice':
    from Models.SpliceModels import *
elif args.dataset == 'pedec':
    from Models.pedecModels import *
elif args.dataset == 'census':
    from Models.censusModels import *

# used to calculate the loss of the validation set
def calculate_cost(model, X, y, batch_size, valid_mat=None):
    n_batches = int(np.ceil(float(len(X)) / float(batch_size)))
    cost_sum = 0.0
    for index in range(n_batches):
        batch_diagnosis_codes = X[batch_size * index: batch_size * (index + 1)]
        batch_diagnosis_codes_sm = smooth_sampling(batch_diagnosis_codes, Dataset, n=sm_num)
        t_labels = y[batch_size * index: batch_size * (index + 1)].cuda()
        t_labels_sm = y[batch_size * index: batch_size * (index + 1)].repeat(sm_num).cuda()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
        t_diagnosis_codes_sm = input_process(batch_diagnosis_codes_sm, Dataset)
        t_diagnosis_codes_sm = torch.autograd.Variable(t_diagnosis_codes_sm.data, requires_grad=True)

        logit = model(t_diagnosis_codes_sm)
        loss = F.cross_entropy(logit, t_labels_sm)
        loss.backward()
        grad_sm = t_diagnosis_codes_sm.grad
        attributions_temp = torch.norm(grad_sm, p=2, dim=[0, 2])
        if Dataset_type[Dataset] == 'multi':
            grad_orig_cate = torch.sum(grad_sm * t_diagnosis_codes_sm, dim=2, keepdim=True).repeat(1, 1,
                                                                                             num_category[Dataset])
            grad_orig_cate = grad_orig_cate * valid_mat
            grad_sm = grad_sm - grad_orig_cate

        grad_norm = torch.norm(grad_sm) / np.sqrt(sm_num)

        logit = model(t_diagnosis_codes)
        clean_loss = F.cross_entropy(logit, t_labels)
        grad_loss = grad_norm * beta
        attr_loss = torch.norm(attributions_temp[:-1]-attributions_temp[1:], p=1) * delta
        loss = clean_loss + grad_loss + attr_loss
        print(clean_loss.item(), grad_loss.item(), attr_loss.item())
        cost_sum += loss.cpu().data.numpy()
    return cost_sum / n_batches


def Training(Dataset, batch_size, n_epoch, lr):
    # devide the dataset into train, validation and test
    train_idx, val_idx, test_idx = dataset_split(Dataset)

    X, y = preparation(Dataset)

    y_Train = y[train_idx]
    X_Train = X[train_idx]
    y_Test = y[test_idx]
    X_Test = X[test_idx]
    X_Validation = X[val_idx]
    y_Validation = y[val_idx]

    output_file = './outputs/' + Dataset + '/' + Model_Name + '/' + str(lr) + '/'
    make_dir('./outputs/')
    make_dir('./outputs/' + Dataset + '/')
    make_dir('./outputs/' + Dataset + '/' + Model_Name + '/')
    make_dir('./outputs/' + Dataset + '/' + Model_Name + '/' + str(lr) + '/')
    make_dir('./Logs/')
    make_dir('./Logs/' + Dataset)
    make_dir('./Logs/' + Dataset + '/training/')
    make_dir('./Logs/' + Dataset + '/training/details/')
    make_dir('./Logs/' + Dataset + '/training/attack/')

    log_f = open(
        './Logs/' + Dataset + '/training/details/TEST_%s_%s.bak' % (
            Model_Name, lr), 'w+')
    print('constructing the optimizer ...', file=log_f, flush=True)

    if args.model == 'MLP':
        model = MLP()
    else:
        model = None
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    print('done!', file=log_f, flush=True)
    # define cross entropy loss function
    CEloss = torch.nn.CrossEntropyLoss().cuda()
    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    print('training start', file=log_f, flush=True)
    model.train()

    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    epoch_duaration = 0.0
    best_epoch = 0.0

    if Dataset in complex_categories.keys():
        valid_mat = model.valid_matrix().cuda()
    else:
        valid_mat = torch.ones(num_feature[Dataset], num_category[Dataset]).cuda()
        valid_mat[:, -1] = 0

    for epoch in range(n_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)

        clean_loss_sum = 0
        grad_loss_sum = 0
        attr_loss_sum = 0
        for index in samples:
            # make X into one hot vectors.
            batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
            batch_diagnosis_codes_sm = smooth_sampling(batch_diagnosis_codes, Dataset, sm_num)
            t_labels = y_Train[batch_size * index: batch_size * (index + 1)].cuda()
            t_labels_sm = y_Train[batch_size * index: batch_size * (index + 1)].repeat(sm_num).cuda()
            t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
            t_diagnosis_codes_sm = input_process(batch_diagnosis_codes_sm, Dataset)
            t_diagnosis_codes_sm = torch.autograd.Variable(t_diagnosis_codes_sm.data, requires_grad=True)

            logit = model(t_diagnosis_codes_sm)
            loss = CEloss(logit, t_labels_sm)
            loss.backward()
            grad_sm = t_diagnosis_codes_sm.grad
            attributions_temp = torch.norm(grad_sm, p=2, dim=[0, 2])

            if Dataset_type[Dataset] == 'multi':
                grad_orig_cate = torch.sum(grad_sm * t_diagnosis_codes_sm, dim=2, keepdim=True).repeat(1, 1, num_category[Dataset])
                grad_orig_cate = grad_orig_cate * valid_mat
                grad_sm = grad_sm - grad_orig_cate

            grad_norm = torch.norm(grad_sm) / np.sqrt(sm_num)

            optimizer.zero_grad()

            logit = model(t_diagnosis_codes)
            clean_loss = CEloss(logit, t_labels)
            grad_loss = grad_norm * beta
            attr_loss = torch.norm(attributions_temp[:-1]-attributions_temp[1:], p=1) * delta
            loss = clean_loss + grad_loss + attr_loss
            clean_loss_sum += clean_loss.item()
            grad_loss_sum += grad_loss.item()
            attr_loss_sum += attr_loss.item()
            loss.backward()

            optimizer.step()
            cost_vector.append(loss.cpu().data.numpy())

            iteration += 1
        print()
        print('epoch:', epoch, 'clean_loss:', clean_loss_sum, 'grad_loss:', grad_loss_sum, 'attr_loss:', attr_loss_sum)

        duration = time.time() - start_time
        train_cost = np.mean(cost_vector)
        validate_cost = calculate_cost(model, X_Validation, y_Validation, batch_size, valid_mat)

        epoch_duaration += duration

        # if the current validation cost is smaller than the current best one, then save the model
        if validate_cost < best_validate_cost:
            # torch.save(rnn.state_dict(), output_file + Dataset + Model_Name + '.' + str(epoch))
            torch.save(model.state_dict(), output_file + Dataset + Model_Name + '.' + str(epoch),
                       _use_new_zipfile_serialization=False)
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_epoch = epoch
        print('epoch:%d, mean_cost:%f, valid_cost:%f, duration:%f'
              % (epoch, np.mean(cost_vector), validate_cost, duration), file=log_f, flush=True)

        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f' % (best_epoch, best_train_cost, best_validate_cost)
        print(buf, file=log_f, flush=True)
        print()

    # test

    print('-----------test--------------', file=log_f, flush=True)
    best_parameters_file = output_file + Dataset + Model_Name + '.' + str(best_epoch)

    print(best_parameters_file)
    model.load_state_dict(torch.load(best_parameters_file))
    torch.save(model.state_dict(), './classifier/' + Dataset + '_' + Model_Name + '.par',
               _use_new_zipfile_serialization=False)
    model.eval()
    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])

    # test for the training set
    for index in range(n_batches):  # n_batches

        batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
        t_labels = y_Train[batch_size * index: batch_size * (index + 1)].numpy()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)

        logit = model(t_diagnosis_codes)
        prediction = torch.max(logit, 1)[1].view((len(t_labels),)).data.cpu().numpy()
        y_true = np.concatenate((y_true, t_labels))
        y_pred = np.concatenate((y_pred, prediction))

    accuary = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Training data')
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1))

    log_a = open(
        './Logs/' + Dataset + '/training/TEST____%s_Adam_%s.bak' % (Model_Name, lr), 'w+')
    print(best_parameters_file, file=log_a, flush=True)
    print('Training data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1), file=log_a, flush=True)

    # test for test set
    y_true = np.array([])
    y_pred = np.array([])
    n_batches_test = int(np.ceil(float(len(X_Test)) / float(batch_size)))
    for index in range(n_batches_test):  # n_batches

        batch_diagnosis_codes = X_Test[batch_size * index: batch_size * (index + 1)]
        t_labels = y_Test[batch_size * index: batch_size * (index + 1)].numpy()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)

        logit = model(t_diagnosis_codes)
        prediction = torch.max(logit, 1)[1].view((len(t_labels),)).data.cpu().numpy()
        y_true = np.concatenate((y_true, t_labels))
        y_pred = np.concatenate((y_pred, prediction))

    accuary = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Testing data')
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1))

    print('Testing data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1), file=log_a, flush=True)

    print('Best validation cost:', best_validate_cost, file=log_a, flush=True)


lr = args.lr
Dataset = args.dataset
sm_num = 5

Model_Name = args.model + '_SGSG'

batch_size = batch_sizes[Dataset]
n_epoch = epochs[Dataset]
beta = betas[Dataset]
delta = deltas[Dataset]

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

Training(Dataset, batch_size, n_epoch, lr)
print(args)
