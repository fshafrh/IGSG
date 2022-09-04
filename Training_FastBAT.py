from attack.pgd_attack_restart import attack_pgd_restart
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from Training_utils import *
from attack.context import ctx_noparamgrad

# creating parser object
parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--dataset', default='pedec', type=str, help='Dataset')
parser.add_argument('--model', default='MLP', type=str, help='LSTM, MLP')
parser.add_argument('--lmbda', default=0, type=float, help="The parameter lambda for Fast-BAT.")
args = parser.parse_args()

# There are three datasets, some models have the same name for the datasets, so we just load one depending on the detaset.
if args.dataset == 'Splice':
    from Models.SpliceModels import *
elif args.dataset == 'pedec':
    from Models.pedecModels import *
elif args.dataset == 'census':
    from Models.censusModels import *


def _attack_loss(predictions, labels):
    return -torch.nn.CrossEntropyLoss(reduction='sum')(predictions, labels)


class BatTrainer:
    def __init__(self, args):
        self.args = args
        self.steps = 1
        self.eps = budgets[Dataset]
        self.attack_lr = 0.2
        if self.args.lmbda != 0.0:
            self.lmbda = self.args.lmbda
        else:
            self.lmbda = 1. / self.attack_lr

        self.constraint_type = 1
        self.log = None
        self.mode = 'fast_bat'

    def test_sa(self, model, data, labels):
        model.eval()
        with torch.no_grad():
            predictions_sa = model(data)
            correct = (torch.argmax(predictions_sa.data, 1) == labels).sum().cpu().item()
        return correct

    def train(self, model, X_Train, y_Train, opt, loss_func, device, scheduler=None, valid_mat=None):

        model.train()
        training_loss = torch.tensor([0.])
        train_sa = torch.tensor([0.])
        train_ra = torch.tensor([0.])

        total = 0

        n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
        samples = random.sample(range(n_batches), n_batches)

        # start training with randomly input batches.
        for index in samples:
            # make X into one hot vectors.
            batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
            labels = y_Train[batch_size * index: batch_size * (index + 1)].cuda()
            data = input_process(batch_diagnosis_codes, Dataset)
            real_batch = data.shape[0]
            features = data.shape[1]
            categories = data.shape[2]
            total += real_batch

            # Record SA along with each batch
            train_sa += self.test_sa(model, data, labels)

            model.train()

            if self.mode == "fast_bat":
                z_init = torch.clamp(
                    data + torch.FloatTensor(data.shape).uniform_(-self.eps / 20, self.eps / 20).to(device),
                    min=0, max=1
                ) - data
                z_init = z_init * valid_mat
                z_init.requires_grad_(True)

                model.clear_grad()
                model.with_grad()
                attack_loss = _attack_loss(model(data + z_init), labels)
                grad_attack_loss_delta = torch.autograd.grad(attack_loss, z_init, retain_graph=True, create_graph=True)[
                    0]
                delta = z_init - self.attack_lr * grad_attack_loss_delta
                delta = torch.clamp(data + delta, min=0, max=1) - data
                delta = delta * valid_mat
                delta_norms = abs(delta.data).sum([1, 2], keepdim=True)
                delta.data = self.eps * delta.data / torch.max(self.eps * torch.ones_like(delta_norms), delta_norms)

                delta = delta.detach().requires_grad_(True)
                attack_loss_second = _attack_loss(model(data + delta), labels)
                grad_attack_loss_delta_second = \
                    torch.autograd.grad(attack_loss_second, delta, retain_graph=True, create_graph=True)[0] \
                        .view(real_batch, 1, features * categories)
                delta_star = delta - self.attack_lr * grad_attack_loss_delta_second.detach().view(data.shape)
                delta_star = torch.clamp(data + delta_star, min=0, max=1) - data
                delta_star = delta_star * valid_mat
                delta_norms = abs(delta_star.data).sum([1, 2], keepdim=True)
                delta_star.data = self.eps * delta_star.data / torch.max(self.eps * torch.ones_like(delta_norms),
                                                                         delta_norms)
                z = delta_star.clone().detach().view(real_batch, -1)

                # H: (batch, feature * categories)
                z_min = torch.max(-data.view(real_batch, -1),
                                  -self.eps * 0.03 * torch.ones_like(data.view(real_batch, -1)))
                z_max = torch.min(1 - data.view(real_batch, -1),
                                  self.eps * 0.03 * torch.ones_like(data.view(real_batch, -1)))
                H = ((z > z_min + 1e-7) & (z < z_max - 1e-7)).to(torch.float32)

                delta_cur = delta_star.detach().requires_grad_(True)

                model.no_grad()
                lgt = model(data + delta_cur)
                delta_star_loss = loss_func(lgt, labels)
                delta_star_loss.backward()
                delta_outer_grad = delta_cur.grad.view(real_batch, -1)

                hessian_inv_prod = delta_outer_grad / self.lmbda
                bU = (H * hessian_inv_prod).unsqueeze(-1)

                model.with_grad()
                model.clear_grad()
                b_dot_product = grad_attack_loss_delta_second.bmm(bU).view(-1).sum(dim=0)
                b_dot_product.backward()
                cross_term = [-param.grad / real_batch for param in model.parameters()]

                model.clear_grad()
                model.with_grad()
                predictions = model(data + delta_star)
                train_loss = loss_func(predictions, labels) / real_batch
                opt.zero_grad()
                train_loss.backward()

                with torch.no_grad():
                    for p, cross in zip(model.parameters(), cross_term):
                        new_grad = p.grad + cross
                        p.grad.copy_(new_grad)

                del cross_term, H, grad_attack_loss_delta_second
                opt.step()

            else:
                raise NotImplementedError()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == labels
                if self.log is not None:
                    self.log(model,
                             loss=train_loss.cpu(),
                             accuracy=correct.cpu(),
                             learning_rate=scheduler.get_last_lr()[0],
                             batch_size=real_batch)
            if scheduler:
                scheduler.step()

            training_loss += train_loss.cpu().sum().item()
            train_ra += correct.cpu().sum().item()
        return model, training_loss

    def eval(self, model, X, y, attack_eps, attack_steps, attack_lr, attack_rs, device):
        total = 0
        robust_total = 0
        correct_total = 0
        test_loss = 0

        n_batches = int(np.ceil(float(len(X)) / float(batch_size)))
        samples = random.sample(range(n_batches), n_batches)

        # start training with randomly input batches.
        for index in samples:
            # make X into one hot vectors.
            batch_diagnosis_codes = X[batch_size * index: batch_size * (index + 1)]
            labels = y[batch_size * index: batch_size * (index + 1)].cuda()
            data = input_process(batch_diagnosis_codes, Dataset)
            real_batch = data.shape[0]
            total += real_batch

            with ctx_noparamgrad(model):
                perturbed_data = attack_pgd_restart(
                    model=model,
                    X=data,
                    y=labels,
                    eps=attack_eps,
                    alpha=attack_lr,
                    attack_iters=attack_steps,
                    n_restarts=attack_rs,
                    rs=(attack_rs > 1),
                    verbose=False,
                    linf_proj=False,
                    l2_proj=False,
                    l2_grad_update=False,
                    l1_proj=True,
                    cuda=True
                ) + data

            if attack_steps == 0:
                perturbed_data = data

            predictions = model(data)
            correct = torch.argmax(predictions, 1) == labels
            correct_total += correct.sum().cpu().item()

            predictions = model(perturbed_data)
            robust = torch.argmax(predictions, 1) == labels
            robust_total += robust.sum().cpu().item()

            robust_loss = torch.nn.CrossEntropyLoss()(predictions, labels)
            test_loss += robust_loss.cpu().sum().item()

            if self.log:
                self.log(model=model,
                         accuracy=correct.cpu(),
                         robustness=robust.cpu(),
                         batch_size=real_batch)

        return correct_total, robust_total, total, test_loss / total


def Training(Dataset, batch_size, n_epoch, lr):
    device = torch.device("cuda")
    # devide the dataset into train, validation and test
    train_idx, val_idx, test_idx = dataset_split(Dataset)

    X, y = preparation(Dataset)
    y_Train = y[train_idx]
    X_Train = X[train_idx]
    X_Validation = X[val_idx]
    y_Validation = y[val_idx]

    print(X_Train.shape, X_Validation.shape)
    y_Test = y[test_idx]
    X_Test = X[test_idx]

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

    model = MLP()

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[milestones[Dataset]],
                                                     gamma=0.1)
    print('done!', file=log_f, flush=True)
    # define cross entropy loss function
    train_loss = nn.CrossEntropyLoss(reduction="sum")

    if Dataset in complex_categories.keys():
        valid_mat = model.valid_matrix().cuda()
    else:
        valid_mat = torch.ones(num_feature[Dataset], num_category[Dataset]).cuda()
        valid_mat[:, -1] = 0

    print('training start', file=log_f, flush=True)
    trainer = BatTrainer(args=args)
    model.train()

    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    epoch_duaration = 0.0
    best_epoch = 0.0

    for epoch in range(n_epoch):
        start_time = time.time()
        model.train()
        model, train_cost = trainer.train(model=model,
                                          X_Train=X_Train,
                                          y_Train=y_Train,
                                          opt=optimizer,
                                          loss_func=train_loss,
                                          scheduler=scheduler,
                                          device=device,
                                          valid_mat=valid_mat)

        correct_total, robust_total, total, validate_cost = trainer.eval(model=model,
                                                                         X=X_Validation,
                                                                         y=y_Validation,
                                                                         attack_eps=budgets[Dataset],
                                                                         attack_steps=30,
                                                                         attack_lr=0.1,
                                                                         attack_rs=10,
                                                                         device=device)

        duration = time.time() - start_time
        epoch_duaration += duration

        print(validate_cost)

        # if the current validation cost is smaller than the current best one, then save the model
        if validate_cost < best_validate_cost:
            # torch.save(rnn.state_dict(), output_file + Dataset + Model_Name + '.' + str(epoch))
            torch.save(model.state_dict(), output_file + Dataset + Model_Name + '.' + str(epoch),
                       _use_new_zipfile_serialization=False)
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, validate_cost, duration), file=log_f, flush=True)

        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_epoch = epoch

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
    n_batches_IG = int(np.ceil(float(len(X_Test)) / float(512)))
    if Dataset_type[Dataset] == 'multi':
        invalid_sample = torch.tensor([num_category[Dataset] - 1] * num_feature[Dataset]).unsqueeze(0)
        invalid_sample = one_hot_samples(invalid_sample, Dataset)[0].cuda()
    elif Dataset_type[Dataset] == 'binary':
        invalid_sample = torch.tensor([0] * num_feature[Dataset]).cuda()
    else:
        invalid_sample = None
    best_attributions, _ = IntegratedGradient(X_Test, y_Test, Dataset, n_batches_IG, invalid_sample, model)
    total_variation_loss = torch.norm(best_attributions[:-1] - best_attributions[1:], p=1)
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

    print(best_attributions, file=log_a, flush=True)
    print('total variation loss:', total_variation_loss, file=log_a, flush=True)

    print('Best validation cost:', best_validate_cost, file=log_a, flush=True)


milestones = {'Splice': 2000, 'pedec': 200, 'census': 100}

lr = args.lr
Dataset = args.dataset

Model_Name = args.model + '_FastBAT'

batch_size = batch_sizes[Dataset]
n_epoch = epochs[Dataset]

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

Training(Dataset, batch_size, n_epoch, lr)
print(args)
