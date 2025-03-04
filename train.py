import argparse
from torch.utils.data import Dataset
import torch.optim as optim
from torch.autograd import Variable
from HMDT_Net_model import HMDTF as mymodel
from utils import *
from utils import LoadDataset, load_folds_data
from torchcrf import CRF
import time
import random
from loss_function import MySpl_loss
from tensorboardX import SummaryWriter
from Contrastive_loss import NTXentLoss_poly

logger = SummaryWriter(log_dir="plot/logs")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

seed = 1029
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(int(seed))


def preliminary_result(model, train_files):
    train_preliminary = result_save()
    model.eval()
    for file in train_files:
        data = LoadDataset([file])
        label_y = data.y_data.reshape((1, -1)).type(
            torch.FloatTensor).cpu().detach().numpy()
        outputs, _, _, _, _ = model(Variable(data.x_data.cuda()))
        outputs = outputs.reshape((1, -1, 5)).cpu().detach().numpy()
        train_preliminary.add_preliminary_result(file, outputs, label_y)
    return train_preliminary


def train_CRF(model, train_preliminary):
    model.eval()
    model_crf = CRF(5, batch_first=True).cuda()
    optimizer = optim.Adam(model_crf.parameters(), lr=0.05, betas=(0.9, 0.99))
    for epoch in range(1):
        losses = []
        model_crf.train()
        for i in train_preliminary.preliminary_result.keys():
            preliminary_result = train_preliminary.preliminary_result[i]
            label_y = train_preliminary.label_y[i]
            preliminary_result = torch.Tensor(preliminary_result)
            label_y = torch.Tensor(label_y)
            loss = (-1) * model_crf(Variable(preliminary_result.cuda(), requires_grad=True),
                                    Variable(label_y.cuda(), requires_grad=True).long())
            loss.backward()
            optimizer.step()
            losses.append(loss.tolist())
            loss_mean = sum(losses) / len(losses)
        print("mean:", sum(losses) / len(losses))
    return model_crf, loss_mean


def test(model, model_crf, val_files):
    val_result = result_save()
    model.eval()
    for file in val_files:
        data = LoadDataset([file])
        label_y = data.y_data.reshape((1, -1)).cpu().detach().numpy()
        output1, _, _, _, _ = model(Variable(data.x_data.cuda()))
        output2 = model_crf.decode(output1.reshape((1, -1, 5)))
        _, preliminary = torch.max(output1.data, 1)
        val_result.add_all(file, preliminary.cpu().detach().numpy(), np.array(output2[0]), label_y[0])

    return val_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMDTNet')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--fold_num', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_ckpt', type=bool, default=True)
    parser.add_argument('--np_data_dir', type=str,
                        default="./")
    parser.add_argument('--output', type=str, default="./")
    args = parser.parse_args()

    fold_data = load_folds_data(args.np_data_dir, args.fold_num)

    for fold in range(args.fold_num):
        train_file, val_file = fold_data.get_file(fold)
        train_dataset = LoadDataset(train_file)
        val_dataset = LoadDataset(val_file)

        data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  drop_last=False,
                                                  num_workers=0, worker_init_fn=_init_fn)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=0, worker_init_fn=_init_fn)

        data_count = calc_count(train_dataset, val_dataset)

        weights_for_each_class = calc_class_weight(data_count)

        model = mymodel().cuda()
        device = torch.device('cuda')

        lr = 0.0001
        criterion = weighted_CrossEntropyLoss
        criterion1 = NTXentLoss_poly(device=device, batch_size=128, temperature=0.2,
                                     use_cosine_similarity=True)
        domain_loss = nn.CrossEntropyLoss()
        spl = MySpl_loss(n_samples=len(data_loader.dataset), batch_size=128, spl_lambda=1.2, spl_gamma=1.1)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001, amsgrad=True)
        acc_best = 0
        trian_loss = 0
        for epoch in range(args.epochs):
            model.train()
            tims = time.time()
            correct = 0
            total = 0
            class_correct = list(0. for i in range(5))
            class_total = list(0. for i in range(5))
            item = 0

            for i, (sdata, tdata) in enumerate(zip(data_loader, val_loader)):

                src_data, src_target = sdata
                tar_data, _ = tdata

                src_data = src_data.type(torch.FloatTensor)
                src_target = src_target.type(torch.FloatTensor)
                tar_data = tar_data.type(torch.FloatTensor)

                src_data = Variable(src_data.cuda())
                src_target = Variable(src_target.cuda())
                tar_data = Variable(tar_data.cuda())

                optimizer.zero_grad()
                outputs_src, t, p, src_domains, src_att = model(src_data)
                outputs_tar, _, _, tar_domains, tar_att = model(tar_data)

                loss_class_src = criterion(outputs_src, src_target.long().cuda(), weights_for_each_class)
                loss_con = criterion1(t, p)
                loss_src_domain = domain_loss(src_domains, torch.ones(len(src_target)).long().cuda())
                loss_tar_domain = domain_loss(tar_domains, torch.zeros(len(tar_data)).long().cuda())
                yu_loss = (loss_tar_domain + loss_src_domain) / 2
                alpha = 0.1
                beta = 0.1
                loss_all = loss_class_src + alpha * loss_con + beta * yu_loss
                loss_all = spl(i,loss_all)
                loss_all.backward()
                optimizer.step()
                spl.increase_threshold()

            train_result = preliminary_result(model, train_file)
            model_crf, mean_loss = train_CRF(model, train_result)

            logger.add_scalar('train_loss', mean_loss, epoch)

            val_result = test(model, model_crf, val_file)
            acc, mf1,f1, k, recall, pre = cal_metric(fold, epoch, val_result)


            logger.add_scalar('acc', acc, epoch)
            logger.add_scalar('mf1', mf1, epoch)
            logger.add_scalar('f1', f1, epoch)
            logger.add_scalar('k', k, epoch)
            logger.add_scalar('recall', recall, epoch)
            logger.add_scalar('pre', pre, epoch)
            logger.add_scalar('loss_all', loss_all, epoch)
            logger.close()
            if acc_best < acc:
                acc_best = acc
                print("[Epoch: %d] acc：%.4f, best_acc: %.4f" % (epoch, acc, acc_best))

                if args.save_ckpt:
                    model_file = str(fold + 1) + "_" + str(epoch + 1) + "_20.pth"
                    torch.save(model, os.path.join(args.output, "fold{}_model.pth".format(fold)))
                    torch.save(model_crf, os.path.join(args.output, "fold{}_crf.pth".format(fold)))

            if epoch == 10:
                lr /= 10
                print('reset learning rate to:', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print(param_group['lr'])

            torch.cuda.empty_cache()





