import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_Net(nn.Module):
    def __init__(self, drop_out=0.5, channel_num=62):
        super(Encoder_Net, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64), bias=True),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(drop_out)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(channel_num, 1), bias=True),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(drop_out),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = x.permute(0, 2, 1, 3)
        return x

class EmotionEncoder_Net(nn.Module):
    def __init__(self, drop_out=0.5, token_dim=128):
        super(EmotionEncoder_Net, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1),  bias=True),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 32),  bias=True),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(4, 1), bias=True),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
        )

        self.block_2 = nn.Linear(2132, token_dim)

    def forward(self, x):
        x = self.block_1(x)
        x = x.view(x.size(0), 1, -1)
        x = self.block_2(x)
        return x

class TaskEncoder_Net(nn.Module):
    def __init__(self, drop_out=0.5, token_dim=128):
        super(TaskEncoder_Net, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1),  bias=True),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 16), bias=True),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(4, 1), bias=True),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.AvgPool2d((1, 8))
        )

        self.block_2 = nn.Linear(2236, token_dim)

    def forward(self, x):
        x = self.block_1(x)
        x = x.view(x.size(0), 1, -1)
        x = self.block_2(x)
        return x

class Task_cls(nn.Module):
    def __init__(self, token_dim, n_cls):
        super(Task_cls, self).__init__()

        self.block_1 = nn.Linear(token_dim, n_cls)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.block_1(x)
        return x

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

def estimate_JSD_MI(joint, marginal, mean=False):
    joint = (torch.log(torch.tensor(2.0)) - F.softplus(-joint))
    marginal = (F.softplus(-marginal)+marginal - torch.log(torch.tensor(2.0)))

    out = joint - marginal
    if mean:
        out = out.mean()
    return out

class MINE(nn.Module):
    def __init__(self, token_dim, GRL=False):
        super(MINE, self).__init__()
        self.fc1_x = nn.Linear(token_dim, 64)
        self.bn1_x = nn.BatchNorm1d(64)

        self.fc1_y = nn.Linear(token_dim, 64)
        self.bn1_y = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(128, 16)
        self.bn2 = nn.BatchNorm1d(16)

        self.fc3 = nn.Linear(16, 1)
        self.GRL = GRL
    def forward(self, x, y, lambd=1):

        x, y = x.view(x.size(0), -1), y.view(y.size(0), -1)

        # GRL
        if self.GRL == True:
            x = GradReverse.grad_reverse(x, lambd)
            y = GradReverse.grad_reverse(y, lambd)

        x = F.dropout(self.bn1_x(self.fc1_x(x)))
        y = F.dropout(self.bn1_y(self.fc1_y(y)))

        h = F.elu(torch.cat((x,y), dim=-1))
        h = F.elu(self.bn2(self.fc2(h)))
        h = self.fc3(h)
        return h

class model_1(nn.Module):
    def __init__(self, token_dim, out_put):
        super(model_1, self).__init__()
        self.encoder = Encoder_Net(drop_out=0.5)
        self.task_encoder = TaskEncoder_Net(drop_out=0.2, token_dim=token_dim)
        self.task_cls = Task_cls(token_dim=token_dim, n_cls=2)

        self.out_put = out_put
    def forward(self, x):
        enc = self.encoder(x)
        task_enc = self.task_encoder(enc)
        pred_task = self.task_cls(task_enc)

        if self.out_put == 'all':
            return pred_task, task_enc
        if self.out_put == 'pred':
            return pred_task


class model_2(nn.Module):
    def __init__(self, token_dim, out_put):
        super(model_2, self).__init__()
        self.encoder = Encoder_Net(drop_out=0.5)
        self.emo_encoder = EmotionEncoder_Net(drop_out=0.2, token_dim=token_dim)
        self.task_encoder = TaskEncoder_Net(drop_out=0.2, token_dim=token_dim)
        self.task_cls = Task_cls(token_dim=token_dim, n_cls=2)

        self.out_put = out_put
    def forward(self, x):
        enc = self.encoder(x)
        emo_enc = self.emo_encoder(enc)
        task_enc = self.task_encoder(enc)
        pred_task = self.task_cls(task_enc)

        if self.out_put == 'all':
            return pred_task, emo_enc, task_enc
        if self.out_put == 'pred':
            return pred_task
        if self.out_put == 'emo':
            return  emo_enc
        if self.out_put == 'task':
            return task_enc


class model_3(nn.Module):
    def __init__(self, token_dim, out_put, device, GRL=True):
        super(model_3, self).__init__()
        self.encoder = Encoder_Net(drop_out=0.5)
        self.emo_encoder = EmotionEncoder_Net(drop_out=0.2, token_dim=token_dim)
        self.task_encoder = TaskEncoder_Net(drop_out=0.2, token_dim=token_dim)
        self.task_cls = Task_cls(token_dim=token_dim, n_cls=2)
        self.mine = MINE(token_dim=token_dim, GRL=GRL)
        self.out_put = out_put
        self.device = device
    def forward(self, x):
        enc = self.encoder(x)
        emo_enc = self.emo_encoder(enc)
        task_enc = self.task_encoder(enc)
        pred_task = self.task_cls(task_enc)
        device = torch.device(self.device)
        task_shuffle = torch.index_select(task_enc, 0, torch.randperm(task_enc.shape[0]).to(device))
        joint = self.mine(emo_enc, task_enc)
        marginal = self.mine(emo_enc, task_shuffle)



        if self.out_put == 'all':
            return pred_task, emo_enc, task_enc, joint, marginal
        if self.out_put == 'pred':
            return pred_task
        if self.out_put == 'emo':
            return  emo_enc
        if self.out_put == 'task':
            return task_enc
        if self.out_put == 'mi':
            return - estimate_JSD_MI(joint, marginal, False)


