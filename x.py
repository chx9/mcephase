from torch.utils.tensorboard import SummaryWriter
writer_train = SummaryWriter('runs/train')
writer_train.add_scalar('test', 1, 0)
# writer_train.close()

writer_test = SummaryWriter('runs/test')
writer_test.add_scalar('test', 1, 0)
# writer_test.close()