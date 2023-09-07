import time
import torch
from evaluation import compute_accuracy
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs/experiment')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Build a function to train our model
def train_model(model, model_autoencoder, num_epochs, train_loader,
                valid_loader, test_loader, optimizer,
                device, logging_interval=50,
                scheduler=None,
                scheduler_on='valid_acc'):

    start_time = time.time()
    epoch_loss = AverageMeter()
    batch_loss_list, train_accuracy_list, valid_accuracy_list = [], [], []

    for epoch in range(num_epochs):

        model.train()
        model_autoencoder.eval()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)
            _, extract_feature = model_autoencoder(features)
            # ## FORWARD AND BACK PROP
            logits = model(features, extract_feature)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            epoch_loss.update(loss.data)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            batch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')
            writer.add_scalar("Epoch training loss", epoch_loss.avg, epoch)
        torch.save(model.state_dict(), "C://Users//ngoth//Documents//UMKC//Dark_VGG1645//vgg-encoder-final_02.ckpt")

        model.eval()
        with torch.no_grad():
            train_acc = compute_accuracy(model, model_autoencoder, train_loader, device=device)
            valid_acc = compute_accuracy(model, model_autoencoder, valid_loader, device=device)
            print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}%')
            train_accuracy_list.append(train_acc.item())
            valid_accuracy_list.append(valid_acc.item())

        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')
        writer.add_scalar("Training accuracy", train_acc, epoch)
        writer.add_scalar("Validation accuracy", valid_acc, epoch)
        if scheduler is not None:
            if scheduler_on == 'valid_acc':
                scheduler.step(valid_accuracy_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(batch_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')

    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, model_autoencoder, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return batch_loss_list, train_accuracy_list, valid_accuracy_list
