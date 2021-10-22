from collections import defaultdict
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, classification_report
import torch
import torch.distributed as dist

from Config import Config

logger = logging.getLogger(__name__)


class Performance:
    def __init__(self):
        self.config = Config.get()
        self.popts = self.config.process
        self.topts = self.config.training

        self.device = self.popts['device']
        self.distributed = self.popts['distributed']

        #self.metric_name2nd = self.topts['metric_name2nd']
        self.metric_name2nd1 = "Pearson"
        self.metric_name2nd2 = "F1"
        self.num_labels1 = self.topts['num_labels1']
        self.num_labels2 = self.topts['num_labels2']
        self.multi_label = self.topts['multi_label']
        self.task1 = self.topts['task1']
        self.task2 = self.topts['task2']
        self.patience = self.topts['patience']

        self.min_valid_loss = np.inf
        self.last_saved_epoch = 0

        self._start_time = 0
        self.training_time = 0

        self._tensors = None
        self.gathered = {}
        self._averages = defaultdict(list)
        self._init_tensors()

    def clear(self):
        self._init_tensors()

    def end_train_loop_time(self):
        self.training_time += time.time() - self._start_time

    def evaluate(self, epoch):
        done_training = False
        save_model = False

        avg_train_loss = torch.mean(self.gathered['train']['losses']).to(self.device)
        #avg_valid_loss = torch.mean(self.gathered['valid']['losses']).to(self.device)

        self._averages['train'].append(avg_train_loss)
        #self._averages['valid'].append(avg_valid_loss)

        train_2nd_metric1 = self._evaluate_2nd_metric('train')[0]
        train_2nd_metric2 = self._evaluate_2nd_metric('train')[1]
        #valid_2nd_metric = self._evaluate_2nd_metric('valid')

        # Log epoch statistics
        
        logger.info(f"Epoch {epoch:,}"
                    f"\nTraining:   loss {avg_train_loss:.6f} - {self.metric_name2nd1} 1: {train_2nd_metric1}"
                    f"\nTraining:   loss {avg_train_loss:.6f} - {self.metric_name2nd2} 2: {train_2nd_metric2}")
                    #f"\nValidation: loss {avg_valid_loss:.6f} - {self.metric_name2nd}: {valid_2nd_metric}"

        self.last_saved_epoch = epoch
        self.min_train_loss = avg_train_loss
        save_model = True

        """
        if avg_valid_loss < self.min_valid_loss:
            logger.info(f"!! Validation loss decreased ({self.min_valid_loss:.6f} --> {avg_valid_loss:.6f}).")
            self.last_saved_epoch = epoch
            self.min_valid_loss = avg_valid_loss
            save_model = True
        else:
            logger.info(f"!! Valid loss not improved. (Min. = {self.min_valid_loss:.6f};"
                        f" last save at ep. {self.last_saved_epoch})")
            if avg_train_loss <= avg_valid_loss:
                logger.warning(f"!! Training loss is lte validation loss. Might be overfitting!")
        """

        # Early-stopping
        if self.patience and (epoch - self.last_saved_epoch) == self.patience:
            logger.info(f"Stopping early at epoch {epoch} (patience={self.patience})...")
            done_training = True
        elif epoch == self.config.training['epochs']:
            done_training = True

        fig = None
        #if done_training:
        #    fig = self._plot_training()

        #return avg_valid_loss, valid_2nd_metric[0] if self.num_labels == 1 else valid_2nd_metric, fig, save_model, done_training
        #if self.num_labels == 1:
        #    return avg_train_loss, train_2nd_metric1[0], train_2nd_metric2[0], fig, save_model, done_training
        #else:
        return avg_train_loss, train_2nd_metric1, train_2nd_metric2, fig, save_model, done_training

    def _evaluate_2nd_metric(self, partition):
        """ Evaluates results of the 2nd metric. Pearson for regression, f1 for classification. """
        # In the rare case where all results are identical, metric calculation might fail
        # return None in that case
        with np.errstate(all='raise'):
            try:
                pearsonrs1 = []
                for i in range(self.num_labels1):
                    labels_i1 = [item[i] for item in self.gathered[partition]['labels1']]
                    preds_i1 = [item[i] for item in self.gathered[partition]['preds1']]
                    r_i1 = pearsonr(labels_i1, preds_i1)
                    pearsonrs1.append(r_i1[0])
                metric_res1 = np.mean(pearsonrs1)
                metric_res2 = f1_score(self.gathered[partition]['labels2'],
                                      self.gathered[partition]['preds2'],
                                      average='macro')
            except FloatingPointError:
                metric_res1 = None
                metric_res2 = None
        return metric_res1, metric_res2

    def evaluate_test(self):
        avg_test_loss = torch.mean(self.gathered['test']['losses']).item()
        test_2nd_metric1 = self._evaluate_2nd_metric('test')[0]
        test_2nd_metric2 = self._evaluate_2nd_metric('test')[1]

        report1 = None
        report2 = None
        report2 = classification_report(self.gathered['test']['labels2'],
                                       self.gathered['test']['preds2'],
                                       target_names=self.topts['target_names2'])
        logger.info(f"Classification report2:\n{report2}")

        # If regression, return the first item of the tuple (r, p-value)
        # If classification, return number (f1)

        return avg_test_loss, test_2nd_metric1, test_2nd_metric2, report1, report2

    def _init_tensors(self):
        tensor_constructor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
        self._tensors = {part: defaultdict(tensor_constructor) for part in ('test', 'train', 'valid')}

        for part in ('test', 'train', 'valid'):
            self._tensors[part]['labels2'] = torch.cuda.LongTensor([]) if self.device.type == 'cuda' else torch.LongTensor([])
            self._tensors[part]['preds2'] = torch.cuda.LongTensor([]) if self.device.type == 'cuda' else torch.LongTensor([])
                
        for part in ('test', 'train', 'valid'):
            self._tensors[part]['sentence_ids1'] = torch.cuda.LongTensor([]) if self.device.type == 'cuda' else torch.LongTensor([])
            self._tensors[part]['sentence_ids2'] = torch.cuda.LongTensor([]) if self.device.type == 'cuda' else torch.LongTensor([])

    def gather_cat(self, *partitions):
        if self.distributed:
            for partition in partitions:
                self.gathered[partition] = {
                    'losses': self._gather_cat('losses', partition).cpu(),
                    'losses1': self._gather_cat('losses1', partition).cpu(),
                    'losses2': self._gather_cat('losses2', partition).cpu(),
                    'labels1': self._gather_cat('labels1', partition).cpu(),
                    'labels2': self._gather_cat('labels2', partition).cpu(),
                    'sentence_ids1': self._gather_cat('sentence_ids1', partition).cpu(),
                    'sentence_ids2': self._gather_cat('sentence_ids2', partition).cpu(),
                    'preds1': self._gather_cat('preds1', partition).cpu(),
                    'preds2': self._gather_cat('preds2', partition).cpu()
                }
        else:
            self.gathered = {part: {attr: tensor.cpu()
                                    for attr, tensor in d.items()}
                             for part, d in self._tensors.items()}

    def _gather_cat(self, attr, partition):
        x = self._tensors[partition][attr]
        gather = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gather, x)
        return torch.cat(gather)

    def _plot_training(self):
        """ Plot loss into plt graph.
            :returns the figure object of the graph """
        train_losses = self._averages['train']
        valid_losses = self._averages['valid']
        fig = plt.figure(dpi=300)
        plt.plot(train_losses, label='Training loss')
        plt.plot(valid_losses, label='Validation loss')
        plt.xlabel('epochs')
        plt.legend(frameon=False)
        plt.title(f"Loss progress ({self.topts['metric_name']})")
        # Set ticks to integers for the epochs rather than floats
        plt.xticks(ticks=range(len(train_losses)), labels=range(1, len(train_losses)+1))
        plt.show()

        return fig

    def start_train_loop_time(self):
        self._start_time = time.time()

    def update(self, attr, partition, tensor):
        # with batch_size 1, the prediction tensor might be zero-dimensions
        if not tensor.size():
            tensor = tensor.unsqueeze(0)
        if self.multi_label == 0:
            self._tensors[partition][attr] = torch.cat((self._tensors[partition][attr], tensor.detach()))
        else:
            if attr in ['sentence_ids1', 'sentence_ids2']:
                self._tensors[partition][attr] = torch.cat((self._tensors[partition][attr], tensor.detach()))
            else:
                self._tensors[partition][attr] = torch.cat((self._tensors[partition][attr], tensor.float().detach()))
