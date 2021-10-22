from math import ceil
import logging

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn import MSELoss, CrossEntropyLoss, Softmax, BCEWithLogitsLoss, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np

try:
    from apex import amp
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from Config import Config
from DfDataset import DfDataset
from Performance import Performance

#logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class TransformerTrainer:
    """ Trainer that executes the training, validation, and testing loops
        given all required information such as model, tokenizer, optimizer and so on. """
    def __init__(self,
                 model,
                 optimizer=None,
                 scheduler=None,
                 map_location=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.map_location = map_location

        self.config = Config.get()
        self.mopts = self.config.model
        self.popts = self.config.process
        self.topts = self.config.training

        self.device = self.popts['device']
        self.distributed = self.popts['distributed']
        self.is_first = self.popts['is_first']

        self.batch_size = self.topts['batch_size']
        self.fp16 = self.topts['fp16']
        self.num_labels1 = self.topts['num_labels1']
        self.num_labels2 = self.topts['num_labels2']
        self.multi_label = self.topts['multi_label']
        self.task1 = self.topts['task1']
        self.task2 = self.topts['task2']
        self.multi_task = self.topts['multi_task']
        

        # When the model is wrapped as DistributedDataParallel,
        # its properties are not overtly available. Use .module to access them
        try:
            self.tokenizer = model.tokenizer_wrapper
        except AttributeError:
            self.tokenizer = model.module.tokenizer_wrapper

        self._set_data_loaders()
        """
        if self.task == 'regression':
            self.criterion = MSELoss().to(self.device)
        else:
            if self.topts['label_weights']:
                self.criterion = CrossEntropyLoss(weight=torch.tensor(self.topts['label_weights'])).to(self.device)
            else:
                if self.multi_label == 0:
                    self.criterion = CrossEntropyLoss().to(self.device)
                else:
                    self.criterion = BCEWithLogitsLoss().to(self.device)
        """
        self.criterion1 = MSELoss().to(self.device)
        self.criterion2 = CrossEntropyLoss().to(self.device)

        #self.softmax = Softmax(dim=1).to(self.device) if self.num_labels > 1 else None
        #self.softmax = Softmax(dim=1).to(self.device) if self.task == 'classification' else None
        self.softmax = Softmax(dim=1).to(self.device)
        #self.softmax = None
        self.sigmoid = Sigmoid().to(self.device)
        self.performer = Performance()

    def _prepare_lines(self, data, t, labels=False):
        """ Basic line preparation, strips away new lines.
            Can also prepare labels as the expected tensor type"""
        if labels:
            # For regression, cast to float (FloatTensor)
            # For classification, cast to int (LongTensor)
            #if self.task == 'regression':
            if t == 1:
                if self.num_labels1 == 1:
                    out = torch.FloatTensor([float(item.rstrip()) for item in data])
                else:
                    out = []
                    for item in data:
                        label_ml = [float(i) for i in item.rstrip().split(',')]
                        out.append(label_ml)
                    out = torch.from_numpy(np.asarray(out)).float()
            #else:
            elif t == 2:
                if self.multi_label2 == 0:
                    out = torch.LongTensor([int(item.rstrip()) for item in data])
                else:
                    out = []
                    for item in data:
                        label_ml = [int(i) for i in item.rstrip().split(',')]
                        out.append(label_ml)
                    out = torch.from_numpy(np.asarray(out)).float()
        else:
            out = [item.rstrip() for item in data]

        return out

    def _process(self, do, epoch=0, inference=False):
        """ Runs the training, validation, or testing (for one epoch) """
        if do == 'train' and not inference:
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        if self.multi_task == 1:
            dataloader2_iterator = iter(self.dataloaders2[do])

        # task 1
        if self.distributed:
            self.samplers1[do].set_epoch(epoch)

        # only run tqdm progressbar for first process
        progress_bar1 = None
        if self.is_first:
            nro_batches1 = ceil(len(self.datasets1[do]) / self.dataloaders1[do].batch_size)
            desc = f"Epoch {epoch:,} ({do})" if do in {'train', 'valid'} else 'Test'
            progress_bar1 = tqdm(desc=desc, total=nro_batches1)

        # Main loop: iterate over dataloader
        for batch_idx, data in enumerate(self.dataloaders1[do], 1):
            # 0. Clear gradients
            if do == 'train' and not inference:
                self.performer.start_train_loop_time()
                self.optimizer.zero_grad()

            # 1. Data prep
            text1 = self._prepare_lines(data['text'], t = 1)
            encoded_inputs1 = self.tokenizer.encode_batch_plus(text1,
                                                              batch_pair=None,
                                                              pad_to_batch_length=True,
                                                              return_tensors='pt')

            if inference:
                sentence_ids1 = data['id'].to(self.device)
            else:
                #if self.task == 'regression':
                if self.num_labels1 == 1:
                    labels1 = data['label'].to(self.device, dtype=torch.int64)
                else:
                    labels1 = self._prepare_lines(data['label'], t = 1, labels=True)
                    labels1 = labels1.to(self.device, dtype=torch.float64)
                #else:
                #    if self.multi_label == 0:
                #        labels1 = data['label'].to(self.device, dtype=torch.int64)
                #    else:
                #        labels1 = self._prepare_lines(data['label'], labels=True)
                #        labels1 = labels1.to(self.device, dtype=torch.float64)
                

            encoded_inputs1['input_ids'] = encoded_inputs1['input_ids'].to(self.device)
            encoded_inputs1['attention_mask'] = encoded_inputs1['attention_mask'].to(self.device)
            encoded_inputs1['token_type_ids'] = encoded_inputs1['token_type_ids'].to(self.device)

            # 2. Predictions
            try:
                preds1 = self.model(t=1, **encoded_inputs1)
            except RuntimeError as e:
                with open('error.log', 'w', encoding='utf-8') as fhout:
                    fhout.write(str(data) + '\n')
                    fhout.write(str(encoded_inputs1) + '\n')

                with open('trace.log', 'w', encoding='utf-8') as traceout:
                    traceout.write(str(e) + '\n')

                raise RuntimeError()

            #if self.task == 'regression':
            preds1 = preds1.squeeze()
            if not inference:
                #if self.num_labels == 1:
                #    loss1 = self.criterion1(preds1.view(-1), labels1.view(-1)).unsqueeze(0)
                #else:
                    #losses = []
                    #for i in range(self.num_labels):
                    #    preds_i = preds[:, i]
                    #    labels_i = labels[:, i]
                    #    loss_i = self.criterion(preds_i, labels_i)
                    #    losses.append(loss_i)
                    #losses = torch.tensor(losses, requires_grad=True)
                    #loss = torch.mean(losses).unsqueeze(0).to(self.device)
                loss1 = self.criterion1(preds1.to(self.device, dtype=torch.float64), labels1.to(self.device, dtype=torch.float64)).unsqueeze(0)
                    #print(loss)
                    #print(loss.type())


            #else:
            #    if not inference:
            #        loss1 = self.criterion(preds1, labels1).unsqueeze(0)
            #    if self.multi_label == 0:
            #        probs1 = self.softmax(preds1)
            #        preds1 = torch.topk(probs1, 1).indices.squeeze()
            #    else:
            #        preds1 = self.sigmoid(preds1)
            #        probs1 = preds1
            #        preds1 = preds1 > 0.5
            #        preds1 = preds1.squeeze()

            if self.multi_task == 1:
                # task 2
                if self.distributed:
                    self.samplers2[do].set_epoch(epoch)

                # Main loop: iterate over dataloader
                try:
                    data = next(dataloader2_iterator)
                except:
                    dataloader2_iterator = iter(self.dataloaders2[do])
                    data = next(dataloader2_iterator)


                # 0. Clear gradients
                if do == 'train' and not inference:
                    self.performer.start_train_loop_time()
                    self.optimizer.zero_grad()
    
                # 1. Data prep
                text2 = self._prepare_lines(data['text'], t = 2)
                encoded_inputs2 = self.tokenizer.encode_batch_plus(text2,
                                                                  batch_pair=None,
                                                                  pad_to_batch_length=True,
                                                                  return_tensors='pt')
    
                if inference:
                    sentence_ids2 = data['id'].to(self.device)
                else:
                    #if self.task == 'regression':
                    #    if self.num_labels == 1:
                    #        labels2 = data['label'].to(self.device, dtype=torch.int64)
                    #    else:
                    #        labels2 = self._prepare_lines(data['label'], labels=True)
                    #        labels2 = labels2.to(self.device, dtype=torch.float64)
                    #else:
                    #if self.multi_label2 == 0:
                    labels2 = data['label'].to(self.device, dtype=torch.int64)
                    #else:
                    #    labels2 = self._prepare_lines(data['label'], labels=True)
                    #    labels2 = labels2.to(self.device, dtype=torch.float64)
                    
    
                encoded_inputs2['input_ids'] = encoded_inputs2['input_ids'].to(self.device)
                encoded_inputs2['attention_mask'] = encoded_inputs2['attention_mask'].to(self.device)
                encoded_inputs2['token_type_ids'] = encoded_inputs2['token_type_ids'].to(self.device)
    
                # 2. Predictions
                try:
                    preds2 = self.model(t=2, **encoded_inputs2)
                except RuntimeError as e:
                    with open('error.log', 'w', encoding='utf-8') as fhout:
                        fhout.write(str(data) + '\n')
                        fhout.write(str(encoded_inputs2) + '\n')
    
                    with open('trace.log', 'w', encoding='utf-8') as traceout:
                        traceout.write(str(e) + '\n')
    
                    raise RuntimeError()
    
                #if self.task == 'regression':
                #    preds2 = preds2.squeeze()
                #    if not inference:
                #        if self.num_labels == 1:
                #            loss2 = self.criterion(preds2.view(-1), labels2.view(-1)).unsqueeze(0)
                #        else:
                #            #losses = []
                #            #for i in range(self.num_labels):
                #            #    preds_i = preds[:, i]
                #            #    labels_i = labels[:, i]
                #            #    loss_i = self.criterion(preds_i, labels_i)
                #            #    losses.append(loss_i)
                #            #losses = torch.tensor(losses, requires_grad=True)
                #            #loss = torch.mean(losses).unsqueeze(0).to(self.device)
                #            loss2 = self.criterion(preds2.to(self.device, dtype=torch.float64), labels2.to(self.device, dtype=torch.float64)).unsqueeze(0)
                #            #print(loss)
                #            #print(loss.type())
    
    
                #else:
                if not inference:
                    loss2 = self.criterion2(preds2, labels2).unsqueeze(0)
                if self.multi_label == 0:
                    probs2 = self.softmax(preds2)
                    preds2 = torch.topk(probs2, 1).indices.squeeze()
                else:
                    preds2 = self.sigmoid(preds2)
                    probs2 = preds2
                    preds2 = preds2 > 0.5
                    preds2 = preds2.squeeze()
    


                # 3. Optimise during training
                if not inference:
                    w1 = (self.topts['task_loss_weights'])[0]
                    w2 = (self.topts['task_loss_weights'])[1]
                    loss = w1*(loss1) + w2*(loss2)


                if do == 'train' and not inference:
                    if self.fp16:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    self.optimizer.step()
    
                # 4. Save results
                if inference:
                    self.performer.update('sentence_ids1', do, sentence_ids1)
                    self.performer.update('sentence_ids2', do, sentence_ids2)
                    #if self.task == 'classification':
                        #self.performer.update('probs1', do, probs1)
                    self.performer.update('probs2', do, probs2)
                else:
                    self.performer.update('labels1', do, labels1)
                    self.performer.update('losses1', do, loss1)
                    self.performer.update('labels2', do, labels2)
                    self.performer.update('losses2', do, loss2)
                    self.performer.update('losses', do, loss)
                self.performer.update('preds1', do, preds1)
                self.performer.update('preds2', do, preds2)
    
            if progress_bar1:
                upd_step = min(self.popts['world_size'], progress_bar1.total - progress_bar1.n)
                progress_bar1.update(upd_step)

            if do == 'train' and not inference:
                self.performer.end_train_loop_time()

        if progress_bar1:
            progress_bar1.close()

    def _save_model(self, valid_metric1, valid_metric2):
        """ Saves current model as well as additional information. """
        info_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'valid_loss': self.performer.min_valid_loss,
            'valid_metric1': valid_metric1,
            'valid_metric2': valid_metric2,
            'epoch': self.performer.last_saved_epoch,
            'training_time': self.performer.training_time
        }

        if self.fp16 and AMP_AVAILABLE:
            info_dict['amp_state_dict'] = amp.state_dict()

        torch.save(info_dict, 'tmp-checkpoint.pth')

    def _set_data_loaders(self):
        """ Create datasets and their respective dataloaders.
            See DfDataset.py """

        # task 1
        train_file1 = self.topts['files1']['train1'] if 'train1' in self.topts['files1'] else None
        valid_file1 = self.topts['files1']['valid1'] if 'valid1' in self.topts['files1'] else None
        test_file1 = self.topts['files1']['test1'] if 'test1' in self.topts['files1'] else None

        self.datasets1 = {
            'train': DfDataset(train_file1, sep=self.topts['sep']) if train_file1 is not None else None,
            'valid': DfDataset(valid_file1, sep=self.topts['sep']) if valid_file1 is not None else None,
            'test': DfDataset(test_file1, sep=self.topts['sep']) if test_file1 is not None else None
        }

        if train_file1:
            logger.info(f"Training set size 1: {len(self.datasets1['train'])}")
        if valid_file1:
            logger.info(f"Validation set size 1: {len(self.datasets1['valid'])}")
        if test_file1:
            logger.info(f"Test set size 1: {len(self.datasets1['test'])}")

        self.samplers1 = {part: None for part in ('train', 'valid', 'test')}
        if self.distributed:
            self.samplers1 = {
                'train': DistributedSampler(self.datasets1['train']) if train_file1 is not None else None,
                'valid': DistributedSampler(self.datasets1['valid']) if valid_file1 is not None else None,
                'test': DistributedSampler(self.datasets1['test']) if test_file1 is not None else None
            }

        self.dataloaders1 = {
            # no advantage of running more than 1 num_workers here
            'train': DataLoader(self.datasets1['train'], batch_size=self.batch_size, shuffle=False, num_workers=1,
                                pin_memory=True, sampler=self.samplers1['train'] if self.samplers1['train'] else None)
                     if train_file1 is not None else None,
            'valid': DataLoader(self.datasets1['valid'], batch_size=self.batch_size, shuffle=False, num_workers=1,
                                pin_memory=True, sampler=self.samplers1['valid'] if self.samplers1['valid'] else None)
                     if valid_file1 is not None else None,
            'test': DataLoader(self.datasets1['test'], batch_size=self.batch_size, shuffle=False, num_workers=1,
                               pin_memory=True, sampler=self.samplers1['test'] if self.samplers1['test'] else None)
                    if test_file1 is not None else None
        }


        # task 2
        train_file2 = self.topts['files2']['train2'] if 'train2' in self.topts['files2'] else None
        valid_file2 = self.topts['files2']['valid2'] if 'valid2' in self.topts['files2'] else None
        test_file2 = self.topts['files2']['test2'] if 'test2' in self.topts['files2'] else None

        self.datasets2 = {
            'train': DfDataset(train_file2, sep=self.topts['sep']) if train_file2 is not None else None,
            'valid': DfDataset(valid_file2, sep=self.topts['sep']) if valid_file2 is not None else None,
            'test': DfDataset(test_file2, sep=self.topts['sep']) if test_file2 is not None else None
        }

        if train_file2:
            logger.info(f"Training set size 2: {len(self.datasets2['train'])}")
        if valid_file2:
            logger.info(f"Validation set size 2: {len(self.datasets2['valid'])}")
        if test_file2:
            logger.info(f"Test set size 2: {len(self.datasets2['test'])}")

        self.samplers2 = {part: None for part in ('train', 'valid', 'test')}
        if self.distributed:
            self.samplers2 = {
                'train': DistributedSampler(self.datasets2['train']) if train_file2 is not None else None,
                'valid': DistributedSampler(self.datasets2['valid']) if valid_file2 is not None else None,
                'test': DistributedSampler(self.datasets2['test']) if test_file2 is not None else None
            }

        self.dataloaders2 = {
            # no advantage of running more than 1 num_workers here
            'train': DataLoader(self.datasets2['train'], batch_size=self.batch_size, shuffle=False, num_workers=1,
                                pin_memory=True, sampler=self.samplers2['train'] if self.samplers2['train'] else None)
                     if train_file2 is not None else None,
            'valid': DataLoader(self.datasets2['valid'], batch_size=self.batch_size, shuffle=False, num_workers=1,
                                pin_memory=True, sampler=self.samplers2['valid'] if self.samplers2['valid'] else None)
                     if valid_file2 is not None else None,
            'test': DataLoader(self.datasets2['test'], batch_size=self.batch_size, shuffle=False, num_workers=1,
                               pin_memory=True, sampler=self.samplers2['test'] if self.samplers2['test'] else None)
                    if test_file2 is not None else None
        }

    def load_model(self, checkpoint_f, eval_mode=False):
        """ Load checkpoint, especially used for testing. """
        checkpoint = torch.load(checkpoint_f, map_location=self.map_location if self.map_location else self.device)

        # running in DDP mode will add module., so might need to remove that in the keys
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            # maybe doing this in comprehension is too memory intensive?
            checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in
                                              checkpoint['model_state_dict'].items()}
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # only load optimizer if not in eval mode
        if not eval_mode and self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.fp16 and AMP_AVAILABLE and 'amp_state_dict' in checkpoint:
                amp.load_state_dict(checkpoint['amp_state_dict'])

        # If we don't do this, it might lead to CUDA OOM issues for larger models
        # I'm not sure why since after exiting the function, I'd expect the variable to
        # be free-able, but this seems to work after testing.
        del checkpoint

    def infer(self, checkpoint_f):
        """ Predicts values for the file in 'test' """
        self.load_model(checkpoint_f, eval_mode=True)

        self._process('test', inference=True)
        self.performer.gather_cat('test')
        #print(self.performer.gathered['test'])

        # task 1
        #if self.task == 'regression':
        if self.num_labels1 == 1:
            pred_type = float
            data1 = {
            'id': self.performer.gathered['test']['sentence_ids1'].numpy().astype(int),
            'pred': self.performer.gathered['test']['preds1'].numpy().astype(pred_type1)
            }
        else:
            pred = self.performer.gathered['test']['preds1'].numpy().tolist()
            new_pred = []
            for label in pred:
                label = [float(instance) for instance in label]
                label = [str(instance) for instance in label]
                label = ','.join(label)
                new_pred.append(label)
            #print(new_pred)
            data1 = {
            'id': self.performer.gathered['test']['sentence_ids1'].numpy().astype(int),
            'pred': new_pred
            }

        #else:
        #    if self.topts['multi_label'] == 0:
        #        pred_type1 = int
        #        prob = self.performer.gathered['test']['probs1'].numpy().tolist()
        #        new_prob = []
        #        for label in prob:
        #            label = [float(instance) for instance in label]
        #            label = [str(instance) for instance in label]
        #            label = ','.join(label)
        #            new_prob.append(label)
        #        data1 = {
        #        'id': self.performer.gathered['test']['sentence_ids1'].numpy().astype(int),
        #        'pred': self.performer.gathered['test']['preds1'].numpy().astype(pred_type),
        #        'prob': new_prob
        #        }
        #    else:
        #        pred = self.performer.gathered['test']['preds1'].numpy().tolist()
        #        prob = self.performer.gathered['test']['probs1'].numpy().tolist()
        #        new_pred = []
        #        for label in pred:
        #            label = [int(instance) for instance in label]
        #            label = [str(instance) for instance in label]
        #            label = ','.join(label)
        #            new_pred.append(label)
        #        new_prob = []
        #        for label in prob:
        #            label = [float(instance) for instance in label]
        #            label = [str(instance) for instance in label]
        #            label = ','.join(label)
        #            new_prob.append(label)
        #        #print(new_pred)
        #        data1 = {
        #        'id': self.performer.gathered['test']['sentence_ids1'].numpy().astype(int),
        #        'pred': new_pred,
        #        'prob': new_prob
        #        }

        # task 2
        #if self.task == 'regression':
        #    if self.num_labels == 1:
        #        pred_type = float
        #        data2 = {
        #        'id': self.performer.gathered['test']['sentence_ids2'].numpy().astype(int),
        #        'pred': self.performer.gathered['test']['preds2'].numpy().astype(pred_type2)
        #        }
        #    else:
        #        pred = self.performer.gathered['test']['preds2'].numpy().tolist()
        #        new_pred = []
        #        for label in pred:
        #            label = [float(instance) for instance in label]
        #            label = [str(instance) for instance in label]
        #            label = ','.join(label)
        #            new_pred.append(label)
        #        #print(new_pred)
        #        data2 = {
        #        'id': self.performer.gathered['test']['sentence_ids2'].numpy().astype(int),
        #        'pred': new_pred
        #        }

        #else:
        if self.topts['multi_label'] == 0:
            pred_type2 = int
            prob = self.performer.gathered['test']['probs2'].numpy().tolist()
            new_prob = []
            for label in prob:
                label = [float(instance) for instance in label]
                label = [str(instance) for instance in label]
                label = ','.join(label)
                new_prob.append(label)
            data2 = {
            'id': self.performer.gathered['test']['sentence_ids2'].numpy().astype(int),
            'pred': self.performer.gathered['test']['preds2'].numpy().astype(pred_type2),
            'prob': new_prob
            }
        else:
            pred = self.performer.gathered['test']['preds2'].numpy().tolist()
            prob = self.performer.gathered['test']['probs2'].numpy().tolist()
            new_pred = []
            for label in pred:
                label = [int(instance) for instance in label]
                label = [str(instance) for instance in label]
                label = ','.join(label)
                new_pred.append(label)
            new_prob = []
            for label in prob:
                label = [float(instance) for instance in label]
                label = [str(instance) for instance in label]
                label = ','.join(label)
                new_prob.append(label)
            #print(new_pred)
            data2 = {
            'id': self.performer.gathered['test']['sentence_ids2'].numpy().astype(int),
            'pred': new_pred,
            'prob': new_prob
            }
        

        df1 = pd.DataFrame.from_dict(data1)
        #df.to_csv('multilabel_data/output/predictions/predictions.csv', index=False)
        #filename = 'eventdna/predictions_' + str(self.topts['files']['test'][36:])
        filename1 = self.topts['pred_path1']
        df1.to_csv(filename1, index=False)

        df2 = pd.DataFrame.from_dict(data2)
        #df.to_csv('multilabel_data/output/predictions/predictions.csv', index=False)
        #filename = 'eventdna/predictions_' + str(self.topts['files']['test'][36:])
        filename2 = self.topts['pred_path2']
        df2 = df2.drop_duplicates()
        df2.to_csv(filename2, index=False)

    def test(self, checkpoint_f):
        """ Wraps testing a given model. Actual testing is done in `self._process()`. """
        self.load_model(checkpoint_f, eval_mode=True)

        self._process('test')
        self.performer.gather_cat('test')
        avg_test_loss, test_2nd_metric1, test_2nd_metric2, report1, report2 = self.performer.evaluate_test()

        return avg_test_loss, test_2nd_metric1, test_2nd_metric2, report1, report2

    def train(self):
        """ Entry point to start training the model. Will run the outer epoch loop containing
            training and validation. Also implements early stopping, set by `self.patience`.
            Actual training/validating is done in `self._process()` """
        logger.info('Training started.')

        done_training = False
        fig = None
        #best_valid_loss = 0
        #best_valid_2nd_metric = 0
        best_train_loss = 0
        best_train_2nd_metric1 = 0
        best_train_2nd_metric2 = 0
        for epoch in range(1, self.topts['epochs'] + 1):
            # TRAINING
            self._process('train', epoch)

            self.performer.gather_cat('train')

            avg_train_loss = torch.empty(1).to(self.device)
            if self.is_first:
                avg_train_loss, train_2nd_metric1, train_2nd_metric2, fig, save_model, done_training = self.performer.evaluate(epoch)
                if save_model:
                    self._save_model(train_2nd_metric1, train_2nd_metric2)
                    best_train_loss = avg_train_loss.item()
                    best_train_2nd_metric1 = train_2nd_metric1
                    best_train_2nd_metric2 = train_2nd_metric2

            if self.distributed:
                # broadcast done_training: due to a bug, broadcasting booleans does not work as expected
                # to by-pass this, cast to long (int) first, and then recast as bool
                # see https://github.com/pytorch/pytorch/issues/24137
                done_training = torch.tensor(done_training).long().to(self.device)
                torch.distributed.broadcast(done_training, src=0)
                done_training = bool(done_training.item())

                # broadcast avg_valid_loss for the scheduler
                torch.distributed.broadcast(avg_train_loss, src=0)

            if done_training:
                break

            # adjust learning rate with scheduler
            if self.scheduler is not None:
                self.scheduler.step(avg_train_loss)

            """
            # VALIDATION
            self._process('valid', epoch)
            self.performer.gather_cat('train', 'valid')

            avg_valid_loss = torch.empty(1).to(self.device)
            if self.is_first:
                avg_valid_loss, valid_2nd_metric, fig, save_model, done_training = self.performer.evaluate(epoch)
                if save_model:
                    self._save_model(valid_2nd_metric)
                    best_valid_loss = avg_valid_loss.item()
                    best_valid_2nd_metric = valid_2nd_metric

            if self.distributed:
                # broadcast done_training: due to a bug, broadcasting booleans does not work as expected
                # to by-pass this, cast to long (int) first, and then recast as bool
                # see https://github.com/pytorch/pytorch/issues/24137
                done_training = torch.tensor(done_training).long().to(self.device)
                torch.distributed.broadcast(done_training, src=0)
                done_training = bool(done_training.item())

                # broadcast avg_valid_loss for the scheduler
                torch.distributed.broadcast(avg_valid_loss, src=0)

            if done_training:
                break

            # adjust learning rate with scheduler
            if self.scheduler is not None:
                self.scheduler.step(avg_valid_loss)
            """

            # clear the performer's saved labels, losses, preds
            self.performer.clear()

        #return fig, best_valid_loss, best_valid_2nd_metric
        return fig, best_train_loss, best_train_2nd_metric1, best_train_2nd_metric2
