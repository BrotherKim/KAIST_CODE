import os
import logging
from tqdm.auto import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter

import sklearn.metrics as metrics

from utils import compute_metrics, get_label, MODEL_CLASSES

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)

        self.config_class, self.model_class, _ = MODEL_CLASSES[args['model_type']]

        #### YOUR CODE HERE ####
        # Load BERT configuration and BertForSequenceClassification from HuggingFace
        self.config = self.config_class.from_pretrained(args['model_name_or_path'],
                                                        num_labels=self.num_labels, 
                                                        finetuning_task=args['task'],
                                                        id2label={str(i): label for i, label in enumerate(self.label_lst)},
                                                        label2id={label: i for i, label in enumerate(self.label_lst)})
        self.model = self.model_class.from_pretrained(args['model_name_or_path'], config=self.config)
        #### END CODE HERE ####

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args['no_cuda'] else "cpu"
        self.model.to(self.device)
        
        # Add Summary Writer
        self.train_writer = SummaryWriter(os.path.join(args['model_dir'], 'train'))
        self.test_writer = SummaryWriter(os.path.join(args['model_dir'], 'test'))

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args['train_batch_size'])

        if self.args['max_steps'] > 0:
            t_total = self.args['max_steps']
            self.args['num_train_epochs'] = self.args['max_steps'] // (len(train_dataloader) // self.args['gradient_accumulation_steps']) + 1
        else:
            t_total = len(train_dataloader) // self.args['gradient_accumulation_steps'] * self.args['num_train_epochs']

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args['weight_decay']},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args['learning_rate'], eps=self.args['adam_epsilon'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args['warmup_steps'], num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args['num_train_epochs'])
        logger.info("  Total train batch size = %d", self.args['train_batch_size'])
        logger.info("  Gradient Accumulation steps = %d", self.args['gradient_accumulation_steps'])
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args['logging_steps'])
        logger.info("  Save steps = %d", self.args['save_steps'])

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args['num_train_epochs']), desc="Epoch")

        for _ in train_iterator:
            # 데이터로더에서 배치를 가져온다.
            epoch_iterator = tqdm(train_dataloader, ncols=120, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                # 훈련 모드로 변경
                self.model.train()
                # 배치를 GPU에 복사
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                # 배치로부터 입력값과 라벨을 가져온다.
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}

                # forward 수행
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args['gradient_accumulation_steps'] > 1:
                    loss = loss / self.args['gradient_accumulation_steps']

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    
                    # Add current loss values to tensorboard
                    self.train_writer.add_scalar('train/loss', loss.item(), global_step)
                    self.train_writer.add_scalar('train/trloss', tr_loss / global_step, global_step)

                    if self.args['logging_steps'] > 0 and global_step % self.args['logging_steps'] == 0:
                        self.evaluate("dev")  # Only test set available for NSMC

                    if self.args['save_steps'] > 0 and global_step % self.args['save_steps'] == 0:
                        self.save_model()

                if 0 < self.args['max_steps'] < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args['max_steps'] < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args['eval_batch_size'])

        # Eval!
        logger.info("*****d Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args['eval_batch_size'])
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, ncols=120, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}

                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info('accuracy : %s', str(metrics.accuracy_score(out_label_ids,preds)))
        logger.info('precision1 : %s', str(metrics.precision_score(out_label_ids,preds, average=None))) #
        logger.info('precision2 : %s', str(metrics.precision_score(out_label_ids,preds, average=None).mean())) #
        logger.info('precision3 : %s', str(metrics.precision_score(out_label_ids,preds, average='macro'))) #1.0
        logger.info('precision4 : %s', str(metrics.precision_score(out_label_ids,preds, average='micro'))) #1.0
        logger.info('recall1 : %s', str(metrics.recall_score(out_label_ids,preds, average=None))) #
        logger.info('recall2 : %s', str(metrics.recall_score(out_label_ids,preds, average=None).mean())) #
        logger.info('recall3 : %s', str(metrics.recall_score(out_label_ids,preds, average='macro'))) #1.0
        logger.info('recall4 : %s', str(metrics.recall_score(out_label_ids,preds, average='micro'))) #1.0
        logger.info('f1_score1 : %s', str(metrics.f1_score(out_label_ids,preds, average=None))) #
        logger.info('f1_score2 : %s', str(metrics.f1_score(out_label_ids,preds, average=None).mean())) #
        logger.info('f1_score3 : %s', str(metrics.f1_score(out_label_ids,preds, average='macro'))) #1.0
        logger.info('f1_score4 : %s', str(metrics.f1_score(out_label_ids,preds, average='micro'))) #1.0

        print(metrics.classification_report(out_label_ids,preds))
        print(metrics.confusion_matrix(out_label_ids,preds))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args['model_dir']):
            os.makedirs(self.args['model_dir'])
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args['model_dir'])

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args['model_dir'], 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args['model_dir'])

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args['model_dir']):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args['model_dir'])
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
