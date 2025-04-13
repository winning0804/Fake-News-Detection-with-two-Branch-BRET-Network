from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler, AdamW
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from timm.loss.asymmetric_loss import AsymmetricLossSingleLabel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import sys
from sklearn.metrics import classification_report, confusion_matrix

train_path = 'liar_dataset/train.tsv'
test_path = 'liar_dataset/test.tsv'
val_path = 'liar_dataset/valid.tsv'

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

labels = {}
statements = {}
subjects = {}
speakers = {}
jobs = {}
states = {}
affiliations = {}
credits = {}
contexts = {}
labels_onehot = {}
metadata = {}
credit_score = {}

max_seq_length_stat = 128
max_seq_length_meta = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
            num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)


def dataset_preprocessing(path,name):
    dataset_df = pd.read_csv(path, sep="\t", header=None)
    dataset_df = dataset_df.fillna(0)
    dataset = dataset_df.values

    labels[name] = [dataset[i][1] for i in range(len(dataset))]
    statements[name] = [dataset[i][2] for i in range(len(dataset))]
    subjects[name] = [dataset[i][3] for i in range(len(dataset))]
    speakers[name] = [dataset[i][4] for i in range(len(dataset))]
    jobs[name] = [dataset[i][5] for i in range(len(dataset))]
    states[name] = [dataset[i][6] for i in range(len(dataset))]
    affiliations[name] = [dataset[i][7] for i in range(len(dataset))]
    credits[name] = [dataset[i][8:13] for i in range(len(dataset))]
    contexts[name] = [dataset[i][13] for i in range(len(dataset))]

    labels_onehot[name] = to_onehot(labels[name])

    # Preparing meta data
    metadata[name] = [0]*len(dataset)
    metadata_load(dataset,name)

    # Credit score calculation
    credit_score[name] = [0]*len(dataset)
    calculate_credit_score(dataset,name)

import torch.nn.functional as F

from timm.loss.asymmetric_loss import AsymmetricLossSingleLabel

class AsymmetricLossSingleLabelOneHot(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=3.0):
        super(AsymmetricLossSingleLabelOneHot, self).__init__()
        self.asl = AsymmetricLossSingleLabel(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
        )

    def forward(self, logits, targets_onehot):
        """
        logits: [batch_size, num_classes]
        targets_onehot: [batch_size, num_classes], one-hot
        """
        target_indices = torch.max(targets_onehot, dim=1)[1]

        loss = self.asl(logits, target_indices)
        return loss


def vat_loss(model, input_ids1, input_ids2, credit_sc, eps=0.2):
    features = model.get_features(input_ids1, input_ids2, credit_sc)
    
    with torch.no_grad():
        logits = model.forward_logits(features)
        p = F.softmax(logits, dim=1)
    p = p.detach()
    
    r = torch.randn_like(features, requires_grad=True)
    
    logits_r = model.forward_logits(features + r)
    log_q = F.log_softmax(logits_r, dim=1)
    
    kl_div = F.kl_div(log_q, p, reduction='batchmean')
    
    grad_r = torch.autograd.grad(kl_div, r, retain_graph=True)[0]
    
    r_adv = F.normalize(grad_r, dim=-1)
    r_adv = eps * r_adv

    logits_r_adv = model.forward_logits(features + r_adv)
    log_q_adv = F.log_softmax(logits_r_adv, dim=1)
    
    vat_loss_value = F.kl_div(log_q_adv, p, reduction='batchmean')
    
    return vat_loss_value

def compute_rdrop_loss(logits1, logits2):
    log_probs1 = F.log_softmax(logits1, dim=1)
    log_probs2 = F.log_softmax(logits2, dim=1)
    probs1 = F.softmax(logits1, dim=1)
    probs2 = F.softmax(logits2, dim=1)
    kl_loss = (F.kl_div(log_probs1, probs2, reduction='batchmean') + 
               F.kl_div(log_probs2, probs1, reduction='batchmean')) / 2.0
    return kl_loss

def to_onehot(a):
    a_cat = [0]*len(a)
    for i in range(len(a)):
        if a[i]=='true':
            a_cat[i] = [1,0,0,0,0,0]
        elif a[i]=='mostly-true':
            a_cat[i] = [0,1,0,0,0,0]
        elif a[i]=='half-true':
            a_cat[i] = [0,0,1,0,0,0]
        elif a[i]=='barely-true':
            a_cat[i] = [0,0,0,1,0,0]
        elif a[i]=='false':
            a_cat[i] = [0,0,0,0,1,0]
        elif a[i]=='pants-fire':
            a_cat[i] = [0,0,0,0,0,1]
        else:
            print('Incorrect label')
    return a_cat

def metadata_load(dataset, name):
    for i in range(len(dataset)):
        subject = subjects[name][i]
        if subject == 0:
            subject = 'None'
        
        speaker = speakers[name][i]
        if speaker == 0:
            speaker = 'None'
        
        job = jobs[name][i]
        if job == 0:
            job = 'None'
        
        state = states[name][i]
        if state == 0:
            state = 'None'

        affiliation = affiliations[name][i]
        if affiliation == 0:
            affiliation = 'None'

        context = contexts[name][i]
        if context == 0 :
            context = 'None'
            
        meta = subject + ' ' + speaker + ' ' + job + ' ' + state + ' ' + affiliation + ' ' + context
        
        metadata[name][i] = meta

def calculate_credit_score(dataset,name):
    for i in range(len(dataset)):
        credit = credits[name][i]
        if sum(credit) == 0:
            score = 0.5
        else:
            score = (credit[3]*0.2 + credit[2]*0.5 + credit[0]*0.75 + credit[1]*0.9 + credit[4]*1)/(sum(credit))
        credit_score[name][i] = [score for i in range(1536)]

class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias
        
class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, num_labels=6): # Change number of labels here.
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)
        
    '''def forward_once(self, x):
        # Forward pass 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output'''
        
    def forward_once(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        attention_mask = (input_ids != 0).long()
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)
        pooled_output = self.dropout(pooled_output)
        #logits = self.classifier(pooled_output)

        return pooled_output
        
    def get_features(self, input_ids1, input_ids2, credit_sc):
        out1 = self.forward_once(input_ids1)
        out2 = self.forward_once(input_ids2)
        out = torch.cat((out1, out2), dim=1)
        out = out + credit_sc  
        return out

    def forward_logits(self, features):
        logits = self.classifier(features)
        return logits
    
    def forward(self, input_ids1, input_ids2, credit_sc):
        # forward pass of input 1
        output1 = self.forward_once(input_ids1, token_type_ids=None, attention_mask=None, labels=None)
        # forward pass of input 2
        output2 = self.forward_once(input_ids2, token_type_ids=None, attention_mask=None, labels=None)
        
        out = torch.cat((output1, output2), 1)
        
        # Multiply the credit score with the output after concatnation
                    
        out = torch.add(credit_sc, out)
        
        #out = self.fc1(out)
        logits = self.classifier(out)
        
        return logits
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

class text_dataset(Dataset):
    def __init__(self,x_y_list, transform=None):
        
        self.x_y_list = x_y_list
        self.transform = transform
        
    def __getitem__(self,index):
        
        # Tokenize statements
        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])

        if len(tokenized_review) > max_seq_length_stat:
            tokenized_review = tokenized_review[:max_seq_length_stat]

        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length_stat - len(ids_review))

        ids_review += padding

        assert len(ids_review) == max_seq_length_stat

        #print(ids_review)
        ids_review = torch.tensor(ids_review)
        
        sentiment = self.x_y_list[3][index] # color        
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        # Tokenize metadata

        tokenized_review_meta = tokenizer.tokenize(self.x_y_list[1][index])

        if len(tokenized_review_meta) > max_seq_length_meta:
            tokenized_review_meta = tokenized_review_meta[:max_seq_length_meta]

        ids_review_meta  = tokenizer.convert_tokens_to_ids(tokenized_review_meta)

        padding = [0] * (max_seq_length_meta - len(ids_review_meta))

        ids_review_meta += padding

        assert len(ids_review_meta) == max_seq_length_meta

        #print(ids_review)
        ids_review_meta = torch.tensor(ids_review_meta)
        
        sentiment = self.x_y_list[3][index] # color        
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        credit_scr = self.x_y_list[2][index] # Credit score
        
        #ones_768 = np.ones((768))
        #credit_scr = credit_scr * ones_768
        credit_scr = torch.tensor(credit_scr)
        
        return [ids_review, ids_review_meta, credit_scr], list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])

def load_dataset(train_path,test_path,val_path):
    dataset_preprocessing(train_path,'train')
    dataset_preprocessing(test_path,'test')
    dataset_preprocessing(val_path,'val')

    # Loading the statements
    X_train = statements['train']
    y_train = labels_onehot['train']

    X_val = statements['val']
    y_val = labels_onehot['val']

    X_test = statements['test']
    y_test = labels_onehot['test']


    # Loading the meta data
    X_train_meta = metadata['train']
    X_val_meta = metadata['val']
    X_test_meta = metadata['test']

    # Loading Credit scores

    X_train_credit = credit_score['train']
    X_val_credit = credit_score['val']
    X_test_credit = credit_score['test']


    # Small data partitioned for debugging
    '''X_train = X_train[:100]
    y_train = y_train[:100]

    X_test = X_test[:100]
    y_test = y_test[:100]

    X_train_just = X_train_just[:100]
    X_test_just = X_test_just[:100]

    X_train_meta = X_train_meta[:100]
    X_test_meta = X_test_meta[:100]

    X_train_credit = X_train_credit[:100]
    X_test_credit = X_test_credit[:100]'''

    batch_size = 128

    # Train Statements and Justifications
    train_lists = [X_train, X_train_meta, X_train_credit, y_train]

    # Val Statements and Justifications
    val_lists = [X_val, X_val_meta, X_val_credit, y_val]

    # Test Statements and Justifications
    test_lists = [X_test, X_test_meta, X_test_credit, y_test]

    # Preparing the data (Tokenize)
    training_dataset = text_dataset(x_y_list = train_lists)
    val_dataset = text_dataset(x_y_list = val_lists)
    test_dataset = text_dataset(x_y_list = test_lists)


    # Prepare the training dictionaries
    dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                    'val':torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
                    'test':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                    }
    dataset_sizes = {'train':len(train_lists[0]),
                    'val':len(val_lists[0]),
                    'test':len(test_lists[0])}
    return dataloaders_dict,dataset_sizes

class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, mode='max'):
        """
        params:
        - patience: how many epochs to wait before stopping the training
        - delta: minimum improvement to not trigger the early stopping
        - mode: 'min' for loss, 'max' for acc
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        # first time
        if self.best_score is None:
            self.best_score = current_score
            return

        # judge the improvement
        if self.mode == 'min':
            improvement = self.best_score - current_score
            if improvement > self.delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'max':
            improvement = current_score - self.best_score
            if improvement > self.delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

def train_model(model, criterion, optimizer, scheduler):
    dataloaders_dict, dataset_sizes = load_dataset(train_path,test_path,val_path)
    print(dataloaders_dict)
    print(dataset_sizes)
    early_stopper = EarlyStopping()
    num_epochs = 100
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    best_acc = 0
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        rdrop_alpha = 0.5

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()
            
            running_loss = 0.0
            sentiment_corrects = 0
        
            for inputs, sentiment in dataloaders_dict[phase]:
                inputs1 = inputs[0].to(device)
                inputs2 = inputs[1].to(device)
                inputs3 = inputs[2].to(device)
                sentiment = sentiment.to(device)
        
                optimizer.zero_grad()
                
                if phase == 'train':
                    logits1 = model(inputs1, inputs2, inputs3)
                    logits2 = model(inputs1, inputs2, inputs3)
                    loss1 = criterion(logits1, sentiment.float())
                    loss2 = criterion(logits2, sentiment.float())
                    base_loss = (loss1 + loss2) / 2.0
        
                    rdrop_loss = compute_rdrop_loss(logits1, logits2)
        
                    loss = base_loss + rdrop_alpha * rdrop_loss
        
                    if epoch >= 2:
                        v_loss = vat_loss(model, inputs1, inputs2, inputs3, eps=0.2)
                        loss += 0.3 * v_loss
        
                    loss.backward()
                    optimizer.step()
        
                    outputs = F.softmax((logits1 + logits2) / 2.0, dim=1)
                else:
                    logits = model(inputs1, inputs2, inputs3)
                    loss = criterion(logits, sentiment.float())
                    outputs = F.softmax(logits, dim=1)
        
                running_loss += loss.item() * inputs1.size(0)
                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])
            
            epoch_loss = running_loss / dataset_sizes[phase]
            sentiment_acc = sentiment_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} total loss: {epoch_loss:.4f}')
            print(f'{phase} sentiment_acc: {sentiment_acc:.4f}')
            
            sentiment_acc_cpu = sentiment_acc.cpu().numpy()
            if phase == 'train':
                train_acc.append(sentiment_acc_cpu)
                train_loss.append(epoch_loss)
            elif phase == 'val' and sentiment_acc > best_acc:
                print(f'Saving with accuracy of {sentiment_acc} improved over previous {best_acc}')
                best_acc = sentiment_acc
                val_acc.append(sentiment_acc_cpu)
                val_loss.append(epoch_loss)
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'bert_model_test_noFC1_triBERT.pth')

        print('Time taken for epoch'+ str(epoch+1)+ ' is ' + str((time.time() - epoch_start)/60) + ' minutes')

        early_stopper(sentiment_acc)
        if early_stopper.early_stop:
            print(f"⛔ Early stopping at epoch {epoch+1}")
            break
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_acc)))
    scheduler.step()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, val_acc, train_loss, val_loss

def main():
    num_labels = 6
    model = BertForSequenceClassification(num_labels)
    model.to(device)

    lrlast = .0001
    lrmain = .00002
    optim1 = AdamW(
        [
            {"params": model.bert.parameters(), "lr": lrmain},
            {"params": model.classifier.parameters(), "lr": lrlast},
        ],
        weight_decay=0.01
    )

    #optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)
    # Observe that all parameters are being optimized
    optimizer_ft = optim1
    criterion = AsymmetricLossSingleLabelOneHot(gamma_pos=1.0, gamma_neg=3.0)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    model_ft1, train_acc, val_acc, train_loss, val_loss = train_model(model, criterion, optimizer_ft, exp_lr_scheduler)
    test_model()

def test_model():
    num_labels = 6
    model = BertForSequenceClassification(num_labels)
    model.to(device)
    criterion = AsymmetricLossSingleLabelOneHot(gamma_pos=1.0, gamma_neg=3.0)
    dataloaders_dict,dataset_sizes = load_dataset(train_path,test_path,val_path)
    
    model.load_state_dict(torch.load('bert_model_test_noFC1_triBERT.pth'))
    model.eval()

    running_loss = 0.0
    sentiment_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
      for inputs, sentiment in dataloaders_dict['test']:           
          inputs1 = inputs[0] # News statement input
          inputs2 = inputs[1] # Meta data input
          inputs3 = inputs[2] # Credit scores input
          
          inputs1 = inputs1.to(device) 
          inputs2 = inputs2.to(device) 
          inputs3 = inputs3.to(device) 

          sentiment = sentiment.to(device)
          outputs = model(inputs1, inputs2, inputs3)
          outputs = F.softmax(outputs,dim=1)
          #print(outputs)
          
          loss = criterion(outputs, sentiment.float())
          
          # statistics
          running_loss += loss.item() * inputs1.size(0)
          sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])
          pred_labels = torch.argmax(outputs, dim=1)
          true_labels = torch.argmax(sentiment, dim=1)
          all_preds.extend(pred_labels.cpu().numpy())
          all_labels.extend(true_labels.cpu().numpy())
      
      epoch_loss = running_loss / dataset_sizes['test']
      sentiment_acc = sentiment_corrects.double() / dataset_sizes['test']
    print(all_preds)
    print(all_preds)
    print(f"\n Evaluation Results:")
    print(f" Average Loss: {epoch_loss:.4f}")
    print(f" Accuracy: {sentiment_acc * 100:.2f}%")

    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    print(" Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

main()