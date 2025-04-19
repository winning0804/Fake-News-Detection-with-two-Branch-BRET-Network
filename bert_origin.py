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
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

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

class BertForSingleInputClassification(nn.Module):
    def __init__(self, num_labels=6):
        super(BertForSingleInputClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)  # 只输出单个BERT结果
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).long()
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class SingleInputDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        text = self.texts[index]
        label = torch.tensor(self.labels[index])
        
        tokens = tokenizer.tokenize(text)
        tokens = tokens[:128]  # max_seq_length
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (128 - len(input_ids))
        input_ids += padding
        input_ids = torch.tensor(input_ids)
        
        return input_ids, label

    def __len__(self):
        return len(self.texts)


def load_dataset(train_path,test_path,val_path):
    dataset_preprocessing(train_path,'train')
    dataset_preprocessing(test_path,'test')
    dataset_preprocessing(val_path,'val')

    train_dataset = SingleInputDataset(statements['train'], labels_onehot['train'])
    val_dataset = SingleInputDataset(statements['val'], labels_onehot['val'])
    test_dataset = SingleInputDataset(statements['test'], labels_onehot['test'])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


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
    train_loader, val_loader, test_loader = load_dataset(train_path, test_path, val_path)
    
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

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                dataloader = train_loader
            else:
                model.eval()   # 验证模式，只做一次前向传播
                dataloader = val_loader
            
            running_loss = 0.0
            sentiment_corrects = 0
            all_preds = []
            all_labels = []

            # 遍历当前 phase 的数据
            for input_ids, labels in dataloader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if phase == 'train':
                    # 只进行一次前向传播
                    logits = model(input_ids)
                    loss = criterion(logits, torch.max(labels, 1)[1])

                    loss.backward()
                    optimizer.step()

                    outputs = F.softmax(logits, dim=1)
                else:
                    # 验证阶段只做一次前向传播
                    logits = model(input_ids)
                    loss = criterion(logits, torch.max(labels, 1)[1])
                    outputs = F.softmax(logits, dim=1)

                # 累计统计
                running_loss += loss.item() * input_ids.size(0)
                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(labels, 1)[1])
                pred_labels = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(labels, dim=1)
                all_preds.extend(pred_labels.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            sentiment_acc = sentiment_corrects.double() / len(dataloader.dataset)

            print(f'{phase} total loss: {epoch_loss:.4f}')
            print(f'{phase} sentiment_acc: {sentiment_acc:.4f}')
            print("\n Classification Report:")
            print(classification_report(all_labels, all_preds, digits=4))
            print("\n Confusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))

            # 保存训练过程中的统计数据（可选）
            sentiment_acc_cpu = sentiment_acc.cpu().numpy()
            if phase == 'train':
                train_acc.append(sentiment_acc_cpu)
                train_loss.append(epoch_loss)
            elif phase == 'val' and sentiment_acc > best_acc:
                print(f'Saving with accuracy of {sentiment_acc} improved over previous {best_acc}')
                best_acc = sentiment_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'bert_model_ori.pth')
            elif phase == 'val':
                val_acc.append(sentiment_acc_cpu)
                val_loss.append(epoch_loss)

        print('Time taken for epoch ' + str(epoch+1) + ' is ' + str((time.time() - epoch_start)/60) + ' minutes')
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
    # 加载最好的模型权重
    model.load_state_dict(best_model_wts)
    torch.save({
        'train_acc': train_acc,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
    },'bert_model_loss_acc_ASL.pth')
    return model, train_acc, val_acc, train_loss, val_loss


def main():
    num_labels = 6
    model = BertForSingleInputClassification(num_labels)
    model.to(device)

    lrlast = .0001
    lrmain = .00002
    optim1 = AdamW(
        [
            {"params": model.bert.parameters(), "lr": lrmain},
            {"params": model.classifier.parameters(), "lr": lrlast},
        ],
        weight_decay=0.01  # ⬅️ 建议加上权重衰减（BERT 默认是 0.01）
    )

    #optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)
    # Observe that all parameters are being optimized
    optimizer_ft = optim1
    criterion = nn.CrossEntropyLoss()

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    model_ft1, train_acc, val_acc, train_loss, val_loss = train_model(model, criterion, optimizer_ft, exp_lr_scheduler)
    test_model()

def test_model():
    num_labels = 6
    model = BertForSingleInputClassification(num_labels)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = load_dataset(train_path, test_path, val_path)
    
    model.load_state_dict(torch.load('bert_model_ori.pth'))
    model.eval()

    running_loss = 0.0
    sentiment_corrects = 0
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
      for input_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids)
            probs = F.softmax(outputs, dim=1)
          
            loss = criterion(outputs, labels.float())
          
            # statistics
            running_loss += loss.item() * input_ids.size(0)
            sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(labels, 1)[1])
            pred_labels = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(labels, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
            # all_preds.extend(pred_labels.cpu().numpy())
            # all_labels.extend(true_labels.cpu().numpy())
      
      epoch_loss = running_loss / len(test_loader.dataset)
      sentiment_acc = sentiment_corrects.double() / len(test_loader.dataset)
    print(f"\n🧪 Evaluation Results:")
    print(f"📉 Average Loss: {epoch_loss:.4f}")
    print(f"✅ Accuracy: {sentiment_acc * 100:.2f}%")

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    print("🔍 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    y_true_bin = label_binarize(all_labels, classes=np.arange(num_labels))

    # 计算 ROC AUC
    roc_auc = roc_auc_score(y_true_bin, np.array(all_probs), average="macro", multi_class="ovr")
    print(f"📊 ROC AUC (OvR Macro): {roc_auc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix_ASL.png", dpi=300)
    plt.show()

main()