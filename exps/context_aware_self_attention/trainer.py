import os
import gc
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from models.ContextAwareDAC import ContextAwareDAC
from data.dataset import DADataset

class DACModel(nn.Module):
    def __init__(self, config):
        super(DACModel, self).__init__()
        self.config = config
        self.model = ContextAwareDAC(
            model_name=self.config['model_name'],
            hidden_size=self.config['hidden_size'],
            num_classes=self.config['num_classes'],
            device=self.config['device']
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    def forward(self, batch):
        logits = self.model(batch)
        return logits

def get_dataloader(data_dir, dataset, split, tokenizer, max_len, text_field, label_field, batch_size, num_workers):
    csv_path = os.path.join(data_dir, dataset, f"{dataset}_{split}.csv")
    data = pd.read_csv(csv_path)
    data_dataset = DADataset(
        tokenizer=tokenizer,
        data=data,
        max_len=max_len,
        text_field=text_field,
        label_field=label_field
    )
    drop_last = True if len(data_dataset.text) % batch_size == 1 else False
    loader = DataLoader(
        dataset=data_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=drop_last
    )
    return loader

def train_epoch(model, loader, optimizer, device, config):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['label'].squeeze().to(device)
        
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': targets}
        
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average=config['average'])
    wandb.log({"train_loss": epoch_loss, "train_accuracy": acc, "train_f1": f1})
    return epoch_loss, acc, f1

def evaluate(model, loader, device, config, phase="val"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['label'].squeeze().to(device)
            batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': targets}
            logits = model(batch)
            loss = F.cross_entropy(logits, targets)
            running_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average=config['average'])
    precision = precision_score(all_targets, all_preds, average=config['average'])
    recall = recall_score(all_targets, all_preds, average=config['average'])
    
    wandb.log({
        f"{phase}_loss": epoch_loss, 
        f"{phase}_accuracy": acc, 
        f"{phase}_f1": f1,
        f"{phase}_precision": precision, 
        f"{phase}_recall": recall
    })
    return epoch_loss, acc, f1, precision, recall

def main(config):
    wandb.init(project=config['project_name'])
    device = torch.device(config['device'])
    
    model = DACModel(config)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    train_loader = get_dataloader(
        config['data_dir'], config['dataset'], "train", model.tokenizer, 
        config['max_len'], config['text_field'], config['label_field'],
        config['batch_size'], config['num_workers']
    )
    val_loader = get_dataloader(
        config['data_dir'], config['dataset'], "valid", model.tokenizer, 
        config['max_len'], config['text_field'], config['label_field'],
        config['batch_size'], config['num_workers']
    )
    test_loader = get_dataloader(
        config['data_dir'], config['dataset'], "test", model.tokenizer, 
        config['max_len'], config['text_field'], config['label_field'],
        config['batch_size'], config['num_workers']
    )
    
    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, device, config)
        print(f"Epoch {epoch+1}/{num_epochs} train_loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, f1: {train_f1:.4f}")
        
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(model, val_loader, device, config, phase="val")
        print(f"Epoch {epoch+1}/{num_epochs} val_loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, f1: {val_f1:.4f}, precision: {val_prec:.4f}, recall: {val_rec:.4f}")
    
    test_loss, test_acc, test_f1, test_prec, test_rec = evaluate(model, test_loader, device, config, phase="test")
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

