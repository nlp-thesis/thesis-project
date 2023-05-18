from transformers import BertTokenizer,BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle


def check_data(input_data_src):
    data = pd.read_excel(f"{input_data_src}")
    return data


def taken_tokens(max_len = 100):
    data = check_data("tweet.xlsx")
    X_train, X_test, y_train, y_test = train_test_split(data['Tweet'],
                                                    data['Segment'], test_size=0.3,
                                                    random_state=42)
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased')
    
    for tweet in X_train:
        input_ids = tokenizer.encode(tweet, add_special_tokens=True)
    print('Max tweet uzunluğu:', max_len)

    
    X_train_tokens = tokenizer.batch_encode_plus(
        X_train.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    X_test_tokens = tokenizer.batch_encode_plus(
        X_test.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    return X_train_tokens, X_test_tokens, y_train, y_test


def build_model(batch_size = 128):
    X_train_tokens, X_test_tokens, y_train, y_test = taken_tokens()
    X_train_seq = torch.tensor(X_train_tokens['input_ids'])
    X_train_mask = torch.tensor(X_train_tokens['attention_mask'])
    y_train = torch.tensor(y_train.tolist())

    X_test_seq = torch.tensor(X_test_tokens['input_ids'])
    X_test_mask = torch.tensor(X_test_tokens['attention_mask'])
    y_test = torch.tensor(y_test.tolist())

    train_data = TensorDataset(X_train_seq, X_train_mask, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    test_data = TensorDataset(X_test_seq, X_test_mask, y_test)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    

    model = BertForSequenceClassification.from_pretrained(
        "dbmdz/bert-base-turkish-uncased",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False
    )
    return train_dataloader, test_dataloader, model


def model(lr = 0.01, momentum = 0.9):
    train_dataloader, test_dataloader, model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    epochs = 7
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    model.train()


    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        running_loss = 0.0
        for batch in tqdm(train_dataloader, desc="Training"):
            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs[0]
            running_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        epoch_loss = running_loss / len(train_dataloader)
        #print(f'Training Loss: {epoch_loss:.4f}')
        
    model.eval()
    predictions = []
    true_labels = []
    for batch in tqdm(test_dataloader, desc="Testing"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        preds = torch.argmax(logits, dim=1).flatten()
        predictions.extend(preds)
        true_labels.extend(b_labels)
    predictions = torch.stack(predictions).cpu()
    true_labels = torch.stack(true_labels).cpu()
    print(classification_report(true_labels, predictions))

    with open('model.pkl', 'wb') as f:
        return pickle.dump(model, f)
    

if __name__ == '__main__':
    model()
