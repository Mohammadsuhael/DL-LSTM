# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset


## DESIGN STEPS
### STEP 1: Collect and Prepare Dataset
Obtain a labeled text dataset where each word is tagged with its entity label (e.g., Person, Location, Organization, O).

### STEP 2: Text Preprocessing
Tokenize the sentences into words, convert them into numerical representations, and create vocabulary indices.

### STEP 3: Generate Word Embeddings
Convert each word into dense vectors using embedding techniques such as Word Embedding from libraries like Word2Vec or GloVe.

### STEP 4: Build the LSTM Model
Construct an LSTM-based neural network consisting of an embedding layer, one or more LSTM layers, and a dense layer with a softmax activation for entity classification.

### STEP 5: Train the Model
Train the LSTM using the labeled sequences to learn contextual relationships between words and their corresponding entity tags.

### STEP 6: Predict and Evaluate
Apply the trained model to new sentences to identify named entities and evaluate performance using metrics such as precision, recall, and F1-s


## PROGRAM

### Name: Mohammad Suhael

### Register Number: 212224230164

```python

class BiLSTMTagger(nn.Module):
    # Include your code here
    def __init__(self, vocab_size, target_size ,embedding_dim=50, hidden_dim=100):
      super(BiLSTMTagger, self).__init__()
      self.embedding = nn.Embedding(vocab_size, embedding_dim)
      self.dropout = nn.Dropout(0.2)
      self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
      self.fc = nn.Linear(hidden_dim * 2, target_size)


    def forward(self, input_ids):
        # Include your code here
        x=self.embedding(input_ids)
        x=self.dropout(x)
        x,_=self.lstm(x)
        x=self.fc(x)
        return x

model = BiLSTMTagger(len(word2idx) + 1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    # Include the training and evaluation functions
    train_losses, val_losses = [], []
    for epoch in range(epochs):
      model.train()
      train_loss = 0
      for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
      train_losses.append(train_loss)

      model.eval()
      val_loss = 0
      with torch.no_grad():
        for batch in test_loader:
          input_ids = batch["input_ids"].to(device)
          labels = batch["labels"].to(device)
          outputs = model(input_ids)
          loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
          val_loss += loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


```

### OUTPUT

## Loss Vs Epoch Plot
<img src="Screenshot 2026-03-15 160116.png"/><br>



### Sample Text Prediction
<img src="Screenshot 2026-03-15 161014.png"/><br>


## RESULT
Thus , LSTM-based model for recognizing the named entities in the text is developed successfully
