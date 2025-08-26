# Ex-2: Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="832" height="834" alt="image" src="https://github.com/user-attachments/assets/19e485d3-530b-4254-a310-3abee161ca74" />


## DESIGN STEPS

### STEP 1:
Import necessary libraries and load the dataset.

### STEP 2:
Encode categorical variables and normalize numerical features.

### STEP 3:
Split the dataset into training and testing subsets.

### STEP 4:
Design a multi-layer neural network with appropriate activation functions.

### STEP 5:
Train the model using an optimizer and loss function.

### STEP 6:
Evaluate the model and generate a confusion matrix.

### STEP 7:
Use the trained model to classify new data samples.

### STEP 8:
Display the confusion matrix, classification report, and predictions.


## PROGRAM


```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16,8)
        #self.fc3 = nn.Linear(16, 8)
        self.fc3=nn.Linear(8,4) #4 output



    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      #x=F.relu(self.fc3(x))
      x=self.fc3(x) #no activation
      return x
        

```
```python
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size= X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):

  for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      outputs = model(X_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()




    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```



## Dataset Information

<img width="1232" height="235" alt="image" src="https://github.com/user-attachments/assets/6ce3c2d2-1318-4e07-8ad2-08cb2cc27f10" />


## OUTPUT



### Confusion Matrix

<img width="703" height="572" alt="image" src="https://github.com/user-attachments/assets/530987e4-2087-4941-a1a1-0b2377016a40" />


### Classification Report

<img width="557" height="377" alt="481924628-1a0ffd67-6cd8-4877-939d-0d4371fa5a95" src="https://github.com/user-attachments/assets/99d4631a-8f45-4123-8a35-58a067e95f38" />




### New Sample Data Prediction

<img width="1007" height="386" alt="pi" src="https://github.com/user-attachments/assets/27a8479e-6172-4987-89ca-702f25e6f5da" />


## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
