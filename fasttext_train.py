import fasttext
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# model = fasttext.train_supervised("fasttext-train.txt")
# model.save_model("fasttext.model")


model = fasttext.load_model("fasttext.model")

df = pd.read_csv('d-X-test.csv')
X_test = df.values
df = pd.read_csv('d-y-test.csv')
y_test = df.values

y_pred = []
for i in tqdm(range(X_test.shape[0])):
    tmp = [str(X_test[i][j]) for j in range(X_test.shape[1])]
    text = " ".join(tmp)
    y_pred.append(int(model.predict(text)[0][0][-1]))

print(precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))
