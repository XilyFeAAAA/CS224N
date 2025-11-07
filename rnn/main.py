from pathlib import Path
import numpy as np
import torch
import tqdm
import time

class RNN(torch.nn.Module):
    
    def __init__(
        self,
        embeddings,
        embed_size,
        hidden_size,
        output_size
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.hidden_size = hidden_size

        
        # xt = [batch_size, seq_len * embed_size]
        # Wx = [seq_len * embed_size, hidden_size]
        # Wh = [batch_size, hidden_size]
        self.embed_to_hidden_weight = torch.nn.Parameter(torch.empty(embed_size, hidden_size))
        torch.nn.init.xavier_uniform_(self.embed_to_hidden_weight)
        
        self.hidden_to_logits_weight = torch.nn.Parameter(torch.empty(hidden_size, output_size))
        torch.nn.init.xavier_uniform_(self.hidden_to_logits_weight)
        
        self.last_to_new_weight = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        torch.nn.init.xavier_uniform_(self.last_to_new_weight)
        
        embed_to_hidden_bias_tensor = torch.empty(hidden_size)
        torch.nn.init.uniform_(embed_to_hidden_bias_tensor)
        self.embed_to_hidden_bias = torch.nn.Parameter(embed_to_hidden_bias_tensor)

        hidden_to_logits_bias_tensor = torch.empty(output_size)
        torch.nn.init.uniform_(hidden_to_logits_bias_tensor)
        self.hidden_to_logits_bias = torch.nn.Parameter(hidden_to_logits_bias_tensor)
    
    def forward(self, w):
        x = self.embedding_lookup(w)
        batch_size, seq_len, _ = x.shape
        ht = torch.zeros(batch_size, self.hidden_size)
        for i in range(seq_len):
            inp = x[:, i, :]
            ht = torch.tanh(torch.matmul(inp, self.embed_to_hidden_weight) + torch.matmul(ht, self.last_to_new_weight) + self.embed_to_hidden_bias)
        
        return torch.matmul(ht, self.hidden_to_logits_weight) + self.hidden_to_logits_bias
    
    
    def embedding_lookup(self, w):
        # [batch_size, seq_len] -> [batch_size, seq_len, embed_size]
        return self.embeddings[w]



class Predictor:
    
    def __init__(
        self, 
        train_filepath: str,
        test_filepath: str,
        save_filepath: str
    ) -> None:
        self.train_filepath = Path(train_filepath)
        self.test_filepath = Path(test_filepath)
        self.save_filepath = Path(save_filepath)
        self.config = {
            "epochs": 50,
            "learning_rate": 0.005,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        labels = torch.tensor([i for i in range(26)])
        self.embeddings = torch.nn.functional.one_hot(labels, len(labels)).float()
        self.model = RNN(
            embeddings=self.embeddings,
            embed_size=6,
            hidden_size=48,
            output_size=26
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), 
            lr=self.config["learning_rate"]
        )
    
    def load(self):
        assert self.train_filepath.exists(), "训练集文件不存在"
        assert self.test_filepath.exists(), "测试集文件不存在"
        
        with open(self.train_filepath, "r") as f:
            lines = f.readlines()
            train_dataset = [line.split() for line in lines]
        train_dataset = torch.tensor([[ord(ch) - ord('a') for ch in line] for line in train_dataset])
        self.train_x, self.train_y = train_dataset[:, :-1], train_dataset[:, -1]
        
        with open(self.test_filepath, "r") as f:
            lines = f.readlines()
            test_dataset = [line.split() for line in lines]
        test_dataset = torch.tensor([[ord(ch) - ord('a') for ch in line] for line in test_dataset])
        self.test_x, self.test_y = test_dataset[:, :-1], test_dataset[:, -1]

    
    def train(self):

        self.model.train()
        
        for epoch in tqdm.tqdm(range(self.config["epochs"])):
            self.optimizer.zero_grad()
            pred = self.model(self.train_x)
            loss = self.criterion(pred, self.train_y)
            loss.backward()
            self.optimizer.step()
            print(f"第{epoch+1}/{self.config['epochs']+1}轮训练,loss={loss}")
        
            
    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.test_x)
            loss = self.criterion(pred, self.test_y)
        print(pred.argmax(dim=1))
        print(self.test_y)
        print(f"测试集loss={loss}")



if __name__ == "__main__":
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    start = time.time()
    predictor = Predictor(
        train_filepath=r"D:\dev\github\CS224N\rnn\data\dev.txt",
        test_filepath=r"D:\dev\github\CS224N\rnn\data\test.txt",
        save_filepath="."
    )
    predictor.load()
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    predictor.train()

    print(80 * "=")
    print("TESTING")
    print(80 * "=")
    predictor.test()
    print("Done!")
