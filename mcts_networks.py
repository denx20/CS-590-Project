import torch
import torch.nn.functional as F

class TermPredictor(torch.nn.Module):
    # current version: Z^10 -> F_2^10
    # represented as floats
    def __init__(self, input_length = 10, output_length = 10, use_attention=False) -> None:
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.use_attention = use_attention

        if use_attention:
            self.a1 = torch.nn.MultiheadAttention(3*self.input_length, 1)
            self.a2 = torch.nn.MultiheadAttention(3*self.input_length, 1)
            self.a3 = torch.nn.MultiheadAttention(3*self.input_length, 1)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3*self.input_length, output_length),
            torch.nn.ReLU(),
            torch.nn.Linear(output_length, output_length),
            torch.nn.ReLU(),
            torch.nn.Linear(output_length, output_length),
        )

    def safe_log(self, x, eps=1e-7):
        # so that log doesn't go to 0 when applied twice
        x = F.relu(x)
        x = torch.log(x + eps)
        return x

    def forward(self, x):
        # Concatenate x with log(x) and log(log(x))
        # TODO: fix this
        augmented_tensor = x.repeat((1, 3)) # x is of shape [batch_size, input_length]

        augmented_tensor[:, 0 : self.input_length] = x
        augmented_tensor[:, self.input_length : self.input_length * 2] = self.safe_log(x)
        augmented_tensor[:, self.input_length * 2: self.input_length * 3] = self.safe_log(self.safe_log(x))

        if self.use_attention:
            augmented_tensor = self.a1(augmented_tensor, augmented_tensor, augmented_tensor)[0]
            augmented_tensor = self.a2(augmented_tensor, augmented_tensor, augmented_tensor)[0]
            augmented_tensor = self.a3(augmented_tensor, augmented_tensor, augmented_tensor)[0]

        augmented_tensor = self.mlp(augmented_tensor)

        return augmented_tensor


class MCTS_MLP(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, input_seq_size, embedding_weights=None):
        super(MCTS_MLP, self).__init__()
        
        # sequence encoder
        self.input_seq_size = input_seq_size
        self.seq_encoder = TermPredictor(input_seq_size, embed_size)
        
        self.embed_size = embed_size  
        # embedding layer
        if embedding_weights:
            assert embed_size == embedding_weights.shape[-1]
            self.embedder = torch.nn.Embedding().from_pretrained(embedding_weights)
        else:
              self.embedder = torch.nn.Embedding(vocab_size, embed_size)
        
        # MLP
        mlp_layers = [torch.nn.Linear(2*embed_size, hidden_size),torch.nn.ReLU()] \
                    + [torch.nn.Linear(hidden_size, hidden_size),torch.nn.ReLU()]*(num_layers-2) \
                    + [torch.nn.Linear(hidden_size, hidden_size)]
        self.mlp = torch.nn.Sequential(
            *mlp_layers
        )
        
        # after MLP
        self.choice_head = torch.nn.Linear(hidden_size, vocab_size)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, seq):
        # x: index of function terms
        # seq: target sequence
        
        assert seq.shape[-1] == self.input_seq_size, f'Input sequence length is invalid! Expected{self.input_seq_size}, got {seq.shape[-1]}.'
        
        encoded_seq = self.seq_encoder(seq)
        if encoded_seq.shape != (1,self.embed_size):
            encoded_seq = encoded_seq.reshape(1,self.embed_size)
        
        terms_embedding = self.embedder(x).mean(dim=1)
        
        mlp_input = torch.cat([encoded_seq, terms_embedding], dim=-1)
        mlp_output = self.mlp(mlp_input)
        return self.softmax(self.choice_head(mlp_output)), self.value_head(mlp_output)


class MCTS_GRU(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, input_seq_size, embedding_weights=None):
        super(MCTS_GRU, self).__init__()
        
        # sequence encoder
        self.input_seq_size = input_seq_size
        self.seq_encoder = TermPredictor(input_seq_size, embed_size)
        
        self.embed_size = embed_size  
        # embedding layer
        if embedding_weights:
            assert embed_size == embedding_weights.shape[-1]
            self.embedder = torch.nn.Embedding().from_pretrained(embedding_weights)
        else:
              self.embedder = torch.nn.Embedding(vocab_size, embed_size)
        
        # GRU
        self.gru = torch.nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        
        # after GRU
        self.choice_head = torch.nn.Linear(hidden_size, vocab_size)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, seq):
        # x: index of function terms
        # seq: target sequence
        
        assert seq.shape[-1] == self.input_seq_size, f'Input sequence length is invalid! Expected{self.input_seq_size}, got {seq.shape[-1]}.'
        
        encoded_seq = self.seq_encoder(seq)
        if encoded_seq.shape != (1,1,self.embed_size):
            encoded_seq = encoded_seq.reshape(1,1,self.embed_size)
        
        gru_input = torch.cat([encoded_seq, self.embedder(x)], dim=1)
        gru_output = self.gru(gru_input)[1][-1]
        return self.softmax(self.choice_head(gru_output)), self.value_head(gru_output)
    