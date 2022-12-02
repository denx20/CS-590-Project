import torch
import torch.nn.functional as F

class SequenceEncoder(torch.nn.Module):
    def __init__(self, input_length = 10, output_length = 10, use_attention=False, num_layers=3, attn_dim=64, window_size=4) -> None:
        super(SequenceEncoder, self).__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.use_attention = use_attention
        self.window_size = window_size

        if use_attention:
            self.linear = torch.nn.Linear(2*window_size, attn_dim)
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=attn_dim, nhead=8, dim_feedforward=64, batch_first=True)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.mlp = torch.nn.Linear(attn_dim, output_length)
        
        else:
            mlp_layers = [torch.nn.Linear(2*input_length, output_length),torch.nn.ReLU()] \
                    + [torch.nn.Linear(output_length, output_length),torch.nn.ReLU()]*(num_layers-2) \
                    + [torch.nn.Linear(output_length, output_length)]
            self.mlp = torch.nn.Sequential(
                *mlp_layers
            )

    def sliding_window(self, x, eps=1e-7):
        # x is of shape (batch_size, input_length)
        # returns a tensor of shape (batch_size, input_length-self.window_size+1, self.window_size*2)

        x = x.unfold(0,self.window_size,1) # x has shape (batch_size, input_length-self.window_size+1, self.window_size)
        log_x = self.safe_log(x.abs())

        # normalize within each window
        m = x.mean(-1, keepdim=True)
        s = x.std(-1, unbiased=False, keepdim=True) + eps
        x -= m
        x /= s
        
        x = torch.cat([x, log_x], dim=1)
        return x

    def normalize(self, x, eps=1e-7):
        # x has shape (batch_size, )
        m = x.mean(0, keepdim=True)
        s = x.std(0, unbiased=False, keepdim=True) + eps
        x -= m
        x /= s
        return x

    def safe_log(self, x, eps=1e-7):
        # so that log doesn't go to 0 when applied twice
        x = F.relu(x)
        x = torch.log(x + eps)
        return x

    def forward(self, x):
        # Concatenate x with log(x)
        if self.use_attention:
            x = self.sliding_window(x)
            x = self.linear(x)
            x = x.unsqueeze(0)
            x = self.transformer_encoder(x)
            x = x.mean(1)
            return self.mlp(x)
            
        else: 
            augmented_tensor = x.repeat((1, 2)) # x is of shape [batch_size, input_length]

            augmented_tensor[:, 0 : self.input_length] = self.normalize(x)
            augmented_tensor[:, self.input_length : self.input_length * 2] = self.safe_log(x.abs())
            augmented_tensor = self.mlp(augmented_tensor)

            return augmented_tensor
        
    def forward_output_all(self, x):
        if not self.use_attention:
            raise Exception('Only call this function if using attention in encoder')
        x = self.sliding_window(x)
        x = self.linear(x)
        x = x.unsqueeze(0)
        x = self.transformer_encoder(x)
        return x

class MCTS_MLP(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, input_seq_size, embedding_weights=None, encoder_attn=False):
        super(MCTS_MLP, self).__init__()
        
        # sequence encoder
        self.input_seq_size = input_seq_size
        self.seq_encoder = SequenceEncoder(input_seq_size, embed_size, use_attention=encoder_attn)
        
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
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, input_seq_size, embedding_weights=None, encoder_attn=False):
        super(MCTS_GRU, self).__init__()
        
        # sequence encoder
        self.input_seq_size = input_seq_size
        self.seq_encoder = SequenceEncoder(input_seq_size, embed_size, use_attention=encoder_attn)
        
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

# TODO
# Set Transformer
# Reference: https://github.com/juho-lee/set_transformer/blob/master/models.py
class MCTS_Transformer(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, input_seq_size, embedding_weights=None):
        super(MCTS_Transformer, self).__init__()

        # sequence encoder
        self.input_seq_size = input_seq_size
        self.seq_encoder = SequenceEncoder(input_length=input_seq_size, attn_dim=hidden_size, use_attention=True)
        
        # embedding layer
        if embedding_weights:
            assert embed_size == embedding_weights.shape[-1]
            self.embedder = torch.nn.Embedding().from_pretrained(embedding_weights)
        else:
            self.embedder = torch.nn.Embedding(vocab_size, embed_size)
        
        self.linear = torch.nn.Linear(embed_size, hidden_size)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True, dim_feedforward=64)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # after encoder-decoder transformer
        self.choice_head = torch.nn.Linear(hidden_size, vocab_size)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, x, seq):
        encoded_tokens = self.seq_encoder.forward_output_all(seq)
        tgt = self.linear(self.embedder(x))
        output = self.decoder(tgt, encoded_tokens).mean(1)
        return self.softmax(self.choice_head(output)), self.value_head(output)

    