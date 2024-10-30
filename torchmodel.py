import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Transformer Input Layer
class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=300, maxlen=100, num_hid=40):
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(num_vocab, num_hid)
        self.pos_emb = nn.Embedding(maxlen, num_hid)

    def forward(self, x):
        maxlen = x.size(1)
        x = self.emb(x)
        positions = torch.arange(0, maxlen, device=x.device).unsqueeze(0)
        positions = self.pos_emb(positions)
        return x + positions

# Helper functions
def shape_list(x):
    return list(x.size())

def merge_two_last_dims(x):
    b, t, f, c = shape_list(x)
    return x.view(b, t, -1)

def expand_last_dim(x):
    return x.unsqueeze(-1)

# Conv2D Subsampling Layer
class Conv2dSubsampling(nn.Module):
    def __init__(self, filters, strides=2, kernel_size=3):
        super(Conv2dSubsampling, self).__init__()
        self.conv1 = nn.Conv2d(1, filters, kernel_size=kernel_size, stride=strides, padding=1)
        self.conv1_bn = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=strides, padding=1)
        self.conv2_bn = nn.BatchNorm2d(filters)
        self.pos_emb = nn.Embedding(2300, 100)
        
    def forward(self, inputs):
        inputs = expand_last_dim(inputs)
        outputs = F.relu(self.conv1_bn(self.conv1(inputs)))
        outputs = F.relu(self.conv2_bn(self.conv2(outputs)))
        outputs = merge_two_last_dims(outputs)
        maxlen = outputs.size(1)
        positions = torch.arange(0, maxlen, device=outputs.device).unsqueeze(0)
        positions = self.pos_emb(positions)
        outputs = outputs + positions
        return outputs

# Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.0):
        super(TransformerEncoder, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(rate)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        out1 = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout(ffn_output))

# Transformer Decoder Layer
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.3):
        super(TransformerDecoder, self).__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.self_att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.enc_att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim)
        )
        
    def forward(self, enc_out, target):
        tgt_len = target.size(1)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=target.device)).unsqueeze(0)
        target_att, _ = self.self_att(target, target, target, attn_mask=causal_mask)
        target_norm = self.layernorm1(target + target_att)
        enc_attn_out, _ = self.enc_att(target_norm, enc_out, enc_out)
        enc_out_norm = self.layernorm2(target_norm + enc_attn_out)
        ffn_out = self.ffn(enc_out_norm)
        return self.layernorm3(enc_out_norm + ffn_out)

# Transformer Model
class Transformer(nn.Module):
    def __init__(
        self, num_hid=100, num_head=2, num_feed_forward=128, source_maxlen=1000,
        target_maxlen=100, num_layers_enc=4, num_layers_dec=1, num_classes=10
    ):
        super(Transformer, self).__init__()
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = Conv2dSubsampling(filters=20, strides=2, kernel_size=5)
        self.dec_input = TokenEmbedding(num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid)
        
        self.encoder = nn.ModuleList(
            [TransformerEncoder(num_hid, num_head, num_feed_forward) for _ in range(num_layers_enc)]
        )
        
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoder(num_hid, num_head, num_feed_forward) for _ in range(num_layers_dec)]
        )
        
        self.classifier = nn.Linear(num_hid, num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for dec_layer in self.decoder_layers:
            y = dec_layer(enc_out, y)
        return y

    def forward(self, source, target):
        x = self.enc_input(source)
        for enc_layer in self.encoder:
            x = enc_layer(x)
        y = self.decode(x, target)
        return self.classifier(y)
