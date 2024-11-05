import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Transformer Input Layer
class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=257, maxlen=2300, output_dim=40):
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(num_vocab, output_dim)
        self.pos_emb = nn.Embedding(maxlen, output_dim)

    def forward(self, x):
        maxlen = x.size(1)
        x = self.emb(x)
        positions = torch.arange(0, maxlen, device=x.device).unsqueeze(0)
        positions = self.pos_emb(positions)
        return x + positions



# Conv2D Subsampling Layer
class Conv2dSubsampling(nn.Module):
    def __init__(self, filters, output_dim, strides=1, kernel_size=3, padding = [0,2]):
        super(Conv2dSubsampling, self).__init__()
        self.inp_norm = nn.LayerNorm(20)
        self.conv1 = nn.Conv2d(1, filters, kernel_size=kernel_size, stride=strides, padding=padding)
        self.conv1_bn = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=strides, padding=padding)
        self.conv2_bn = nn.BatchNorm2d(filters)
        self.conv3 = nn.Conv1d(100, output_dim, kernel_size=1)
        self.conv3_bn = nn.BatchNorm1d(output_dim)
        self.pos_emb = nn.Embedding(2300, output_dim)
        self.flattenlayer = nn.Flatten(2,3)

    def forward(self, inputs):
        # Batch Norm the inputs
        inputs = self.inp_norm(inputs)
        inputs = inputs.unsqueeze(-1)
        inputs = inputs.permute((0, 3, 1, 2))
        #print(inputs.size())
        outputs = F.relu(self.conv1_bn(self.conv1(inputs)))
        outputs = F.relu(self.conv2_bn(self.conv2(outputs)))
        outputs = outputs.permute((0, 2, 3, 1))

        #print(outputs.size())
        outputs = self.flattenlayer(outputs)


        # Turn outputs feature dimension into required `output_dim`
        outputs = outputs.permute((0, 2, 1))
        outputs = F.relu(self.conv3_bn(self.conv3(outputs)))
        outputs = outputs.permute((0, 2, 1))


        maxlen = outputs.size(1)
        positions = torch.arange(0, maxlen, device=outputs.device).unsqueeze(0)
        positions = self.pos_emb(positions)
        outputs = outputs + positions
        return outputs


class CTCModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, dim_feedforward, nlayers, dropout=0.1):
        super(CTCModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=dim_feedforward, num_encoder_layers=nlayers, batch_first = True)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ink_encoder = Conv2dSubsampling(filters=20, output_dim = ninp, strides=[1,2], kernel_size=[1,5])

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, src_key_padding_mask, has_mask=True):
        if has_mask:
            if self.batch_first:
                seq_len = src.size(1)
            else:
                seq_len = len(src)
            #print(seq_len)
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != seq_len:
                mask = self._generate_square_subsequent_mask(seq_len).to(device)
                self.src_mask = mask
                #print(self.src_mask)
        else:
            self.src_mask = None

        #src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.ink_encoder(src)
        output = self.encoder(src, src_key_padding_mask = src_key_padding_mask, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
