import torch
from utils import levenshtein
from utils import VectorizeChar

vectorizer = VectorizeChar(257)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined


greedy_decoder = GreedyCTCDecoder(vectorizer.get_vocabulary())
vocab = vectorizer.get_vocabulary()

def train(model, data_loader, optimizer, criterion, device, debug = False, verbose_freq = 300, padding = 0, kernel_size = 1, stride = 1):
    model.train()
    epoch_loss = 0
    step_cnt = 0

    for batch in data_loader:
        src = batch['src']
        tgt = batch['tgt']
        src_msk = batch['src_msk']
        tgt_msk = batch['tgt_msk']
        src_msk = src_msk.to(device)
        tgt_msk = tgt_msk.to(device)
        src, tgt = src.to(device), tgt.to(device)
        input_lengths = torch.sum(dim = 1, input = torch.exp(src_msk), dtype = torch.int32)
        output_lengths = torch.sum(dim = 1, input = torch.exp(tgt_msk), dtype = torch.int32) - 2
        for i in range(2):
            input_lengths = (input_lengths + 2*padding - kernel_size)//stride + 1

        optimizer.zero_grad()
        if debug:
            with torch.autograd.detect_anomaly():
                output = model(src, src_msk)  # (batch_size, seq_len, output_dim)
                output = output.permute(1, 0, 2) # (seq_len, batch_size, output_dimm)

                tgt = tgt[:, 1:-1]
                #tgt = tgt[1:, :].view(-1)  # Exclude <sos> token

                loss = criterion(output, tgt, input_lengths, output_lengths)
                loss.backward()
                optimizer.step()

        else:
            output = model(src, src_msk)  # (batch_size, seq_len, output_dim)
            output = output.permute(1, 0, 2) # (seq_len, batch_size, output_dimm)

            tgt = tgt[:, 1:-1]
            #tgt = tgt[1:, :].view(-1)  # Exclude <sos> token

            loss = criterion(output, tgt, input_lengths, output_lengths)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        step_cnt+=1

        if step_cnt % verbose_freq == 0:
            print(f'{step_cnt}-th step, loss: {loss.item()}')
            torch.cuda.empty_cache()

        # Remove unnecessary tensors
        del batch['src']
        del batch['tgt']
        del batch['src_msk']
        del batch['tgt_msk']

    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device, debug = True, padding = 0, kernel_size = 1, stride = 1):
    printed = not debug
    # works with batch size = 1
    model.eval()
    epoch_loss = 0
    total_char = 0
    error_char = 0

    with torch.no_grad():
        # for src, tgt in data_loader:
        #     src, tgt = src.to(device), tgt.to(device)
        #     tgt_input = tgt[:-1, :]

        #     output = model(src, tgt_input)
        #     output_dim = output.shape[-1]
        #     output = output[1:, :, :].view(-1, output_dim)
        #     tgt = tgt[1:, :].view(-1)

        #     loss = criterion(output, tgt)
        #     epoch_loss += loss.item()

        for batch in data_loader:
            src = batch['src']
            tgt = batch['tgt']
            src_msk = batch['src_msk']
            tgt_msk = batch['tgt_msk']
            src_msk = src_msk.to(device)
            tgt_msk = tgt_msk.to(device)
            src, tgt = src.to(device), tgt.to(device)
            input_lengths = torch.sum(dim = 1, input = torch.exp(src_msk), dtype = torch.int32)
            for i in range(2):
                input_lengths = (input_lengths + 2*padding - kernel_size)//stride + 1
            output_lengths = torch.sum(dim = 1, input = torch.exp(tgt_msk), dtype = torch.int32) - 2

            output = model(src, src_msk)  # (batch_size, seq_len, output_dim)

            if not printed:
                print(tgt[0])

            tgt = tgt[:, 1:-1]
            if not printed:
                print(tgt[0])

            for i in range(output.shape[0]):
                tgt_sentence = "".join([vocab[j] for j in tgt[i]])
                if not printed:
                    print(tgt_sentence)
                    printed = True
                total_char += len(tgt_sentence)
                error_char += levenshtein(greedy_decoder(output[i]), tgt_sentence)

            output = output.permute(1, 0, 2) # (seq_len, batch_size, output_dimm)
            loss = criterion(output, tgt, input_lengths, output_lengths)
            epoch_loss += loss.item()

            # Remove unnecessary tensors
            del batch['src']
            del batch['tgt']
            del batch['src_msk']
            del batch['tgt_msk']
    print(f'CER: {error_char/total_char}')
    return epoch_loss / len(data_loader)
