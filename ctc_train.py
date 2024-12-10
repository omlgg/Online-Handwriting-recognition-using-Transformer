import torch
from utils import levenshtein
from utils import VectorizeChar
import re
import jiwer


_COMMAND_RE = re.compile(r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)')


def tokenize_expression(s: str) -> list[str]:
  r"""Transform a Latex math string into a list of tokens.

  Tokens are strings that are meaningful in the context of Latex
  e.g. '1', r'\alpha', r'\frac'.

  Args:
    s: unicode input string (ex: r"\frac{1}{2}")

  Returns:
    tokens: list of tokens as unicode strings.
  """
  tokens = []
  while s:
    if s[0] == '\\':
      tokens.append(_COMMAND_RE.match(s).group(0))
    else:
      tokens.append(s[0])

    s = s[len(tokens[-1]) :]

  return tokens


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


def compute_cer(truth_and_output: list[tuple[str, str]]):
  """Computes CER given pairs of ground truth and model output."""
  class TokenizeTransform(jiwer.transforms.AbstractTransform):
    def process_string(self, s: str):
      return tokenize_expression(r'{}'.format(s))

    def process_list(self, tokens: list[str]):
      return [self.process_string(token) for token in tokens]

  ground_truth, model_output = zip(*truth_and_output)

  return jiwer.cer(truth=list(ground_truth),
            hypothesis=list(model_output),
            reference_transform=TokenizeTransform(),
            hypothesis_transform=TokenizeTransform(),
      )


def evaluate(model, data_loader, criterion, device, decoder = greedy_decoder, debug = True, padding = 0, kernel_size = 1, stride = 1):
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
        truth_and_output = []
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
                pred_sentence = decoder(output[i])
                truth_and_output.append((tgt_sentence, pred_sentence))

            output = output.permute(1, 0, 2) # (seq_len, batch_size, output_dimm)
            loss = criterion(output, tgt, input_lengths, output_lengths)
            epoch_loss += loss.item()

            # Remove unnecessary tensors
            del batch['src']
            del batch['tgt']
            del batch['src_msk']
            del batch['tgt_msk']
    print(f'CER: {compute_cer(truth_and_output)}')
    return epoch_loss / len(data_loader)
