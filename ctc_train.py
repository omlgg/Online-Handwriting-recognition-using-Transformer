import torch

def train(model, data_loader, optimizer, criterion, device, debug = False, verbose_freq = 300):
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

def evaluate(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0

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
            output_lengths = torch.sum(dim = 1, input = torch.exp(tgt_msk), dtype = torch.int32) - 2

            output = model(src, src_msk)  # (batch_size, seq_len, output_dim)
            output = output.permute(1, 0, 2) # (seq_len, batch_size, output_dimm)

            tgt = tgt[:, 1:-1]

            loss = criterion(output, tgt, input_lengths, output_lengths)
            epoch_loss += loss.item()

            # Remove unnecessary tensors
            del batch['src']
            del batch['tgt']
            del batch['src_msk']
            del batch['tgt_msk']

    return epoch_loss / len(data_loader)
