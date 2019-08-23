import torch
from torch import optim
from torch import nn
import random
import time
from dataset import MAX_LENGTH ,SOS_token,EOS_token,PAD_token
from util import AttnDecoderRNN , Encoder
from dataset import tensors_from_pair, prepare_data
from utils import time_since
from evaluate import evaluate_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256



def train_iteration(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_func):
    # Zero the model gradients.
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    try:
        input_length = input_tensor.index(0)-1
    except:
        input_length = len(input_tensor)-1
    target_length = target_tensor.index(0)
    loss = 0
    try:
        input_tensor = torch.tensor(input_tensor, dtype=torch.long).view(1, MAX_LENGTH).to(device)
        input_length = torch.tensor(input_length, dtype=torch.long).view(-1, 1).to(device)
        target_tensor = torch.tensor(target_tensor, dtype=torch.long).view(1, MAX_LENGTH).to(device)


        output, _ = encoder(input_tensor,  torch.tensor(MAX_LENGTH, dtype=torch.long).view(-1, 1).to(device))

        decoder_input = torch.tensor([[SOS_token]], device=device)
        output = output[0]
        decoder_hidden = output[input_length].view(1,1, hidden_size).to(device)
        use_teacher_forcing = False

        # Decoder.
        if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
            for di in range(1,target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, output)

                loss += loss_func(decoder_output, target_tensor[0][di].view(1))
                decoder_input = target_tensor[0][di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(1,target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, output)
                _, top_index = decoder_output.topk(1)
                decoder_input = top_index.squeeze().detach()  # detach from history as input
                loss += loss_func(decoder_output, target_tensor[0][di].view(1))
                if decoder_input.item() == EOS_token:
                    break
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
    except:
        return 0
    return loss.item() / target_length


def train(encoder, decoder, n_iters, pairs, input_lang, output_lang, print_every=1000, plot_every=1000, learning_rate=0.01):
    print('train begin:')
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # Randomly sample pairs from training set.
    training_pairs = []
    print('data trans')
    for i in range(n_iters):
        lang1_sample, lang2_sample= tensors_from_pair(random.choice(pairs), input_lang, output_lang)
        training_pairs.append((lang1_sample, lang2_sample))
    print('over')
    loss_func = nn.NLLLoss()

    for iter_ in range(1, n_iters + 1):
        training_pair = training_pairs[iter_ - 1]  # Get a training pair.
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train_iteration(input_tensor, target_tensor, encoder,
                               decoder, encoder_optimizer, decoder_optimizer, loss_func)
        print_loss_total += loss
        plot_loss_total += loss
        if iter_ % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter_ / n_iters),
                                         iter_, iter_ / n_iters * 100, print_loss_avg))

        if iter_ % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses


def main():
    input_lang, output_lang, pairs = prepare_data('ques', 'ans', '../data.json',reverse=False)
    encoder = Encoder(input_lang.n_words, MAX_LENGTH).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH).to(device)


    rate = 0.9
    epoch = 10
    pairs_train,pairs_test = pairs[0:int(len(pairs)*rate)], pairs[int(len(pairs)*rate):]
    for i in range(epoch):
        encoder.train()
        attn_decoder.train()
        train(encoder, attn_decoder, len(pairs_train), pairs=pairs_train, input_lang=input_lang,output_lang=output_lang, print_every=10)
        encoder.eval()
        attn_decoder.eval()
        evaluate_all(encoder, attn_decoder, pairs_test, max_length=MAX_LENGTH, input_lang=input_lang, output_lang=output_lang,
                         n=len(pairs_test))
        torch.save(encoder.state_dict(), 'model/encoder-' + str(i) + '.model')
        torch.save(attn_decoder.state_dict(), 'model/decoder-' + str(i) + '.model')
    #show_plot(loss_history)
    print('done training')


if __name__ == "__main__":
    main()
