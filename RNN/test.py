import torch
from dataset import prepare_data
from model import AttnDecoderRNN, EncoderRNN
from evaluate import evaluate_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256
teacher_forcing_ratio = 0.5


def main():
    input_lang, output_lang, pairs = prepare_data('ques', 'ans', '../debug.json', reverse=False)
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=1000).to(device)

    rate = 0.9
    pairs_train, pairs_test = pairs[0:int(len(pairs) * rate)], pairs[int(len(pairs) * rate):]
    encoder.load_state_dict(torch.load('model/encoder-0.model'))
    encoder.eval()
    attn_decoder.load_state_dict(torch.load('model/decoder-0.model'))
    attn_decoder.eval()
    evaluate_all(encoder, attn_decoder, pairs_test, max_length=1000, input_lang=input_lang,
                          output_lang=output_lang, n = len(pairs_test))
    # show_plot(loss_history)
    print('done test')


if __name__ == "__main__":
    main()
