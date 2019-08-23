import torch
from dataset import prepare_data
from transformer import Transformer
from evaluate import evaluate_all
from dataset import MAX_LENGTH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256
teacher_forcing_ratio = 0.5


def main():
    input_lang, output_lang, pairs = prepare_data('ques', 'ans', '../test.json', reverse=False)
    model = Transformer( src_vocab_size = input_lang.n_words,
                         src_max_len = MAX_LENGTH,
                         tgt_vocab_size = output_lang.n_words,
                         tgt_max_len = MAX_LENGTH,).to(device)


    rate = 0.9
    pairs_train, pairs_test = pairs[0:int(len(pairs) * rate)], pairs[int(len(pairs) * rate):]
    model.load_state_dict(torch.load('model/transformer-0.model'))
    model.eval()
    evaluate_all(model, pairs_train, max_length=100, input_lang=input_lang, output_lang=output_lang,
                 n=len(pairs_train))
    # show_plot(loss_history)
    print('done test')


if __name__ == "__main__":
    main()
