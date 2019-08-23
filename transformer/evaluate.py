import torch
import random
import numpy 
from dataset import SOS_token, EOS_token,MAX_LENGTH,PAD_token
from dataset import tensor_from_sentence
from utils import show_attention
from util import padding_mask
hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, max_length, input_lang, output_lang):
    with torch.no_grad():

        input_tensor = tensor_from_sentence(input_lang, sentence)
        try:
           input_length = input_tensor.index(0)-1
        except:
           input_length = len(input_tensor)-1
        input_tensor = torch.tensor(input_tensor, dtype=torch.long).view(1, MAX_LENGTH).to(device)
        input_length = torch.tensor(input_length, dtype=torch.long).view(-1, 1).to(device)
        decoded_words = [[SOS_token]]
        decoded_result = [torch.tensor([SOS_token], device=device)]


        output, _ = encoder(input_tensor, torch.tensor(MAX_LENGTH, dtype=torch.long).view(-1, 1).to(device))
        decoder_input = torch.tensor([[SOS_token]], device=device)
        output = output[0]
        decoder_hidden = output[input_length].view(1, 1, hidden_size).to(device)

        decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, output)
            decoder_attentions[di] = decoder_attention.data
            _, top_index = decoder_output.data.topk(1)
            if top_index.item() == EOS_token:
                decoded_words.append('<EOS>')
                decoded_result.append(torch.tensor([[EOS_token]], device=device))
                break
            else:
                decoded_words.append(output_lang.index2word[top_index.item()])

            decoder_input = top_index.squeeze().detach()
            decoded_result.append(decoder_input)

        return decoded_words, decoder_attentions[:di + 1],decoded_result
def cal_acc(pre_seq, true_seq):
    count = 0
    # print(true_seq)
    #print(pre_seq)
    for i in range(1, list(true_seq).index(0)-2):
        try:

            if pre_seq[i].item() == true_seq[i]:
                count += 1
        except:
            continue
    return count / (list(true_seq).index(0)-2)
    
def evaluate_all(encoder, decoder, pairs, max_length, input_lang, output_lang, n=1):
    ALL_ACC = 0
    for i in range(n):
        #pair = random.choice(pairs)
        pair = pairs[i]
        #print('question:', pair[0])
        #print('answer:', pair[1])
        output_words, attentions, output = evaluate(encoder, decoder, pair[0], max_length, input_lang, output_lang)
        output_sentence = ' '.join(output_words[1:-1])
        pre_seq = numpy.array(output)
        true_seq = numpy.array(tensor_from_sentence(output_lang,pair[1]))
        ALL_ACC += cal_acc(pre_seq,true_seq)
        #print('predict:', output_sentence)
        #print('')
    print("ACC:" + str(ALL_ACC/n))


def evaluate_and_show_attention(input_sentence, encoder, attn_decoder, max_length, input_lang, output_lang):
    output_words, attentions,_ = evaluate(
        encoder, attn_decoder, input_sentence, max_length, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)
