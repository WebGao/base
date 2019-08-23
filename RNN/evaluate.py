import torch
import random
import numpy 
from dataset import SOS_token, EOS_token
from dataset import tensor_from_sentence
from utils import show_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(encoder, decoder, sentence, max_length, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence).to(device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden().to(device)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = ['<SOS>']
        decoder_attentions = torch.zeros(max_length, max_length)
        decoded_result = [decoder_input]
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            _, top_index = decoder_output.data.topk(1)
            #if top_index.item() == EOS_token:
            #    decoded_words.append('<EOS>')
            #    break
            #else:
            #    decoded_words.append(output_lang.index2word[top_index.item()])

            decoder_input = top_index.squeeze().detach()
            decoded_result.append(decoder_input)
            if top_index.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[top_index.item()])

        return decoded_words, decoder_attentions[:di + 1],decoded_result

def cal_acc(pre_seq,true_seq):
    count = 0
    for i in range(1,len(true_seq)-1):
        try:
           
            if pre_seq[i].item() == true_seq[i]:
                count+=1
        except:
            continue
    return count/ (len(true_seq)-2)
        
    
def evaluate_all(encoder, decoder, pairs, max_length, input_lang, output_lang, n=1):
    ALL_ACC = 0
    for i in range(n):
        #pair = random.choice(pairs)
        pair = pairs[i]
        #print('question:', pair[0])
        #print('answer:', pair[1])
        output_words, attentions, output = evaluate(encoder, decoder, pair[0], max_length, input_lang, output_lang)
        output_sentence = ' '.join(output_words)
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
