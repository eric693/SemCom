import os, platform, json, time, pickle, sys, argparse
import torch
from math import log
sys.path.append('./')
from data_loader import Dataset_sentence_test, collate_func
from model import LSTMEncoder, LSTMDecoder, Embeds
from utils import Normlize_tx, Channel, smaple_n_times

import logging
import datetime
import random
import re

import os, platform, json, time, pickle, sys, argparse
import torch
from math import log
sys.path.append('./')
from data_loader import Dataset_sentence_test, collate_func
from model import LSTMEncoder, LSTMDecoder, Embeds
from utils import Normlize_tx, Channel, smaple_n_times
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction


_snr = 20
_iscomplex = True
channel_dim = 8




def float_to_fixed_point_sign_5_5_6(x):
    # 設定位元數量
    sign_bits = 13
    int_bits = 13
    frac_bits = 6
    
    # 計算上下限
    max_val = 2 ** (int_bits + frac_bits - 1) - 2 ** frac_bits
    min_val = -2 ** (int_bits + frac_bits - 1)

    # 限制 x 的值域
    x = torch.clamp(x, min_val, max_val)

    # 將小數點移動到左邊
    x_fixed = x * 2 ** frac_bits

    # 取整數部分
    x_fixed = x_fixed.long()

    # 轉換為定點數的 Tensor
    x_fixed = x_fixed.view(-1, 1)

    # 計算符號位元的值
    sign = torch.sign(x_fixed)

    # 計算整數位元的值
    int_part = torch.abs(x_fixed) // 2 ** frac_bits

    # 計算小數位元的值
    frac_part = torch.abs(x_fixed) % 2 ** frac_bits

    # 將符號、整數、小數部分合併成一個 Tensor
    fixed_point = torch.cat([sign, int_part, frac_part], dim=1)

    # 將定點數轉換為二進制表示
    fixed_point_bin = torch.zeros((fixed_point.shape[0], sign_bits + int_bits + frac_bits), dtype=torch.int)

    # 轉換符號位元
    for i in range(sign_bits):
        fixed_point_bin[:, i] = (fixed_point[:, 0] < 0).long()

    # 轉換整數位元
    if int_bits >= 5:
        fixed_point_bin[:, sign_bits:sign_bits+int_bits] = torch.abs(int_part)  # Repeat removed
    else:
        fixed_point_bin[:, sign_bits:sign_bits+int_bits] = torch.abs(int_part)

    # 轉換小數位元
    for i in range(frac_bits):
        fixed_point_bin[:, sign_bits+int_bits+i] = (fixed_point[:, 2] // 2 ** (frac_bits - 1 - i)) % 2

    return fixed_point_bin

def flip_bits(x, n):
    # Flatten the tensor into a 1D vector and randomly choose n indices
    flat_x = x.flatten()
    num_bits = len(flat_x)
    indices = random.sample(range(num_bits), n)
    
    # Flip the bits at the chosen indices
    flipped_bits = torch.zeros(num_bits, dtype=torch.bool)
    flipped_bits[indices] = True
    flipped_x = torch.where(flipped_bits, 1-flat_x, flat_x)
    
    # Reshape the flattened tensor back into its original shape
    return flipped_x.reshape(x.shape)

def fixed_point_to_float_sign_5_5_6(x_fixed, int_bits, frac_bits):
    # 轉換為定點數的 Tensor
    x_fixed = x_fixed.view(-1, int_bits + frac_bits + 13)

    # 轉換符號位元
    sign = torch.where(torch.sum(x_fixed[:, :13], dim=1) > 6, -1, 1)

    # 轉換整數位元
    int_part = x_fixed[:, 13:13+int_bits]
    int_part = torch.where(torch.sum(int_part, dim=1) > 6, 1, 0)

    # 轉換小數位元
    frac_part = x_fixed[:, 13+int_bits:13+int_bits+frac_bits]
    frac_part = torch.sum(frac_part * (1 / (2 ** torch.arange(1, frac_bits + 1))), dim=1)

    # 將整數和小數部分合併
    x = int_part + frac_part

    # 將符號、整數、小數部分合併
    x = x * sign.float()

    return x.view(1, -1)


device = torch.device("cpu:0")
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
embeds_shared = Embeds(vocab_size=108, num_hidden=512).to(device)
encoder = LSTMEncoder(channel_dim=channel_dim, embedds=embeds_shared).to(device)
decoder = LSTMDecoder(channel_dim=channel_dim, embedds=embeds_shared, vocab_size = 108).to(device)

encoder = encoder.eval()
decoder = decoder.eval()
embeds_shared = embeds_shared.eval()


normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
channel = Channel(_iscomplex=_iscomplex)

def do_test(input_data, encoder, decoder, normlize_layer, channel, len_batch):

    with torch.no_grad():
        print("input_data :")
        print(input_data)
        output, _ = encoder(input_data, len_batch)
        print("encoder :")
        print(output)
        output = normlize_layer.apply(output)
        print("normlize_layer :")
        print(output)

        print("float_point convert bit vector :")
        pred1 = float_to_fixed_point_sign_5_5_6(output)
        print(pred1) #fixed_point

        print('Channel noise :')
        flipped = flip_bits(pred1, n=24)
        print(flipped)

        # 將二進制表示轉換為浮點數
        pred8 = fixed_point_to_float_sign_5_5_6(flipped, int_bits=13, frac_bits=6)

        print(pred8)

        

        
        print("bit vector convert float_point :")
      

        print(pred8)  
        #pred8 = pred8.unsqueeze(0)

      

        output = decoder.sample_max_batch(pred8, None)
        print("decoder :")
        print(output)

    return output
SemanticRL_example = [          
                                
                     
                 #       'downlink information and vrb will map to prb and retransmission and a serving cell received',
                         'inference at lower precision'
                #        'vrb to prb is an interleaved mapping way',
                #        'vrb directly mapped to prb',
                #        'version zero redundancy indicates first control information',
                #        'redundancy version two mapped to third control information',
                #        'second control information required for version one redundancy',
                #        'three of redundancy version shows fourth control information',
                #        'sent new data',
                      
                      ]
#input_str = "['this message will send downlink information'\n'vrb to prb is an interleaved mapping way' ]"
input_str = "['inference at lower precision']"
#input_str = "['downlink information and vrb will map to prb and retransmission and a serving cell received']"


processed_str = input_str.strip("[]").replace("'", "")
print(processed_str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("--ckpt_pathCE", type=str, default='./ckpt_AWGN_CE_Stage2')
    parser.add_argument("--ckpt_pathRL", type=str, default='./ckpt_AWGN_RL')  # or './ckpt_AWGN_RL'
    args = parser.parse_args()

    dict_train = pickle.load(open('./train_dict.pkl', 'rb'))
    rev_dict = {vv: kk for kk, vv in dict_train.items()}
    
    
    success_count = 0
    failure_count = 0
    logging.basicConfig(filename='log.txt', level=logging.INFO)
    with open('log.txt', 'w') as log_file:
        for input_str in SemanticRL_example:

            input_vector = [dict_train[x] for x in input_str.split(' ')] + [2]
            input_len = len(input_vector)
            input_vector = torch.tensor(input_vector)
    
            for ckpt_dir in [args.ckpt_pathRL]:#, args.ckpt_pathRL
                model_name = os.path.basename(ckpt_dir)
    
                encoder.load_state_dict(torch.load(ckpt_dir + '/encoder_epoch201.pth', map_location='cpu'))
                decoder.load_state_dict(torch.load(ckpt_dir + '/decoder_epoch201.pth', map_location='cpu'))
                embeds_shared.load_state_dict(torch.load(ckpt_dir + '/embeds_shared_epoch201.pth',  map_location='cpu'))
    
                for _ in range(5000):
                    output = do_test(input_vector.unsqueeze(0), encoder, decoder, normlize_layer, channel,
                        len_batch=torch.tensor(input_len).view(-1, ))
                    output = output.cpu().numpy()[0]
                    res = ' '.join(rev_dict[x] for x in output if x!=0 and x!=2)  # remove 'PAD' and 'EOS'
                    
                    if res == processed_str:
                        success_count += 1
                    else:
                        failure_count += 1
                #print('result of {}:            {}'.format(model_name, res))
                    print('{}'.format(res))
                    print('--------------------------------------------------')
                   
                
                 
                #logging.info('{} {}'.format(datetime.datetime.now(), res))
                
                    log_file.write('{} {}\n'.format(datetime.datetime.now(), res))
    # Print the success and failure counts
    print('Success count:', success_count)
    print('Failure count:', failure_count)