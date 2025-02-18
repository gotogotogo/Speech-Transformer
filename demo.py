import argparse
import pickle
import random
from shutil import copyfile

import torch
import json

from config import pickle_file, device, input_dim, LFR_m, LFR_n
from data_gen import build_LFR_features
from transformer.transformer import Transformer
from utils import extract_feature, ensure_folder


def parse_args():
    parser = argparse.ArgumentParser(
        "End-to-End Automatic Speech Recognition Decoding.")
    # decode
    parser.add_argument('--beam_size', default=5, type=int,
                        help='Beam size')
    parser.add_argument('--nbest', default=5, type=int,
                        help='Nbest size')
    parser.add_argument('--decode_max_len', default=100, type=int,
                        help='Max output length. If ==0 (default), it uses a '
                             'end-detect function to automatically find maximum '
                             'hypothesis lengths')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open('char_list.pkl', 'rb') as file:
        char_list = pickle.load(file)
    with open(pickle_file + '_test.pkl', 'rb') as file:
        samples = pickle.load(file)
    # samples = data['test']

    filename = 'speech-transformer-cn.pt'
    print('loading model: {}...'.format(filename))
    model = Transformer()
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()

    samples = random.sample(samples, 10)
    ensure_folder('audios')
    results = []

    for i, sample in enumerate(samples):
        feature = sample['wave']
        trn = sample['label']

        # copyfile(wave, 'audios/audio_{}.wav'.format(i))

        # feature = extract_feature(input_file=wave, feature='fbank', dim=input_dim, cmvn=True)
        feature = build_LFR_features(feature, m=LFR_m, n=LFR_n)
        # feature = np.expand_dims(feature, axis=0)
        input = torch.from_numpy(feature).to(device)
        input_length = [input[0].shape[0]]
        input_length = torch.LongTensor(input_length).to(device)
        nbest_hyps = model.recognize(input, input_length, char_list, args)
        out_list = []
        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [char_list[idx] for idx in out]
            out = ''.join(out)
            out_list.append(out)
        print('OUT_LIST: {}'.format(out_list))

        gt = [char_list[idx] for idx in trn]
        gt = ''.join(gt)
        print('GT: {}\n'.format(gt))

        results.append({'out_list_{}'.format(i): out_list, 'gt_{}'.format(i): gt})


    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
