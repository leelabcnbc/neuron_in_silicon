import matplotlib.pyplot as plt
from basic_stimuli import generate_stimuli, generate_center_black_grating
import torch
import numpy as np
from tqdm import tqdm

from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def alter_contrast(img, percentage):
    # img -= img.min()
    # img /= (img.max() - img.min())
    img = img * percentage
    img = img + (1 - percentage) / 2
    return img


def get_best_orientation(net):
    net.to(device)
    # YOUR SOLUTION START HERE

    # END OF YOUR SOLUTION
    return best_orientation


def get_diff_freq_response(net):
    all_neuron_rsp = []
    best_orientation = get_best_orientation(net)
    net.to(device)
    for i, orientation in enumerate(tqdm(best_orientation)):
        all_freq_rsp = []
        for freq in range(5, 26, 5):
            all_rsp = []
            stimuli = generate_stimuli(freq, orientation, (25, 25), 0.8, 50, range(1, 26), np.arange(0,1,0.25), gray=0.5)
            for s_radi in stimuli:
                cur_rsp = []
                for s in s_radi:
                    s = torch.FloatTensor(s).to(device)
                    s = torch.reshape(s, (1, 1, 50, 50))
                    rsp = net(s)
                    cur_rsp.append(rsp.detach().cpu().numpy())
                cur_rsp = np.array(cur_rsp)
                all_rsp.append(np.mean(cur_rsp, axis=0))
            all_rsp = np.stack(all_rsp)[:, 0, i]
            all_freq_rsp.append(all_rsp)
        all_freq_rsp = np.stack(all_freq_rsp)
        all_neuron_rsp.append(all_freq_rsp)
    all_neuron_rsp = np.stack(all_neuron_rsp)
    return all_neuron_rsp


def get_surround_only_response(net):
    best_orientation = get_best_orientation(net)
    net.to(device)
    all_neuron_rsp = []
    for i, orientation in enumerate(tqdm(best_orientation)):
        all_rsp = []
        stimuli = generate_center_black_grating(16, orientation, (25, 25), 0.8, 50, 25, 0, range(1, 26), gray=0.0)
        for s_radi in stimuli:
            cur_rsp = []
            for s in s_radi:
                s = torch.FloatTensor(s).to(device)
                s = torch.reshape(s, (1, 1, 50, 50))
                rsp = net(s)
                cur_rsp.append(rsp.detach().cpu().numpy())
            cur_rsp = np.array(cur_rsp)
            all_rsp.append(np.max(cur_rsp, axis=0))
        all_rsp = np.stack(all_rsp)[:, 0, i]
        all_neuron_rsp.append(all_rsp)
    all_neuron_rsp = np.stack(all_neuron_rsp)
    return all_neuron_rsp

def get_center_only_response(net):
    all_neuron_rsp = []
    best_orientation = get_best_orientation(net)
    net.to(device)
    for i, orientation in enumerate(tqdm(best_orientation)):
        all_rsp = []
        stimuli = generate_stimuli(16, orientation, (25, 25), 0.8, 50, range(1, 26, 1), np.arange(0, 1, 0.2), gray=0.0)
        for s_radi in stimuli:
            cur_rsp = []
            for s in s_radi:
                s = torch.FloatTensor(s).to(device)
                s = torch.reshape(s, (1, 1, 50, 50))
                rsp = net(s)
                cur_rsp.append(rsp.detach().cpu().numpy())
            cur_rsp = np.array(cur_rsp)
            all_rsp.append(np.max(cur_rsp, axis=0))
        all_rsp = np.stack(all_rsp)[:, 0, i]
        all_neuron_rsp.append(all_rsp)
    all_neuron_rsp = np.stack(all_neuron_rsp)
    return all_neuron_rsp

def get_best_freq_response(net):
    all_neuron_rsp = []
    best_orientation = get_best_orientation(net)
    net.to(device)
    for i, orientation in enumerate(tqdm(best_orientation)):
        all_freq_rsp = []
        for freq in range(5, 26, 5):
            all_rsp = []
            stimuli = generate_stimuli(freq, orientation, (25, 25), 0.8, 50, range(1, 26, 1), np.arange(0,1,0.25), gray=0.5)
            for s_radi in stimuli:
                cur_rsp = []
                for s in s_radi:
                    s = torch.FloatTensor(s).to(device)
                    s = torch.reshape(s, (1, 1, 50, 50))
                    rsp = net(s)
                    cur_rsp.append(rsp.detach().cpu().numpy())
                cur_rsp = np.array(cur_rsp)
                all_rsp.append(np.mean(cur_rsp, axis=0))
            all_rsp = np.stack(all_rsp)[:, 0, i]
            all_freq_rsp.append(all_rsp)
        #find highest point
        all_freq_rsp = np.stack(all_freq_rsp)
        freq_rsp_avg = np.average(all_freq_rsp,axis=0)
        max_point = np.argmax(freq_rsp_avg)
        max_slice = all_freq_rsp[:,max_point]
        max_freq_idx = np.argmax(max_slice)
        max_freq_rsp = all_freq_rsp[max_freq_idx]
        all_neuron_rsp.append(max_freq_rsp)
    all_neuron_rsp = np.stack(all_neuron_rsp)
    return all_neuron_rsp
