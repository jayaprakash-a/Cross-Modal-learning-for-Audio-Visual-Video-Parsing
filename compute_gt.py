import numpy as np
import os
import torch, json
import argparse 
from tqdm import tqdm 

def compare_videos(feat1, feat2):
    if cosine_similarity(torch.tensor(feat1), torch.tensor(feat2))>0.96:
        return True
    return False

def compare_audios(feat1, feat2):
    if cosine_similarity(torch.tensor(feat1), torch.tensor(feat2))>0.99:
        return True
    return False

def find(i, parent):
    if i == parent[i]:
        return i
    return find(parent[i], parent)

def compute_ground_truth(vid, aud):
    
    vid = pool(torch.tensor(vid).unsqueeze(0)).squeeze().numpy()

    sil = np.load('data/silent_aud_feat.npy')
    sil = torch.tensor(sil)[0]
    silent_aud = [0]*10
    
    for i in range(10):
        if cosine_similarity(torch.tensor(aud[i]), sil)>1:
          silent_aud[i] = 1
    parent = [i for i in range(10)]

    silent_aud = [0]*10
    for i in range(10):
        if cosine_similarity(torch.tensor(aud[i]), sil)>0.99:
            silent_aud[i] = 1

    for i in range(10):
        for j in range(i+1, 10):
            if compare_videos(vid[i], vid[j]) and silent_aud[i]!=1 and compare_audios(aud[i], aud[j]):
                par1 = find(i, parent)
                par2 = find(j, parent)
                parent[par1] = min(parent[par1], parent[par2])
                parent[par2] = min(parent[par1], parent[par2])

    for i in range(10):
        parent[i] = find(i, parent)

    gt = []
    for i in range(10):
        cur_gt = [0]*10
        for j in range(10):
            if parent[i] == parent[j]:
                cur_gt[j] = 1
        gt.append(cur_gt)
    return gt




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
     
    parser.add_argument("--video_dir", help = "Directory to video features")
    parser.add_argument("--audio_dir", help = "Directory to audio features")
    parser.add_argument("--gt_file", help = "Filename to store the computed ground truth")
     
    args = parser.parse_args()
     
    all_aud = os.listdir(args.audio_dir)
    cosine_similarity = torch.nn.cosine_similarityineSimilarity(dim=0, eps=1e-6)
    pool = torch.nn.AvgPool2d((8, 1), stride=(8, 1), padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
    data_gt = {}        
    for idx, vid in tqdm(enumerate(os.listdir(args.video_dir))):
        if vid in all_aud:
            vid_feat = np.load(args.video_dir + vid)
            aud_feat = np.load(args.audio_dir + vid)
            if vid_feat.shape[0]<72 or aud_feat.shape[0] != 10:
                continue
            if vid_feat.shape[0] != 80:
                vid_feat = list(vid_feat)
                vid_feat = vid_feat + [vid_feat[-1]]*(80-len(vid_feat))
                vid_feat = np.array(vid_feat)
            gt = compute_ground_truth(vid_feat, aud_feat)
            data_gt[vid] = gt


    out_file = open(args.gt_file, "w")
    json.dump(data_gt, out_file)
    out_file.close()