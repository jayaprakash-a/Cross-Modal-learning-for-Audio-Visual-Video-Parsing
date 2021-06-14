import json, pickle
import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from model.model import *
from torch import optim
from tqdm import tqdm
import pandas as pd
from model.eval_metrics import segment_level, event_level
import argparse
from pathlib import Path


def ids_to_multinomial(ids):
    """ label encoding

    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))
    for id in ids:
        index = id_to_idx[id]
        y[index] = 1
    return y

class AudioSet(Dataset):

    def __init__(self, split, gt, video_dir, audio_dir):
        self.split = split
        self.gt = gt
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.pool = torch.nn.AvgPool2d((8, 1), stride=(8, 1), padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        video = (torch.FloatTensor(np.load(os.path.join(self.video_dir, self.split[idx])))[:80, :]).unsqueeze(0)
        audio = torch.FloatTensor(np.load(os.path.join(self.audio_dir, self.split[idx])))[:10, :]
        
        if video.shape[1]<80:
            video = np.load(os.path.join(self.video_dir, self.split[idx]))
            video = list(video)
            video = video + [video[-1]]*(80-len(video))
            video = torch.FloatTensor(video).unsqueeze(0)
        video = self.pool(video).squeeze(0)
        gt = torch.FloatTensor(self.gt[self.split[idx]])
        return video, audio, gt

class LLP_dataset(Dataset):

    def __init__(self, label, audio_dir, video_dir, st_dir, transform=None):
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.filenames = self.df["filename"]
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.st_dir = st_dir
        self.pool = torch.nn.AvgPool2d((8, 1), stride=(8, 1), padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        name = row[0][:11]
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s = self.pool(torch.tensor(np.load(os.path.join(self.video_dir, name + '.npy'))).unsqueeze(0)).squeeze()
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))
        ids = row[-1].split(',')
        label = ids_to_multinomial(ids)
        sample = {'audio': audio, 'video_s': video_s, 'video_st': video_st, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def eval(model, val_loader, set, rgb_pos_feat, audio_pos_feat, r2p1d_pos_feat, args):
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    model.eval()

    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv(args.label_dir+"AVVP_eval_audio.csv", header=0, sep='\t')
    df_v = pd.read_csv(args.label_dir+"AVVP_eval_visual.csv", header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio, video, video_st, frame_prob = sample['audio'].to(args.device), sample['video_s'].to(args.device),sample['video_st'].to(args.device), sample['label'].to(args.device)
            cur_batch = video.shape[0]
            attention_mask = torch.ones(cur_batch, 30).to(args.device)
            rgb_masks = torch.ones(cur_batch, 10).to(args.device)
            audio_masks = torch.ones(cur_batch, 10).to(args.device)  
            r2p1d_masks = torch.ones(cur_batch, 10).to(args.device)  
            gather_index = torch.ones(cur_batch, 20, dtype=torch.int64).to(args.device)          
            output, a_prob, v_prob, frame_prob = model(video, rgb_pos_feat, video_st, r2p1d_pos_feat, audio, audio_pos_feat, attention_mask, gather_index, rgb_masks, audio_masks, r2p1d_masks, 'avvp')
            o = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)

            Pa = frame_prob[0, :, 0, :].cpu().detach().numpy()
            Pv = frame_prob[0, :, 1, :].cpu().detach().numpy()

            # filter out false positive events with predicted weak labels
            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)

            # extract audio GT labels
            GT_a = np.zeros((25, 10))
            GT_v =np.zeros((25, 10))

            df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num >0:
                for i in range(num):

                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1

            # extract visual GT labels
            df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1

            GT_av = GT_a * GT_v

            # obtain prediction matrices
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)

    
    print('Audio Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_a))))
    print('Visual Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_v))))
    print('Audio-Visual Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_av))))

    avg_type = (100 * np.mean(np.array(F_seg_av))+100 * np.mean(np.array(F_seg_a))+100 * np.mean(np.array(F_seg_v)))/3.
    avg_event = 100 * np.mean(np.array(F_seg))
    print('Segment-levelType@Avg. F1: {:.1f}'.format(avg_type))
    print('Segment-level Event@Avg. F1: {:.1f}'.format(avg_event))

    print('Audio Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_a))))
    print('Visual Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_v))))
    print('Audio-Visual Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_av))))

    avg_type_event = (100 * np.mean(np.array(F_event_av)) + 100 * np.mean(np.array(F_event_a)) + 100 * np.mean(
        np.array(F_event_v))) / 3.
    avg_event_level = 100 * np.mean(np.array(F_event))
    print('Event-level Type@Avg. F1: {:.1f}'.format(avg_type_event))
    print('Event-level Event@Avg. F1: {:.1f}'.format(avg_event_level))
    return avg_type

def train(args):
    f = open(args.gt_file)
    gt = json.load(f)
    vids = list(gt.keys())
    random.shuffle(vids)
    train_split = vids[:int(len(vids)*0.8)]
    dev_split = vids[int(len(vids)*0.8):]

    train_dataset = AudioSet(train_split, gt, args.audioset_dir+'resnet152/', args.audioset_dir+'vgg_feat/')
    train_loader = DataLoader(train_dataset, args.pre_batch_size, shuffle = True, num_workers=20)
    val_dataset = AudioSet(dev_split, gt, args.audioset_dir+'resnet152/', args.audioset_dir+'vgg_feat/')
    val_loader = DataLoader(val_dataset, args.pre_batch_size, shuffle = True, num_workers=20)
    
    config = UniterConfig(args.hidden_size)
    rgb_pos_feat = torch.tensor([i for i in range(10)]).unsqueeze(0).to(args.device)
    audio_pos_feat = torch.tensor([i for i in range(10)]).unsqueeze(0).to(args.device)
    r2p1d_pos_feat = torch.tensor([i for i in range(10)]).unsqueeze(0).to(args.device)

    model = avg(config, 2048, 512, 128).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_epoch_loss = 10**7

    cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6).to(args.device)

    for epoch in range(args.pre_epochs):
        model.train()
        epoch_loss = 0.0
        for vid, aud, labels in tqdm(train_loader):
            vid = vid.to(args.device)
            aud = aud.to(args.device)
            labels = labels.to(args.device)
            
            # Individual and joint Modality mask
            cur_batch = vid.shape[0]
            attention_mask = torch.ones(cur_batch, 30).to(args.device)
            attention_mask[:, 20:] = 0.0
            rgb_masks = torch.ones(cur_batch, 10).to(args.device)
            audio_masks = torch.ones(cur_batch, 10).to(args.device)
            r2p1d_masks = torch.ones(cur_batch, 10).to(args.device)
            gather_index = torch.ones(cur_batch, 20, dtype=torch.int64).to(args.device)
            r2p1d = torch.zeros(cur_batch, 10, 512).to(args.device)

            optimizer.zero_grad()
            embed, vid_embed, aud_embed, _ = model(vid, rgb_pos_feat, r2p1d, r2p1d_pos_feat, aud, audio_pos_feat, attention_mask, gather_index, rgb_masks, audio_masks, r2p1d_masks, 'avg')
            index = torch.LongTensor(1).random_(0, 10)[0]
            indi_vid_embed = vid_embed[:, index, :].unsqueeze(1)
            indi_aud_embed = aud_embed[:, index, :].unsqueeze(1)
            
            # Cross Modal Loss
            if epoch%2==0:
                vid_ground = cos(indi_vid_embed, aud_embed)
                aud_ground = cos(indi_aud_embed, vid_embed)
            # Uni Modal Loss
            else:
                vid_ground = cos(indi_vid_embed, vid_embed)
                aud_ground = cos(indi_aud_embed, aud_embed)

            gt = labels[:, index]
            
            # Cross Entropy Loss
            loss_vid = torch.where(gt==1., (0.7-vid_ground).clip(min=0), (vid_ground-0.7).clip(min=0))
            loss_aud = torch.where(gt==1., (0.7-aud_ground).clip(min=0), (aud_ground-0.7).clip(min=0))
            loss = loss_vid.mean() + loss_aud.mean()

            epoch_loss += loss/2
            loss.backward()
            optimizer.step()


        print(epoch+1, "Epoch AVG Train Loss:", epoch_loss/len(train_loader))

        model.eval()
        with torch.no_grad():
            loss = 0
            epoch_loss = 0.0
            for vid, aud, labels in tqdm(val_loader):
                vid = vid.to(args.device)
                aud = aud.to(args.device)
                labels = labels.to(args.device)

                # Individual and joint Modality mask
                cur_batch = vid.shape[0]
                attention_mask = torch.ones(cur_batch, 30).to(args.device)
                attention_mask[:, 20:] = 0.0
                rgb_masks = torch.ones(cur_batch, 10).to(args.device)
                audio_masks = torch.ones(cur_batch, 10).to(args.device)
                r2p1d_masks = torch.ones(cur_batch, 10).to(args.device)
                gather_index = torch.ones(cur_batch, 20, dtype=torch.int64).to(args.device)
                r2p1d = torch.zeros(cur_batch, 10, 512).to(args.device)
                
                embed, vid_embed, aud_embed, _ = model(vid, rgb_pos_feat, r2p1d, r2p1d_pos_feat, aud, audio_pos_feat, attention_mask, gather_index, rgb_masks, audio_masks, r2p1d_masks)
                for idx in range(10):
                    
                    # Cross Modal Loss
                    indi_vid_embed = vid_embed[:, idx, :].unsqueeze(1)
                    indi_aud_embed = aud_embed[:, idx, :].unsqueeze(1)
                    vid_ground = cos(indi_vid_embed, aud_embed)
                    aud_ground = cos(indi_aud_embed, vid_embed)
                
                    gt = torch.zeros(cur_batch, 10).to(args.device)
                    gt[:, idx] = 1
                    loss_vid = torch.where(gt==1, (0.7-vid_ground).clip(min=0), (vid_ground-0.7).clip(min=0))
                    loss_aud = torch.where(gt==1, (0.7-aud_ground).clip(min=0), (aud_ground-0.7).clip(min=0))
                    loss = loss_vid.mean() + loss_aud.mean()


                    # Uni Modal Loss
                    vid_ground = cos(indi_vid_embed, vid_embed)
                    aud_ground = cos(indi_aud_embed, aud_embed)
                    gt = torch.zeros(cur_batch, 10).to(args.device)
                    gt[:, idx] = 1
                    loss_vid = torch.where(gt==1, (0.7-vid_ground).clip(min=0), (vid_ground-0.7).clip(min=0))
                    loss_aud = torch.where(gt==1, (0.7-aud_ground).clip(min=0), (aud_ground-0.7).clip(min=0))
                    loss += loss_vid.mean() + loss_aud.mean()
                    epoch_loss += loss
            
            print(epoch+1, "Epoch AVG Val Loss:", epoch_loss/(len(val_loader)*40))

            if epoch_loss < best_epoch_loss:
            	best_epoch_loss = epoch_loss
            	torch.save(model.state_dict(), args.model_save_dir+"pretrained.pt")

    model.load_state_dict(torch.load(args.model_save_dir+"pretrained.pt", map_location=args.device))

    train_dataset = LLP_dataset(label=args.label_dir+'AVVP_train.csv', audio_dir=args.llp_dir+'vggish/', video_dir=args.llp_dir+'res152/', st_dir=args.llp_dir+'r2plus1d_18/')
    val_dataset = LLP_dataset(label=args.label_dir+'AVVP_val_pd.csv', audio_dir=args.llp_dir+'vggish/', video_dir=args.llp_dir+'res152/', st_dir=args.llp_dir+'r2plus1d_18/')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)
    
    criterion = nn.BCELoss()
    best_F = 0
    
    for epoch in range(args.epochs):
        for batch_idx, sample in enumerate(train_loader):
            audio, video, video_st, target = sample['audio'].to(args.device), sample['video_s'].to(args.device), sample['video_st'].to(args.device), sample['label'].type(torch.FloatTensor).to(args.device)

            optimizer.zero_grad()
            cur_batch = video.shape[0]
            attention_mask = torch.ones(cur_batch, 30).to(args.device)
            rgb_masks = torch.ones(cur_batch, 10).to(args.device)
            audio_masks = torch.ones(cur_batch, 10).to(args.device)
            r2p1d_masks = torch.ones(cur_batch, 10).to(args.device)
            gather_index = torch.ones(cur_batch, 20, dtype=torch.int64).to(args.device)            
            output, a_prob, v_prob, _ = model(video, rgb_pos_feat, video_st, r2p1d_pos_feat, audio, audio_pos_feat, attention_mask, gather_index, rgb_masks, audio_masks, r2p1d_masks, 'avvp')
            output.clamp_(min=1e-7, max=1 - 1e-7)
            a_prob.clamp_(min=1e-7, max=1 - 1e-7)
            v_prob.clamp_(min=1e-7, max=1 - 1e-7)

            # label smoothing
            a = 1.0
            v = 0.9 
            Pa = a * target + (1 - a) * 0.5
            Pv = v * target + (1 - v) * 0.5

            # individual guided learning
            loss =  criterion(a_prob, Pa) + criterion(v_prob, Pv) + criterion(output, target) 

            loss.backward()
            optimizer.step()
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(audio), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

        F = eval(model, val_loader, args.label_dir+'AVVP_val_pd.csv', rgb_pos_feat, audio_pos_feat, r2p1d_pos_feat, args)
        if F >= best_F:
            best_F = F
            torch.save(model.state_dict(), args.model_save_dir+"final.pt")
    test(args)

def test(args):
    config = UniterConfig(args.hidden_size)
    model = avg(config, 2048, 512, 128).to(args.device)
    model.load_state_dict(torch.load(args.model_save_dir+"final.pt", map_location=args.device))

    test_dataset = LLP_dataset(label=args.label_dir+'AVVP_test_pd.csv', audio_dir=args.llp_dir+'vggish/', video_dir=args.llp_dir+'res152/', st_dir=args.llp_dir+'r2plus1d_18/')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    rgb_pos_feat = torch.tensor([i for i in range(10)]).unsqueeze(0).to(args.device)
    audio_pos_feat = torch.tensor([i for i in range(10)]).unsqueeze(0).to(args.device)
    r2p1d_pos_feat = torch.tensor([i for i in range(10)]).unsqueeze(0).to(args.device)

    eval(model, test_loader, args.label_dir+'AVVP_test_pd.csv', rgb_pos_feat, audio_pos_feat, r2p1d_pos_feat, args)

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Video Parsing')
    parser.add_argument("--audioset_dir", type=str, default='../data/audioset/', help="audioset path")
    parser.add_argument("--llp_dir", type=str, default='../data/feats/LLP/', help="LLP dir")
    parser.add_argument("--hidden_size", type=int, default=1024, help="pretrain embedding size")
    parser.add_argument("--label_dir", type=str, default="../data/", help="csv directory")
    parser.add_argument('--pre_batch_size', type=int, default=16, metavar='N', help='input batch size for pretraining (default: 16)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument('--pre_epochs', type=int, default=40, metavar='N', help='number of epochs to pretrain (default: 60)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument("--mode", type=str, default='train', help="with mode to use")
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--model_save_dir", type=str, default='../checkpoint/', help="model save dir")
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device number')
    parser.add_argument('--gt_file', type=str, default='./', help='Ground truth for AVG file')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    print(args)
    if args.mode == 'train':
        train(args)
    else:
        test(args)
if __name__ == '__main__':
    main()
