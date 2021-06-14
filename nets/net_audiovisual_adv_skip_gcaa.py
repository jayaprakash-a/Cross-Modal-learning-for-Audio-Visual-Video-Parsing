import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import copy
import math
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None




class GlobalContextAwareAttention(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.qx_linear = nn.Linear(d_model, d_model)
        self.vx_linear = nn.Linear(d_model, d_model)
        self.kx_linear = nn.Linear(d_model, d_model)

        self.qc_linear = nn.Linear(d_model, d_model)
        self.kc_linear = nn.Linear(d_model, d_model)

        self.Vqx_linear = nn.Linear(d_model, 1)
        self.Vqc_linear = nn.Linear(d_model, 1)
        self.Vkx_linear = nn.Linear(d_model, 1)
        self.Vkc_linear = nn.Linear(d_model, 1)

        self.sigmoid=nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, qx, kx, vx, qc, kc, mask=None):

        qx = qx.permute(1, 0, 2)
        vx = vx.permute(1, 0, 2)
        kx = kx.permute(1, 0, 2)
        qc = qc.permute(1, 0, 2)
        kc = kc.permute(1, 0, 2)

        bs = qx.size(0)

        # perform linear operation 
        kx = self.kx_linear(kx)
        qx = self.qx_linear(qx)
        v = self.vx_linear(vx)
        qc = self.qc_linear(qc)
        kc = self.kc_linear(kc)

        lambda_q=self.sigmoid(self.Vqx_linear(qx)+self.Vqc_linear(qc))
        lambda_k=self.sigmoid(self.Vkx_linear(kx)+self.Vkc_linear(kc))

        # print((1-lambda_q).shape)
        # print(qx.shape)
        q=(1-lambda_q)*qx+lambda_q*qc
        k=(1-lambda_k)*kx+lambda_k*kc

        # split into h heads
        k = k.view(bs, -1, self.h, self.d_k)
        q = q.view(bs, -1, self.h, self.d_k)
        v = v.view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)

        
        output = self.out(concat)
        output = output.permute(1, 0, 2)
        return output


    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)
            
        output = torch.matmul(scores, v)
        return output





def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)

        return output_a, output_v




class SANLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(SANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src_qc = torch.mean(src_q, 1).unsqueeze(1)
        src_qc = src_qc.repeat(1, src_q.size(1), 1)

        src_q = src_q.permute(1, 0, 2)
        src_qc = src_qc.permute(1, 0, 2)

        src1 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # src1 = self.self_attn(src_q, src_q, src_q, src_qc, src_qc, mask=src_mask)[0]
        
        src_q = src_q + self.dropout11(src1)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)


class CMANLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(CMANLayer, self).__init__()
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)


        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src_qc = torch.mean(src_q, 1).unsqueeze(1)
        src_qc = src_qc.repeat(1, src_q.size(1), 1)

        src_vc = torch.mean(src_v, 1).unsqueeze(1)
        src_vc = src_vc.repeat(1, src_v.size(1), 1)

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src_qc = src_qc.permute(1, 0, 2)
        src_vc = src_vc.permute(1, 0, 2)


        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask)[0]
        
        
        src_q = src_q + self.dropout11(src1)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)


class Modality_classifier(nn.Module):
    def __init__(self):
        super(Modality_classifier, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.norm1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU(True)

        self.fc2 = nn.Linear(128, 10)
        self.norm2 = nn.BatchNorm1d(10)
        self.relu2 = nn.ReLU(True)

        self.fc3 = nn.Linear(10, 1)

    def forward(self, reverse_features):
        feat = self.fc1(reverse_features)
        # feat = feat.permute(0, 2, 1)
        feat = self.norm1(feat)
        # feat = feat.permute(0, 2, 1)
        feat = self.relu1(feat)

        feat = self.fc2(feat)
        # feat = feat.permute(0, 2, 1)
        feat = self.norm2(feat)
        # feat = feat.permute(0, 2, 1)
        feat = self.relu2(feat)

        feat = self.fc3(feat)
        mod_pred = torch.sigmoid(feat)
        mod_pred = mod_pred.squeeze(-1)
        return mod_pred
        


class MMIL_Net(nn.Module):

    def __init__(self):
        super(MMIL_Net, self).__init__()

        self.shared_fc = nn.Linear(512, 512)

        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_av_att = nn.Linear(512, 25)
        self.fc_a =  nn.Linear(128, 512)
        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)

        self.sa_layer_a = SANLayer(d_model=512, nhead=1, dim_feedforward=512)
        self.sa_layer_v = SANLayer(d_model=512, nhead=1, dim_feedforward=512)

        self.cma_layer = Encoder(CMANLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)

        self.mod_classifier = Modality_classifier()
        self.activation = nn.ReLU()
        self.norm_a = nn.LayerNorm(512)
        self.norm_v= nn.LayerNorm(512)





    def forward(self, audio, visual, visual_st, alpha=0.5):

        x1_0 = self.fc_a(audio)

        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        # vid_s = self.fc_v(visual)
        vid_st = self.fc_st(visual_st)
        x2 = torch.cat((vid_s, vid_st), dim =-1)
        # print(x2.shape)
        x2_0 = self.fc_fusion(x2)
        # print(x2.shape)


        x1=self.sa_layer_a(x1_0)
        x2=self.sa_layer_v(x2_0)

        # HAN
        x1_mm, x2_mm = self.cma_layer(x1, x2)

        # prediction
        x = torch.cat([x1_mm.unsqueeze(-2), x2_mm.unsqueeze(-2)], dim=-2)
        x = self.shared_fc(x)

        x1 = x[:, :, 0, :].squeeze(-2)+x1
        x1 = self.norm_a(x1)
        x2 = x[:, :, 1, :].squeeze(-2)+x2
        x2 = self.norm_v(x2)

        x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2)


        x_mod = self.activation(x)
        reverse_features = ReverseLayerF.apply(x_mod, alpha)
        reverse_features = reverse_features.view(-1, 512)

        mod_gt = torch.ones(reverse_features.size(0)).to('cuda')
        mod_gt[::2] = 0

        perm = torch.randperm(mod_gt.size(0)).to('cuda')
        mod_gt = mod_gt[perm]
        reverse_features = reverse_features[perm, :]
        mod_pred = self.mod_classifier(reverse_features)



        frame_prob = torch.sigmoid(self.fc_prob(x))

        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
        av_att = torch.softmax(self.fc_av_att(x), dim=2)
        temporal_prob = (frame_att * frame_prob)
        global_prob = (temporal_prob*av_att).sum(dim=2).sum(dim=1)

        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
        v_prob =temporal_prob[:, :, 1, :].sum(dim=1)

        return global_prob, a_prob, v_prob, frame_prob, mod_pred, mod_gt

