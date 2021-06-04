import copy
import json
import logging
from io import open

import torch.nn.functional as F
import math
import torch
from torch import nn
from torch import optim

from layer import BertLayer, BertPooler


class UniterConfig(object):
    """Configuration class to store the configuration of a `UniterModel`.
    """
    def __init__(self,
                 hidden_size=512,
                 num_hidden_layers=12,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 initializer_range=0.02):
        """Constructs UniterConfig.
        Args:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer
                encoder.
            num_attention_heads: Number of attention heads for each attention
                layer in the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e.
                feed-forward) layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string)
                in the encoder and pooler. If string, "gelu", "relu" and
                "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully
                connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this
                model might ever be used with. Typically set this to something
                large just in case (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer
                for initializing all weight matrices.
        """
        if isinstance(hidden_size, int):
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be hidden layer size (int) ")

    # @classmethod
    # def from_dict(cls, json_object):
    #     """Constructs a `UniterConfig` from a
    #        Python dictionary of parameters."""
    #     config = UniterConfig(vocab_size_or_config_json_file=-1)
    #     for key, value in json_object.items():
    #         config.__dict__[key] = value
    #     return config

    # @classmethod
    # def from_json_file(cls, json_file):
    #     """Constructs a `UniterConfig` from a json file of parameters."""
    #     with open(json_file, "r", encoding='utf-8') as reader:
    #         text = reader.read()
    #     return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class UniterPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, UniterConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `UniterConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        """
        Instantiate a UniterPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        # Load config
        config = UniterConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                            model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                            model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                                   model.__class__.__name__,
                                   "\n\t".join(error_msgs)))
        return model


def make_pos_embed(max_seq_len, d_model):
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i) / d_model)))
            pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return pe


class UniterRGBEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.fc_v= nn.Linear(img_dim, img_dim)
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # In Image they take 7d vector for height width and so on, but in video we have time domain which is continous.
        self.pos_embed = make_pos_embed(10, config.hidden_size)
        self.position_embeddings = nn.Embedding.from_pretrained(self.pos_embed, freeze=True)

        # tf naming convention for layer norm
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_ids, type_embeddings, img_masks=None):

        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.position_embeddings(img_pos_ids)

        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterR2p1dEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.fc_v= nn.Linear(img_dim, img_dim)
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # In Image they take 7d vector for height width and so on, but in video we have time domain which is continous.
        self.pos_embed = make_pos_embed(10, config.hidden_size)
        self.position_embeddings = nn.Embedding.from_pretrained(self.pos_embed, freeze=True)

        # tf naming convention for layer norm
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_ids, type_embeddings, img_masks=None):

        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.position_embeddings(img_pos_ids)

        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class UniterAudioEmbeddings(nn.Module):
    def __init__(self, config, audio_dim):
        super().__init__()
        self.audio_linear = nn.Linear(audio_dim, config.hidden_size)
        self.audio_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.mask_embedding = nn.Embedding(2, audio_dim, padding_idx=0)

        # In Image they take 7d vector for height width and so on, but in audio we have time domain which is continous.
        self.pos_embed = make_pos_embed(10, config.hidden_size)
        self.position_embeddings = nn.Embedding.from_pretrained(self.pos_embed, freeze=True)
        
        # tf naming convention for layer norm
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, audio_feat, audio_pos_ids, type_embeddings, audio_masks=None):
        if audio_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(audio_masks.long())
            audio_feat = audio_feat + mask

        transformed_im = self.audio_layer_norm(self.audio_linear(audio_feat))
        transformed_pos = self.position_embeddings(audio_pos_ids)
        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class UniterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, input_, attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class UniterModel(UniterPreTrainedModel):
    """ Modification for Joint Visual-Audio Encoding
    """
    def __init__(self, config, rgb_dim, r2p1d_dim, audio_dim):
        super().__init__(config)
        self.rgb_embeddings = UniterRGBEmbeddings(config, rgb_dim)
        self.r2p1d_embeddings = UniterR2p1dEmbeddings(config, r2p1d_dim)
        self.audio_embeddings = UniterAudioEmbeddings(config, audio_dim)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.encoder = UniterEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def _compute_rgb_embeddings(self, rgb_feat, rgb_pos_feat, rgb_masks=None,
                                rgb_type_ids=None):
        if rgb_type_ids is None:
            rgb_type_ids = torch.zeros_like(rgb_feat[:, :, 0].long())
        rgb_type_embeddings = self.token_type_embeddings(
            rgb_type_ids)
        output = self.rgb_embeddings(rgb_feat, rgb_pos_feat,
                                     rgb_type_embeddings, rgb_masks)
        return output

    def _compute_audio_embeddings(self, audio_feat, audio_pos_feat, audio_masks=None,
                                audio_type_ids=None):
        if audio_type_ids is None:
            audio_type_ids = torch.ones_like(audio_feat[:, :, 0].long())
        audio_type_embeddings = self.token_type_embeddings(
            audio_type_ids)
        output = self.audio_embeddings(audio_feat, audio_pos_feat,
                                     audio_type_embeddings, audio_masks)
        return output


    def _compute_r2p1d_embeddings(self, r2p1d_feat, r2p1d_pos_feat, r2p1d_masks=None,
                                r2p1d_type_ids=None):
        if r2p1d_type_ids is None:
            r2p1d_type_ids = torch.ones_like(r2p1d_feat[:, :, 0].long())
        r2p1d_type_embeddings = self.token_type_embeddings(
            r2p1d_type_ids)
        output = self.r2p1d_embeddings(r2p1d_feat, r2p1d_pos_feat,
                                     r2p1d_type_embeddings, r2p1d_masks)
        return output

    def _compute_rgb_audio_embeddings(self, rgb_feat, rgb_pos_feat,
                                    audio_feat, audio_pos_feat, 
                                    gather_index, rgb_masks=None, audio_masks=None,
                                    audio_type_ids=None, rgb_type_ids=None):
        rgb_emb = self._compute_rgb_embeddings(
                rgb_feat, rgb_pos_feat, rgb_masks, rgb_type_ids)
        audio_emb = self._compute_audio_embeddings(
            audio_feat, audio_pos_feat, audio_masks, audio_type_ids)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.cat([rgb_emb, audio_emb], dim=1)
        return embedding_output

    
    def _compute_rgb_audio_r2p1d_embeddings(self, rgb_feat, rgb_pos_feat,
                                    audio_feat, audio_pos_feat, 
                                    r2p1d_feat, r2p1d_pos_feat, 
                                    gather_index, rgb_masks=None, audio_masks=None, r2p1d_masks=None,
                                    audio_type_ids=None, rgb_type_ids=None, r2p1d_type_ids=None):
        rgb_emb = self._compute_rgb_embeddings(
                rgb_feat, rgb_pos_feat, rgb_masks, rgb_type_ids)
        audio_emb = self._compute_audio_embeddings(
            audio_feat, audio_pos_feat, audio_masks, audio_type_ids)
        r2p1d_emb = self._compute_r2p1d_embeddings(
            r2p1d_feat, r2p1d_pos_feat, r2p1d_masks, r2p1d_type_ids)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.cat([rgb_emb, audio_emb, r2p1d_emb], dim=1)
        return embedding_output

    def forward(self, rgb_feat, rgb_pos_feat,
                r2p1d_feat, r2p1d_pos_feat,
                audio_feat, audio_pos_feat,
                attention_mask, gather_index=None, rgb_masks=None, audio_masks=None, r2p1d_masks=None,
                output_all_encoded_layers=False,
                rgb_type_ids=None, r2p1d_type_ids=None, audio_type_ids=None):
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self._compute_rgb_audio_r2p1d_embeddings(
                 rgb_feat, rgb_pos_feat, audio_feat, audio_pos_feat,  r2p1d_feat, r2p1d_pos_feat, 
                gather_index, rgb_masks, audio_masks, r2p1d_masks,
                audio_type_ids, rgb_type_ids, r2p1d_type_ids)
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers

class avg(nn.Module):
    def __init__(self, config, rgb_dim=2048, r2p1d_dim=2048, audio_dim=128):
        super().__init__()
        self.uniter = UniterModel(config, rgb_dim, r2p1d_dim, audio_dim) 
        self.fc_a = nn.Linear(config.hidden_size, 512)
        self.fc_v = nn.Linear(config.hidden_size, 512)
        self.fc_st = nn.Linear(config.hidden_size, 512)
        self.fc_fusion = nn.Linear(1024, 512)
        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_av_att = nn.Linear(512, 25)
        self.config = config

    def forward(self,rgb_feat, rgb_pos_feat, r2p1d_feat, r2p1d_pos_feat, audio_feat, audio_pos_feat, attention_mask, gather_index, rgb_masks, audio_masks, r2p1d_masks, task='avg'):
        final_embed = self.uniter(rgb_feat, rgb_pos_feat, r2p1d_feat, r2p1d_pos_feat, audio_feat, audio_pos_feat, attention_mask, gather_index, rgb_masks, audio_masks, r2p1d_masks)
        vid_embed = final_embed[:, :10, :]
        aud_embed = final_embed[:, 10:20, :]
        r2p_embed = final_embed[:, 20:, :]
        
        if task == 'avg':
            return final_embed, vid_embed, aud_embed, r2p_embed

        elif task == 'avvp':

            x1 = self.fc_a(aud_embed)

            # 2d and 3d visual feature fusion
            vid_s = self.fc_v(vid_embed)
            vid_st = self.fc_st(r2p_embed)
            x2 = torch.cat((vid_s, vid_st), dim =-1)
            x2 = self.fc_fusion(x2)
            x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2)
            frame_prob = torch.sigmoid(self.fc_prob(x))

            # attentive MMIL pooling
            frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
            av_att = torch.softmax(self.fc_av_att(x), dim=2)
            temporal_prob = (frame_att * frame_prob)
            global_prob = (temporal_prob*av_att).sum(dim=2).sum(dim=1)

            a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
            v_prob =temporal_prob[:, :, 1, :].sum(dim=1)
            
            return global_prob, a_prob, v_prob, frame_prob
        else:
            raise ValueError("Task is not defined")
        