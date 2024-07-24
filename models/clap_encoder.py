import random
import torch
import torch.nn as nn
import torchaudio
from models.CLAP.open_clip import create_model
from models.CLAP.training.data import get_audio_features
from transformers import RobertaTokenizer


class CLAP_Encoder(nn.Module):
    def __init__(
        self,
        pretrained_path='checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt',
        sampling_rate=32000,
        amodel="HTSAT-base",
        training=False,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.precision = "fp32"
        self.amodel = amodel  # or 'PANN-14'
        self.tmodel = "roberta"  # the best text encoder in our training
        self.enable_fusion = False  # False if you do not want to use the fusion model
        self.fusion_type = "aff_2d"
        self.pretrained = pretrained_path
        self.sampling_rate = sampling_rate
        self.tokenize = RobertaTokenizer.from_pretrained("roberta-base")
        
        self.model, self.model_cfg = create_model(
            self.amodel,
            self.tmodel,
            self.pretrained,
            precision=self.precision,
            device=self.device,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )

        if training is False:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        else:
            self.model.train()
        self.encoder_type = 'CLAP'

    def batch_to_list(self, batch):
        ret = []
        for i in range(batch.size(0)):
            ret.append(batch[i])
        return ret

    def _get_audio_embed(self, batch, training=False):

        if training is False:
            # batch: [B, samples]
            with torch.no_grad():
                audio_dict_list = []
                assert (
                    self.sampling_rate == 32000
                ), "We only support 32000 sampling rate"

                # batch: [bs, 1, t-samples]
                batch = torchaudio.functional.resample(
                    batch, orig_freq=self.sampling_rate, new_freq=48000
                )
                for waveform in self.batch_to_list(batch):
                    audio_dict = {}
                    audio_dict = get_audio_features(
                        audio_dict,
                        waveform,
                        480000,
                        data_truncating="fusion",
                        data_filling="repeatpad",
                        audio_cfg=self.model_cfg["audio_cfg"],
                    )
                    audio_dict_list.append(audio_dict)
                # [bs, 512]
                embed = self.model.get_audio_embedding(audio_dict_list)

                return embed.detach()
        else:
            audio_dict_list = []
            assert (
                    self.sampling_rate == 32000
            ), "We only support 32000 sampling rate"

            # batch: [bs, 1, t-samples]
            batch = torchaudio.functional.resample(
                batch, orig_freq=self.sampling_rate, new_freq=48000
            )
            for waveform in self.batch_to_list(batch):
                audio_dict = {}
                audio_dict = get_audio_features(
                    audio_dict,
                    waveform,
                    480000,
                    data_truncating="fusion",
                    data_filling="repeatpad",
                    audio_cfg=self.model_cfg["audio_cfg"],
                )
                audio_dict_list.append(audio_dict)
            # [bs, 512]
            embed = self.model.get_audio_embedding(audio_dict_list)

            return embed

    def _get_text_embed(self, batch, training=False):
        double_batch = False
        if len(batch) == 1:
            batch = batch * 2
            double_batch = True
        if training is False:
            with torch.no_grad():
                # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
                text_data = self.tokenizer(batch)
                embed = self.model.get_text_embedding(text_data)
        else:
            text_data = self.tokenizer(batch)
            embed = self.model.get_text_embedding(text_data)
        if double_batch:
            embed = embed[0].unsqueeze(0)
        
        return embed.detach() if training is False else embed


    def get_query_embed(self, modality, audio=None, text=None, use_text_ratio=0.5, device=None, training=False):
        if modality == 'audio':
            embed = self._get_audio_embed(audio, training=training)
        elif modality == 'text':
            embed = self._get_text_embed(text, training=training)
        elif modality == 'hybird':
            if random.random() > use_text_ratio:
                embed = self._get_audio_embed(audio, training=training)
            else:
                embed = self._get_text_embed(text, training=training)
        else:
            raise NotImplementedError("Please check flag 'training_modality'.")

        return embed.float()

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}
