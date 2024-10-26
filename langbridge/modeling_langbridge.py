# code adapted from flamingo-mini https://github.com/dhansmair/flamingo-mini

from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List
import contextlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformers import PreTrainedModel, AutoModel, AutoConfig, PreTrainedTokenizer, MT5EncoderModel, UMT5EncoderModel, XGLMForCausalLM, GPT2LMHeadModel, Qwen2ForCausalLM
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

from .configuration_langbridge import LangBridgeConfig
from .alignment_modules import LinearWithAddedEos, PerceiverResampler, FFNWithAddedEos, LinearNoEos


@contextlib.contextmanager
def suppress_model_loading_warnings(suppress: bool = True):
    if suppress:
        logger = logging.getLogger('transformers.modeling_utils')
        level = logger.level
        logger.setLevel(logging.CRITICAL)
        yield
        logger.setLevel(level)
    else:
        yield


class LBBaseModel(ABC, PreTrainedModel):

    config: LangBridgeConfig
    enc: PreTrainedModel
    lm: PreTrainedModel
    dec: PreTrainedModel
    dec_head: nn.Linear
    enc_embeddings: nn.Embedding

    config_class = LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True, suppress_warnings=True):
        super().__init__(config)
        #输入给Qwen模型的第i层（从0开始计算）
        self.enc_output_index = config.enc_output_index
        #输入给LLaMA模型的第i层（从0开始计算）
        self.lm_input_index = config.lm_input_index
        #LLaMA模型的第i层输出（从0开始计算）
        self.lm_output_index = config.lm_output_index
        #输入给第二个Qwen模型的第i层（从0开始计算）
        self.dec_input_index = config.dec_input_index
        if 'umt5' in config.enc.lower():
            enc_class = UMT5EncoderModel
        elif 'mt5' in config.enc.lower():
            enc_class = MT5EncoderModel
        elif 'mgpt' in config.enc.lower():
            enc_class = GPT2LMHeadModel
        elif 'qwen' in config.enc.lower():
            enc_class = Qwen2ForCausalLM
        else:
            enc_class = AutoModel

        if random_init:
            enc_config = AutoConfig.from_pretrained(config.enc)
            self.enc = enc_class(config=enc_config)
        else:
            print('loading encoder from pretrained')
            self.enc = enc_class.from_pretrained(config.enc)
            self.enc_embeddings = self.enc.get_input_embeddings()
            self.dec = enc_class.from_pretrained(config.enc)
            self.dec_head = self.dec.lm_head

        if config.alignments == 'linear':
            self.alignment_bottom = LinearNoEos(dim=config.dim_enc, out_dim=config.dim_lm)
            self.alignment_top = LinearNoEos(dim=config.dim_lm, out_dim=config.dim_enc)
        elif config.alignments == 'ffn':
            self.alignment_bottom = FFNWithAddedEos(dim=config.dim_enc, out_dim=config.dim_lm)
            self.alignment_top = FFNWithAddedEos(dim=config.dim_lm, out_dim=config.dim_enc)
        elif config.alignments == 'latent':
            self.alignment_bottom = PerceiverResampler(dim=config.dim_enc, out_dim=config.dim_lm, num_latents=config.num_latents)
            self.alignment_top = PerceiverResampler(dim=config.dim_lm, out_dim=config.dim_enc, num_latents=config.num_latents)
        else:
            raise ValueError(
                f'unknown alignment type {config.alignments}')

    def freeze_encoder(self):
        """freeze vision model """
        for param in self.enc.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        """freeze vision model """
        for param in self.dec.parameters():
            param.requires_grad = False

    def freeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    def forward(
        self,
        enc_ids: torch.Tensor,
        enc_mask: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> CausalLMOutputWithPast:
        batch_size, seq_length = enc_ids.shape[:2]
        device = input_ids.device if input_ids is not None else enc_ids.device

        if input_ids is not None:
            enc_ids = torch.cat([enc_ids, input_ids], dim=1)
            enc_mask = torch.cat([enc_mask, attention_mask], dim=1)


        # 通过第一个Qwen模型
        enc_out = self.enc(input_ids=enc_ids, attention_mask=enc_mask, output_hidden_states=True)
        enc_features = self.alignment_bottom(enc_out.hidden_states[self.enc_output_index + 1], enc_mask)

        # 通过MetaMath模型
        # 获取LLaMA模型的前5层输出
        lm_hidden_states = []
        for i in range(self.lm_input_index + 1):
            lm_hidden_state = torch.zeros_like(enc_features)
            lm_hidden_states.append(lm_hidden_state)
        
        # 将enc_features替换第self.lm_input_index层的输入
        lm_hidden_states[self.lm_input_index] = enc_features
        
        # 从第i层开始继续前向传播
        for i in range(self.lm_input_index, len(self.lm.layers)):
            # 根据报错信息，attention_mask的形状应该是(batch_size, 1, seq_len, seq_len)
            # 我们需要调整enc_mask的形状以匹配这个要求
            #adjusted_mask = enc_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, enc_features.shape[1], enc_features.shape[1])
            
            layer_outputs = self.lm.layers[i](
                hidden_states=lm_hidden_states[i],
                #attention_mask=adjusted_mask,
                output_attentions=False,
                use_cache=False,
            )
            lm_hidden_states.append(layer_outputs[0])

        lm_features = self.alignment_top(lm_hidden_states[self.lm_output_index + 1], enc_mask)


        # 通过第二个Qwen模型
        dec_hidden_states = []
        for i in range(self.dec_input_index + 1):
            dec_hidden_state = torch.zeros_like(lm_features)
            dec_hidden_states.append(dec_hidden_state)
        
        # 将enc_features替换第self.lm_input_index层的输入
        dec_hidden_states[self.dec_input_index] = lm_features
        
        # 从第i层开始继续前向传播
        for i in range(self.dec_input_index, len(self.dec.model.layers)):
            # 根据报错信息，attention_mask的形状应该是(batch_size, 1, seq_len, seq_len)
            # 我们需要调整enc_mask的形状以匹配这个要求
            #adjusted_mask = enc_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, enc_features.shape[1], enc_features.shape[1])
            
            layer_outputs = self.dec.model.layers[i](
                hidden_states=dec_hidden_states[i],
                #attention_mask=adjusted_mask,
                output_attentions=False,
                use_cache=False,
            )
            dec_hidden_states.append(layer_outputs[0])





        # dec_out = self.dec(attention_mask=attn_mask, inputs_embeds=lm_features, output_hidden_states=True)
        logits = self.dec_head(dec_hidden_states[-1])

        # 计算损失
        loss = None
        if labels is not None:
            #lm_feature_length = lm_features.shape[1]
            
            # no loss for soft prompts
            no_loss_labels = torch.full((batch_size, seq_length), -100, device=device, dtype=torch.long)

            # 将无损失标签与实际标签拼接
            full_labels = torch.cat([no_loss_labels, labels], dim=1)
            
            # 准备计算损失的logits和labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()

            # 计算损失
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction=loss_reduction)
            
            if loss_reduction == 'none':
                # 重塑损失以匹配批次大小和序列长度
                loss = rearrange(loss, '(b s) -> b s', b=batch_size)
                # 移除软提示部分的损失
                loss = loss[:, seq_length:]

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=dec_hidden_states,
        )


# used for debbuging with opt-125m
class LBOPT(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
        from transformers import OPTForCausalLM, OPTModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            base_lm: OPTForCausalLM = OPTForCausalLM(config=model_config)
        else:
            print('loading lm from pretrained')
            base_lm: OPTForCausalLM = OPTForCausalLM.from_pretrained(
                config.lm)
        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm=} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: OPTModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class LBLlama(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
        from transformers import LlamaForCausalLM, LlamaModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            try:
                model_config.attn_implementation = 'flash_attention_2'
                base_lm: LlamaForCausalLM = LlamaForCausalLM(
                    config=model_config)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: LlamaForCausalLM = LlamaForCausalLM(
                    config=model_config)
        else:
            print('loading lm from pretrained')
            try:
                base_lm: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
                    config.lm, use_flash_attention_2=True)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
                    config.lm)

        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm=} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: LlamaModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class LBMistral(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
        from transformers import MistralForCausalLM, MistralModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            try:
                model_config.attn_implementation = 'flash_attention_2'
                base_lm: MistralForCausalLM = MistralForCausalLM(
                    config=model_config)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: MistralForCausalLM = MistralForCausalLM(
                    config=model_config)
        else:
            try:
                base_lm: MistralForCausalLM = MistralForCausalLM.from_pretrained(
                    config.lm, use_flash_attention_2=True)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: MistralForCausalLM = MistralForCausalLM.from_pretrained(
                    config.lm)
        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm=} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: MistralModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()



class LangBridgeModel(PreTrainedModel):
    config: LangBridgeConfig
    config_class = LangBridgeConfig

    _LANGUAGE_MODEL_VERSIONS = {
        'facebook/opt': LBOPT,
        'EleutherAI/llemma': LBLlama,
        'codellama/CodeLlama': LBLlama,
        'microsoft/Orca-2': LBLlama,
        'meta-math/MetaMath': LBLlama,
        'meta-llama/Llama-2-7b-hf': LBLlama,
        'mistralai/Mistral-7B-v0.1': LBMistral,
    }

    def __init__(self, config: LangBridgeConfig, random_init=True, model_class=None):
        super().__init__(config)

        if model_class is None:
            model_class = self._find_lm_class(config.lm)
        self.lb: LBBaseModel = model_class(config, random_init=random_init)

        if config.freeze_language_model:
            self.freeze_lm()

        if config.freeze_encoder:
            self.freeze_encoder()
        if config.freeze_decoder:
            self.freeze_decoder()

    @classmethod
    def _find_lm_class(cls, language_model_id: str):
        for prefix, lm_class in cls._LANGUAGE_MODEL_VERSIONS.items():
            if language_model_id.startswith(prefix):
                return lm_class
        raise ValueError(f'unsupported language model {language_model_id}')

    def freeze_encoder(self):
        self.lb.freeze_encoder()

    def freeze_decoder(self):
        self.lb.freeze_decoder()


    def freeze_lm(self):
        self.lb.freeze_lm()

    def unfreeze_lm(self):
        self.lb.unfreeze_lm()

    def forward(
        self,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> CausalLMOutputWithPast:

        return self.lb(
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels=labels,
            loss_reduction=loss_reduction,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        past=None,
        past_key_values=None,
        **kwargs
    ) -> Dict[str, Any]:
        """ hf specific function. Overridden from PreTrainedModel for text generation purposes.

        if use_cache is used, past is not None, then only the last column will be passed as input_ids.
        TODO was `past` renamed to `past_key_values` in transformers 4.26?
        """

        if past_key_values is not None or past is not None:
            input_ids = input_ids[:, -1:]

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            past_key_values=past_key_values if past_key_values is not None else past,
            **kwargs
        )

    def _reorder_cache(self, past, beam_idx):
        """ hf specific function. Overridden from PreTrainedModel.

        this is required for beam search in combination with use_cache.

        Args: 
            past is a tuple of past_key_values of the xattn layers, and of the LM layers.
            beam_idx: index of the beam
        """
        xattn_past, lm_past = past

        xattn_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in xattn_past
        )

        lm_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in lm_past
        )

        return xattn_past_beam, lm_past_beam

    # a simple function to test the model
    @torch.no_grad()
    def generate_from_prefix(
        self,
        enc_tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        **kwargs
    ):
        enc_input = enc_tokenizer(prompts, return_tensors='pt', padding=True)
        enc_ids = enc_input['input_ids'].to(self.device)
        enc_mask = enc_input['attention_mask'].to(self.device)

        input_ids = torch.LongTensor([enc_tokenizer.bos_token_id])
        input_ids = input_ids.repeat(enc_ids.shape[0], 1).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        out_ids = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            early_stopping=True,
            use_cache=True,
            bos_token_id=enc_tokenizer.bos_token_id,
            eos_token_id=enc_tokenizer.eos_token_id,
            pad_token_id=enc_tokenizer.eos_token_id,
            **kwargs
        )

        completions = enc_tokenizer.batch_decode(
            out_ids, skip_special_tokens=True)
        # TODO: don't know why batch_decode doesn't remove <|im_end|>, since it's in the special tokens
        completions = [s.replace('<|im_end|>', '') for s in completions]
        return completions


