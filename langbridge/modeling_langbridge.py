# code adapted from flamingo-mini https://github.com/dhansmair/flamingo-mini

from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List
import contextlib
import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformers import PreTrainedModel, AutoModel, AutoConfig, PreTrainedTokenizer, MT5EncoderModel, UMT5EncoderModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

from .configuration_langbridge import LangBridgeConfig
from .alignment_modules import LinearWithAddedEos, PerceiverResampler, FFNWithAddedEos


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
    lm_head: nn.Linear
    embeddings: nn.Embedding

    config_class = LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True, suppress_warnings=True):
        super().__init__(config)
        if 'umt5' in config.enc.lower():
            enc_class = UMT5EncoderModel
        elif 'mt5' in config.enc.lower():
            enc_class = MT5EncoderModel
        else:
            enc_class = AutoModel

        with suppress_model_loading_warnings(suppress_warnings):
            if random_init:
                enc_config = AutoConfig.from_pretrained(
                    config.enc)
                self.enc = enc_class(config=enc_config)
            else:
                print('loading encoder from pretrained')
                self.enc = enc_class.from_pretrained(config.enc)

        # self.enc.gradient_checkpointing_enable(
            # gradient_checkpointing_kwargs={'use_reentrant': False})

        if config.alignments == 'linear':  # default
            self.alignment = LinearWithAddedEos(
                dim=config.dim_enc, out_dim=config.dim_lm)
        elif config.alignments == 'ffn':  # mlp
            self.alignment = FFNWithAddedEos(
                dim=config.dim_enc, out_dim=config.dim_lm)
        elif config.alignments == 'latent':
            self.alignment = PerceiverResampler(
                dim=config.dim_enc, out_dim=config.dim_lm, num_latents=config.num_latents)
        else:
            raise ValueError(
                f'unknown alignment type {config.alignments}')

    def freeze_encoder(self):
        """freeze vision model """
        for param in self.enc.parameters():
            param.requires_grad = False

    def freeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    # get soft prompts
    def get_encoder_features(self, enc_ids: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        if self.config.freeze_encoder:
            with torch.no_grad():
                enc_features = self.enc(
                    input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state  # (b, s, d)
        else:
            enc_features = self.enc(
                input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state
        enc_features = self.alignment(enc_features, enc_mask)
        return enc_features

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
        # sanity check
        assert return_dict, "can only use return_dict=True at the moment!"

        # find the input shape
        batch_size, seq_length = input_ids.shape[:
                                                 2] if input_ids is not None else enc_ids.shape[:2]
        device = input_ids.device if input_ids is not None else enc_ids.device
        lm_past_key_values = None if past_key_values is None else past_key_values[0]

        if input_ids is not None:
            embeddings = self.embeddings(input_ids)
        bos_shifted = False
        if lm_past_key_values is None:
            assert enc_ids.size(0) == batch_size

            enc_features = self.get_encoder_features(
                enc_ids, enc_mask)

            if input_ids is not None:
                first_input_ids = input_ids[:, 0]
                if all(first_input_ids == self.lm.config.bos_token_id):
                    # move bos embedding to the front
                    bos_shifted = True
                    embeddings = torch.cat(
                        [embeddings[:, 0].unsqueeze(dim=1), enc_features, embeddings[:, 1:]], dim=1)
                else:
                    embeddings = torch.cat([enc_features, embeddings], dim=1)
            else:
                embeddings = enc_features
            enc_feature_length = enc_features.shape[1]
        else:
            enc_feature_length = past_key_values[1]

        if input_ids is not None:
            if self.config.alignments not in ['linear', 'ffn']:
                attn_mask = torch.cat(
                    [torch.ones((batch_size, enc_feature_length), device=device, dtype=torch.long), attention_mask], dim=1)
            else:
                # use the encoder masks for the soft prompts
                if bos_shifted:
                    # TODO: messy code
                    # torch.ones are there since the alignment adds a single learnable eos token at the enc
                    attn_mask = torch.cat(
                        [attention_mask[:, 0].unsqueeze(dim=1), enc_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long), attention_mask[:, 1:]], dim=1)
                else:
                    attn_mask = torch.cat(
                        [enc_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long), attention_mask], dim=1)
        else:
            attn_mask = enc_mask

        # pass through LM
        out: BaseModelOutputWithPast = self.lm(
            attention_mask=attn_mask,
            inputs_embeds=embeddings,
            use_cache=use_cache,
            past_key_values=lm_past_key_values,
            return_dict=True,
            **kwargs
        )

        logits: torch.Tensor = self.lm_head(out.last_hidden_state)

        loss = None
        if labels is not None:
            # no loss for soft prompts
            no_loss_labels = torch.zeros(
                (batch_size, enc_feature_length), device=device, dtype=torch.long) + -100

            if bos_shifted:
                full_labels = torch.cat(
                    [labels[:, 0].unsqueeze(dim=1), no_loss_labels, labels[:, 1:]], dim=1)
            else:
                full_labels = torch.cat(
                    [no_loss_labels, labels], dim=1)
            # logits shape (batch, seq_length, #words)
            shift_logits = logits[..., :-1, :].contiguous()
            # labels shape (batch, seq_length)
            shift_labels = full_labels[..., 1:].contiguous()

            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction=loss_reduction)
            if loss_reduction == 'none':
                # CrossEntropyLoss will flatten all dimensions by default
                loss = rearrange(loss, '(b s) -> b s', b=batch_size)
                # remove soft promtps from loss
                loss = loss[:, enc_feature_length:]
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(out.past_key_values,
                             enc_feature_length) if use_cache else None,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )

    def lm_forward(
        self,
        base_inputs,
        return_dict=None
    ):
        base_outputs = self.model(**base_inputs, return_dict=return_dict)
        return base_outputs

    # def generate(
    #     self,
    #     input_ids: Optional[torch.Tensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     max_new_tokens: Optional[int] = 100,
    #     do_sample: bool = False,
    #     top_p: float = 1.0,
    #     temperature: float = 1.0,
    #     logits_processor: Optional[LogitsProcessorList] = None,
    #     stopping_criteria: Optional[StoppingCriteriaList] = None,
    #     return_logits_for_analysis: bool = False,
    #     **kwargs
    # ):
    #     base_kwargs = kwargs.copy()
    #     base_input_ids = input_ids.to(base_input_ids.device)
    #     base_kwargs['attention_mask'] = attention_mask

    #     # keep track of which sequences are already finished
    #     unfinished_sequences = torch.ones(base_input_ids.shape[0], dtype=torch.long, device=base_input_ids.device)
    #     eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(base_input_ids.device)

    #     if return_logits_for_analysis:
    #         analysis_data = defaultdict(list)
        
    #     input_length = len(base_input_ids[0])
    #     cal = True
    #     for step in range(max_new_tokens):
    #         # prepare model inputs with past_key_values and attention_mask
    #         base_inputs = self.prepare_inputs_for_generation(base_input_ids, **base_kwargs)            
    #         base_outputs,  = self.forward(
    #             base_inputs, return_dict=True
    #         )
    #             base_next_token_logits = base_outputs.logits[..., -1, :]
    #             pos_next_token_logits = pos_outputs.logits[..., -1, :]
    #             neg_next_token_logits = neg_outputs.logits[..., -1, :]

    #             # sometimes our experts have extra (irrelevant) tokens at the end of the normal vocabulary
    #             pos_next_token_logits = pos_next_token_logits[:, :base_next_token_logits.shape[-1]]
    #             neg_next_token_logits = neg_next_token_logits[:, :base_next_token_logits.shape[-1]]
    #             # DExperts!
    #             if method == "all_log_softmax":
    #                 base_next_token_logits = F.log_softmax(base_next_token_logits, dim=-1)
    #                 pos_next_token_logits = F.log_softmax(pos_next_token_logits, dim=-1)
    #                 neg_next_token_logits = F.log_softmax(neg_next_token_logits, dim=-1)
    #             elif method == "pos_neg_log_softmax":
    #                 pos_next_token_logits = F.log_softmax(pos_next_token_logits, dim=-1)
    #                 neg_next_token_logits = F.log_softmax(neg_next_token_logits, dim=-1)
    #             elif method == "all_softmax":
    #                 base_next_token_logits = F.softmax(base_next_token_logits, dim=-1)
    #                 pos_next_token_logits = F.softmax(pos_next_token_logits, dim=-1)
    #                 neg_next_token_logits = F.softmax(neg_next_token_logits, dim=-1)
    #             elif method == "pos_neg_softmax":
    #                 pos_next_token_logits = F.softmax(pos_next_token_logits, dim=-1)
    #                 neg_next_token_logits = F.softmax(neg_next_token_logits, dim=-1)
    #             entropy_base = compute_entropy(base_next_token_logits).unsqueeze(dim=1)
    #             #entropy_base = torch.where(entropy_base < 0.1, torch.tensor(0.0).to(entropy_base.device), entropy_base)
    #             if method == "1":
    #                 #entropy_base = torch.where(entropy_base <= 0.5, torch.tensor(1).to(entropy_base.device), entropy_base)
    #                 entropy_base = torch.where(entropy_base >= 0.5, torch.tensor(1).to(entropy_base.device), entropy_base)
    #             elif method == "2":
    #                 entropy_base = torch.where(entropy_base >= self.threshold, torch.tensor(2).to(entropy_base.device), entropy_base)
    #             elif method == "0":
    #                 entropy_base = entropy_base
                    
    #             else:
    #                 print("method must be '1' or '2' or 0")
    #             if weight_method == "entropy":
    #                 next_token_logits = (
    #                     base_next_token_logits +
    #                     entropy_base * (pos_next_token_logits - neg_next_token_logits)
    #                 )
    #             elif weight_method == "alpha":
    #                 next_token_logits = (
    #                     base_next_token_logits +
    #                     self.alpha * (pos_next_token_logits - neg_next_token_logits)
    #                 )
    #             else:
    #                 raise ValueError("weight_method must be 'entropy' or 'alpha'")
    #         else:
    #             base_outputs = self.forward(
    #                 base_inputs, return_dict=True)
    #             pos_outputs, neg_outputs = None, None
    #             base_next_token_logits = base_outputs.logits[..., -1, :]
                
    #             # DExperts!
    #             next_token_logits = (
    #                 base_next_token_logits
    #             )
           
    #         # pre-process logits
    #         # if logits_processor:
    #         #     next_token_logits = logits_processor(input_ids, next_token_logits)
    #         # top_k_values, top_k_indices = torch.topk(base_next_token_logits, k=5, dim=-1)
    #         # top_k_values, top_k_indices = torch.topk(pos_next_token_logits, k=5, dim=-1)
    #         # top_k_values, top_k_indices = torch.topk(neg_next_token_logits, k=5, dim=-1)
    #         # top_k_values, top_k_indices = torch.topk(pos_next_token_logits - neg_next_token_logits, k=5, dim=-1)
    #         # top_k_values, top_k_indices = torch.topk(next_token_logits, k=5, dim=-1)
    #         # self.tokenizer.batch_decode(top_k_indices[0])
    #         # pre-process logits
    #         # if logits_processor:
    #         #     next_token_logits = logits_processor(input_ids, next_token_logits)
    #         # warp logits
    #         if temperature != 1.0:
    #             next_token_logits = next_token_logits / temperature
    #         if top_p < 1.0:
    #             next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

    #         # decode
    #         if do_sample:
    #             probs = F.softmax(next_token_logits, dim=-1)
    #             next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    #         else:
    #             next_tokens = torch.argmax(next_token_logits, dim=-1)

    #         next_tokens = (
    #             next_tokens * unfinished_sequences +
    #             self.tokenizer.pad_token_id * (1 - unfinished_sequences)
    #         )
    #         if return_logits_for_analysis:
    #             next_token_logits_dict = {
    #                 'dexperts': next_token_logits,
    #                 'base': base_next_token_logits,
    #                 'pos': pos_next_token_logits,
    #                 'neg': neg_next_token_logits
    #             }
    #             analysis_data = self.update_analysis_data(analysis_data, next_tokens, next_token_logits_dict)

    #         # update model inputs for next step
    #         base_input_ids = torch.cat([base_input_ids, next_tokens[:, None]], dim=-1)
    #         pos_input_ids = torch.cat([pos_input_ids, next_tokens[:, None]], dim=-1)
    #         neg_input_ids = torch.cat([neg_input_ids, next_tokens[:, None]], dim=-1)

    #         # update kwargs
    #         base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
    #         pos_kwargs = self._update_model_kwargs_for_generation(pos_outputs, pos_kwargs)
    #         neg_kwargs = self._update_model_kwargs_for_generation(neg_outputs, neg_kwargs)

    #         # stopping criteria
    #         if stopping_criteria and stopping_criteria(base_input_ids, None):
    #             break

    #         # if eos_token was found in one sentence, set sentence to finished
    #         unfinished_sequences = unfinished_sequences.mul(
    #             next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
    #         )

    #         # stop when each sentence is finished
    #         if unfinished_sequences.max() == 0:
    #             break
    #     if return_logits_for_analysis:
    #         for k in analysis_data.keys():
    #             if k.startswith('logits'):
    #                 analysis_data[k] = torch.cat(analysis_data[k], dim=1)
    #         return base_input_ids, analysis_data
        
    #     return base_input_ids

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

    @classmethod
    def _find_lm_class(cls, language_model_id: str):
        for prefix, lm_class in cls._LANGUAGE_MODEL_VERSIONS.items():
            if language_model_id.startswith(prefix):
                return lm_class
        raise ValueError(f'unsupported language model {language_model_id}')

    def freeze_encoder(self):
        self.lb.freeze_encoder()

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
        lm_tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        **kwargs
    ):
        enc_input = enc_tokenizer(prompts, return_tensors='pt', padding=True)
        enc_ids = enc_input['input_ids'].to(self.device)
        enc_mask = enc_input['attention_mask'].to(self.device)

        input_ids = torch.LongTensor([lm_tokenizer.bos_token_id])
        input_ids = input_ids.repeat(enc_ids.shape[0], 1).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        out_ids = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            early_stopping=True,
            use_cache=True,
            bos_token_id=lm_tokenizer.bos_token_id,
            eos_token_id=32002,  # <|im_end|>
            pad_token_id=lm_tokenizer.eos_token_id,
            **kwargs
        )

        completions = lm_tokenizer.batch_decode(
            out_ids, skip_special_tokens=True)
        # TODO: don't know why batch_decode doesn't remove <|im_end|>, since it's in the special tokens
        completions = [s.replace('<|im_end|>', '') for s in completions]
        return completions
