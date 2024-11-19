from transformers.configuration_utils import PretrainedConfig


class LangBridgeConfig(PretrainedConfig):

    def __init__(
        self,
        enc: str = 'DKYoon/mt5-base-lm-adapt',
        lm: str = 'facebook/opt-125m',
        dim_enc: int = 768,
        dim_lm: int = 768,
        freeze_language_model: bool = True,
        freeze_encoder: bool = True,
        freeze_decoder: bool = True,
        alignments: str = 'linear',
        enc_output_index: int = -1,
        lm_input_index: int = -1,
        lm_output_index: int = -1,
        dec_input_index: int = -1,
        training_stage: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lm = lm
        self.enc = enc
        self.dec = enc
        self.dim_enc = dim_enc
        self.dim_lm = dim_lm
        self.freeze_language_model = freeze_language_model
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.alignments = alignments
        self.enc_output_index = enc_output_index
        self.lm_input_index = lm_input_index
        self.lm_output_index = lm_output_index
        self.dec_input_index = dec_input_index
        self.training_stage = training_stage
