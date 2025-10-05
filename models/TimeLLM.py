from math import sqrt
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from layers.stgr import SpatialGraphReprogramming
from utils.contextual_prompt import DynamicContextPromptBuilder

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0, quantiles: Optional[Sequence[float]] = None):
        super().__init__()
        self.n_vars = n_vars
        self.target_window = target_window
        self.flatten = nn.Flatten(start_dim=-2)
        self.quantiles = list(quantiles) if quantiles else None
        out_features = target_window if self.quantiles is None else target_window * len(self.quantiles)
        self.linear = nn.Linear(nf, out_features)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        if self.quantiles:
            bsz, n_vars, _ = x.shape
            x = x.view(bsz, n_vars, self.target_window, len(self.quantiles))
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.quantiles: Optional[List[float]] = None
        if hasattr(configs, 'quantiles') and configs.quantiles:
            self.quantiles = sorted({float(q) for q in configs.quantiles})
        self.median_quantile_index: Optional[int] = None
        if self.quantiles:
            self.median_quantile_index = int(
                min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))
            )

        self.enable_stgr = getattr(configs, 'enable_stgr', False)
        self.graph_heads = getattr(configs, 'graph_heads', 4)
        self.graph_dropout = getattr(configs, 'graph_dropout', configs.dropout)
        self.enable_dynamic_prompt = getattr(configs, 'enable_dynamic_prompt', False)
        self.dynamic_prompt_keys = getattr(configs, 'dynamic_prompt_keys', None)
        if isinstance(self.dynamic_prompt_keys, str):
            self.dynamic_prompt_keys = [self.dynamic_prompt_keys]

        self._graph_adj: Optional[torch.Tensor] = None
        self._graph_locations: Optional[Sequence[str]] = None
        self._external_context: Optional[Sequence] = None

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.spatial_reprogrammer: Optional[SpatialGraphReprogramming] = None
        if self.enable_stgr:
            hidden_channels = getattr(configs, 'graph_hidden', configs.d_model)
            self.spatial_reprogrammer = SpatialGraphReprogramming(
                num_nodes=configs.enc_in,
                in_channels=configs.d_model,
                hidden_channels=hidden_channels,
                heads=self.graph_heads,
                dropout=self.graph_dropout,
            )

        self.dynamic_prompt_builder = (
            DynamicContextPromptBuilder(self.dynamic_prompt_keys)
            if self.enable_dynamic_prompt
            else None
        )

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout,
                                                 quantiles=self.quantiles)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.quantiles:
                return dec_out[:, -self.pred_len:, :, :]
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        if self.enable_dynamic_prompt and self.dynamic_prompt_builder is not None:
            if self._external_context is not None:
                context_prompts = self.dynamic_prompt_builder(self._external_context, self.pred_len)
            else:
                context_prompts = ["No additional external context provided."] * len(prompt)
            prompt = [
                text.replace("<|<end_prompt>|>", f" External context: {context_prompts[idx]}<|<end_prompt>|>")
                for idx, text in enumerate(prompt)
            ]
            self._external_context = None

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        if self.enable_stgr and self.spatial_reprogrammer is not None:
            enc_out = self._apply_spatial_graph(enc_out, n_vars, x_enc.device)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])

        if self.quantiles:
            dec_out = dec_out.permute(0, 3, 1, 2).contiguous()
            dec_out = self._denormalize_quantiles(dec_out.to(torch.float32))
            return dec_out

        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def _apply_spatial_graph(self, enc_out: torch.Tensor, n_vars: int, device: torch.device) -> torch.Tensor:
        if enc_out.shape[0] % n_vars != 0:
            return enc_out
        batch = enc_out.shape[0] // n_vars
        patches = enc_out.shape[1]
        channels = enc_out.shape[2]
        spatial_dtype = self.spatial_reprogrammer.input_projection.weight.dtype
        spatial_input = enc_out.view(batch, n_vars, patches, channels).to(dtype=spatial_dtype)

        if self._graph_adj is None:
            adjacency = torch.eye(n_vars, device=device, dtype=spatial_dtype)
        else:
            if isinstance(self._graph_adj, torch.Tensor):
                adjacency = self._graph_adj.to(device=device, dtype=spatial_dtype)
            else:
                adjacency = torch.tensor(self._graph_adj, dtype=spatial_dtype, device=device)

            if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
                raise ValueError(
                    f"Loaded adjacency must be a square matrix, got shape {tuple(adjacency.shape)}."
                )

            adjacency_size = adjacency.shape[0]
            if adjacency_size == n_vars:
                pass
            elif adjacency_size < n_vars:
                if n_vars % adjacency_size != 0:
                    raise ValueError(
                        "Loaded adjacency size must evenly divide n_vars when it is smaller. "
                        f"Received adjacency with size {adjacency_size} for n_vars={n_vars}."
                    )
                repeat_factor = n_vars // adjacency_size
                adjacency = adjacency.repeat_interleave(repeat_factor, dim=0)
                adjacency = adjacency.repeat_interleave(repeat_factor, dim=1)
                assert adjacency.shape[0] == n_vars == adjacency.shape[1], (
                    "Expanded adjacency should match the number of variables after repeating."
                )
            else:
                raise ValueError(
                    f"Loaded adjacency size {adjacency_size} exceeds n_vars={n_vars}."
                )

        spatial_out = self.spatial_reprogrammer(spatial_input, adjacency)
        return spatial_out.view(batch * n_vars, patches, channels).to(enc_out.dtype)

    def _denormalize_quantiles(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = getattr(self.normalize_layers, 'mean', None)
        stdev = getattr(self.normalize_layers, 'stdev', None)
        if mean is None or stdev is None:
            return tensor
        mean = mean.to(tensor.device)
        stdev = stdev.to(tensor.device)
        tensor = tensor * stdev.unsqueeze(1).unsqueeze(-1)
        tensor = tensor + mean.unsqueeze(1).unsqueeze(-1)
        return tensor

    def set_graph_structure(self, adjacency, locations: Optional[Sequence[str]] = None):
        if adjacency is None:
            return
        if isinstance(adjacency, torch.Tensor):
            self._graph_adj = adjacency.float()
        else:
            self._graph_adj = torch.tensor(adjacency, dtype=torch.float32)
        self._graph_locations = locations

    def update_exogenous_context(self, context_batch: Sequence):
        self._external_context = context_batch


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
