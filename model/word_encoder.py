import logging
import torch
from transformers import BertPreTrainedModel, BertConfig, BertModel

from .prompt_encoder import PromptEncoder

logger = logging.getLogger(__name__)

class BERTWordEncoder(BertPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    
    def __init__(self, config: BertConfig, args):
        super().__init__(config)
        self.config = config
        self.args = args
        self.bert = BertModel(config)

        self.tokenizer = self.args.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.embeddings = self.get_input_embeddings()

        self.pseudo_token_id = None
        
        self.tokenizer.add_special_tokens({'additional_special_tokens': [args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]

        self.prompt = args.prompt

        if self.prompt == 0:
            self.spell_length = self.args.N + 1
            self.prompt_encoder = PromptEncoder(self.spell_length, self.embeddings.embedding_dim)

        self.init_weights()

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings
    
    def embed_input(self, sentences):
        sentences_for_embedding = sentences.clone()
        sentences_for_embedding[(sentences == self.pseudo_token_id)] = self.tokenizer.unk_token_id

        raw_embeds = self.embeddings(sentences_for_embedding)

        if self.prompt == 0 and self.pseudo_token_id is not None:
            bz = sentences.shape[0]
            blocked_indices = torch.nonzero(sentences == self.pseudo_token_id).reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
            replace_embeds = self.prompt_encoder(torch.LongTensor(list(range(self.spell_length))).to(sentences.device))

            for bidx in range(bz):
                for i in range(self.spell_length):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        return raw_embeds

    def forward(self, input_ids):

        inputs_embeds = self.embed_input(input_ids)
        attention_mask = (input_ids != self.pad_token_id).bool()

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_state = outputs.last_hidden_state
        # last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs.hidden_states[-4:]], 0)
        # hidden_state = torch.sum(last_four_hidden_states, 0)
        return hidden_state
