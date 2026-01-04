## Captioning Process in sd-scripts Backend
This codemap traces the captioning process in sd-scripts from raw caption loading through augmentation, tokenization, and caching. Key locations include the caption preprocessing logic [2a-2e], the strategy pattern for tokenization [3a-3c], and the caching mechanism [4a-4c].
### 1. Caption Loading and Preprocessing
How captions are loaded and initially processed during data loading
### 1a. Load Original Caption (`dataset.py:1042`)
Retrieve the base caption from image metadata
```text
caption = image_info.caption  # default
```
### 1b. Process Caption When Needed (`dataset.py:1062`)
Apply caption processing only when tokenization is required
```text
if tokenization_required:
                caption = self.process_caption(subset, image_info.caption)
```
### 1c. Add to Batch (`dataset.py:1092`)
Include processed caption in the training batch
```text
captions.append(caption)
```
### 2. Caption Augmentation Pipeline
The complete caption processing including dropout, shuffling, and wildcard replacement
### 2a. Add Prefix/Suffix (`dataset.py:194`)
Apply configured prefix and suffix to captions
```text
if subset.caption_prefix:
            caption = subset.caption_prefix + " " + caption
        if subset.caption_suffix:
            caption = caption + " " + subset.caption_suffix
```
### 2b. Caption Dropout Check (`dataset.py:200`)
Randomly drop entire captions based on dropout rate
```text
is_drop_out = subset.caption_dropout_rate > 0 and random.random() < subset.caption_dropout_rate
```
### 2c. Wildcard Processing (`dataset.py:211`)
Handle multiline captions and wildcard replacements
```text
if subset.enable_wildcard:
                # if caption is multiline, random choice one line
                if "\n" in caption:
                    caption = random.choice(caption.split("\n"))
```
### 2d. Token Shuffling (`dataset.py:283`)
Randomly shuffle the flexible tokens in the caption
```text
if subset.shuffle_caption:
                    random.shuffle(flex_tokens)
```
### 2e. Reconstruct Caption (`dataset.py:288`)
Reassemble the caption after processing
```text
caption = ", ".join(fixed_tokens + flex_tokens + fixed_suffix_tokens)
```
### 3. Tokenization Strategy Pattern
How captions are converted to token IDs using different strategies
### 3a. Tokenize Caption (`dataset.py:1064`)
Convert processed caption to token IDs using strategy pattern
```text
input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]
```
### 3b. SDXL Tokenization (`strategy_sdxl.py:28`)
SDXL uses dual tokenizers for text encoding
```text
def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        return (
            torch.stack([self._get_input_ids(self.tokenizer1, t, self.max_length) for t in text], dim=0),
            torch.stack([self._get_input_ids(self.tokenizer2, t, self.max_length) for t in text], dim=0),
        )
```
### 3c. SD Tokenization (`strategy_sd.py:35`)
SD1.x/2.x uses single tokenizer
```text
def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        return [torch.stack([self._get_input_ids(self.tokenizer, t, self.max_length) for t in text], dim=0)]
```
### 4. Text Encoder Caching
How text encoder outputs are cached to avoid recomputation
### 4a. Check Cache (`dataset.py:1050`)
Load cached outputs from memory or disk if available
```text
if image_info.text_encoder_outputs is not None:
                # cached
                text_encoder_outputs = image_info.text_encoder_outputs
            elif image_info.text_encoder_outputs_npz is not None:
                # on disk
                text_encoder_outputs = self.text_encoder_output_caching_strategy.load_outputs_npz(
                    image_info.text_encoder_outputs_npz
                )
```
### 4b. Encode and Cache (`caching.py:186`)
Generate text encoder outputs for caching
```text
with torch.no_grad():
        b_hidden_state1, b_hidden_state2, b_pool2 = get_hidden_states_sdxl(
            max_token_length,
            input_ids1,
            input_ids2,
            tokenizers[0],
            tokenizers[1],
            text_encoders[0],
            text_encoders[1],
            dtype,
        )
```
### 4c. Store Cached Outputs (`caching.py:203`)
Save outputs to memory or disk
```text
for info, hidden_state1, hidden_state2, pool2 in zip(image_infos, b_hidden_state1, b_hidden_state2, b_pool2):
        if cache_to_disk:
            save_text_encoder_outputs_to_disk(info.text_encoder_outputs_npz, hidden_state1, hidden_state2, pool2)
        else:
            info.text_encoder_outputs1 = hidden_state1
            info.text_encoder_outputs2 = hidden_state2
            info.text_encoder_pool2 = pool2
```
### 5. Dataset Integration
How the captioning system integrates with the overall dataset loading
### 5a. Add Captions to Output (`dataset.py:1166`)
Include processed captions in the training example
```text
example["captions"] = captions
```
### 5b. Add Token IDs (`dataset.py:1139`)
Include tokenized input IDs in the training example
```text
example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)
```
### 5c. Add Cached Outputs (`dataset.py:1138`)
Include cached text encoder outputs when available
```text
example["text_encoder_outputs_list"] = none_or_stack_elements(text_encoder_outputs_list, torch.FloatTensor)
```