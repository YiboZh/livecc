from transformers import Qwen2_5OmniProcessor
processor = Qwen2_5OmniProcessor.from_pretrained('Qwen/Qwen2.5-Omni-7B', padding_side='right')
# Test what keys the processor returns
conversation = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]}]
texts = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
inputs = processor(text=texts, return_tensors='pt')
print('Keys returned by processor:', list(inputs.keys()))
for key, value in inputs.items():
    if hasattr(value, 'shape'):
        print(f'{key}: shape={value.shape}')
    else:
        print(f'{key}: {type(value)}')