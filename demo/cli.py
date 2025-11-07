import json
from infer import LiveCCDemoInfer

if __name__ == '__main__':
    model_path = '/orcd/scratch/orcd/002/qua/data/reaction_data/checkpoints/livecc_lr1e-5_20250718_180349/checkpoint-410'
    # video_path = "/orcd/scratch/orcd/002/qua/data/reaction_data/kal_21/output_segments/cropped/segment_27_label_2_crop.mp4"
    video_path = "/orcd/home/002/qua/code/reaction/livecc/test1.mp4"
    query = """You are watching a short video. As you watch, continuously and in real time describe: your inner thoughts and reasoning inside <think></think> tags; concise reactions phrases, including emotions, facial expressions, body movements, or gestures, inside <expr></expr> tags, no subjects, verbs, or filler. Your output must strictly use <think> and <expr> blocks, and only this format: <think>...</think><expr>...</expr>."""
    # model_path = 'chenjoya/LiveCC-7B-Instruct'
    # query = """Please describe the video."""

    tokens = ["<think>", "</think>", "<expr>", "</expr>", "<|im_start|>", "<|im_end|>"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    token_ids = [tokenizer.encode(token) for token in tokens]
    for i, token in enumerate(tokens):
        print(token, token_ids[i])

    infer = LiveCCDemoInfer(model_path=model_path)
    state = {'video_path': video_path}
    commentaries = []
    t = 0
    for t in range(31):
        state['video_timestamp'] = float(t)
        for (start_t, stop_t), response, state in infer.live_cc(
            message=query, state=state, 
            max_pixels = 4816896, repetition_penalty=1.21, 
            streaming_eos_base_threshold=0, streaming_eos_threshold_step=-0.1
        ):
            print(f'{start_t}s-{stop_t}s: {response}')
            commentaries.append([start_t, stop_t, response])
        if state.get('video_end', False):
            break
        t += 1
    result = {'video_path': video_path, 'query': query, 'commentaries': commentaries}
    result_path = video_path.replace('/assets/', '/results/').replace('.mp4', '.json')
    print(f"{video_path=}, {query=} => {model_path=} => {result_path=}")
    json.dump(result, open(result_path, 'w'))