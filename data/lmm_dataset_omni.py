"""
LMMDatasetOmni - Simplified dataset for Qwen2.5-Omni models

This dataset class is specifically designed for the Qwen2.5-Omni model, which can process
both vision and audio from video data. It's significantly simpler than the original LMMDataset
because the Qwen Omni processor handles most of the multimodal preprocessing internally.
"""

from dataclasses import dataclass, field
import json, torch, random, tqdm, os, tempfile, hashlib
from torch.utils.data import Dataset
from transformers import logging, Qwen2_5OmniProcessor

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_omni_utils_patch import process_mm_info

import subprocess
import av
from concurrent.futures import ThreadPoolExecutor

import traceback
logger = logging.get_logger(__name__)


# --- some utils ---
def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2) # avoid last \n
        while f.read(1) != b"\n":  
            f.seek(-2, 1)
        return f.readline()
# --- some utils ---

@dataclass
class DataArgumentsOmni:
    annotation_paths: list[str] = field(default_factory=list)
    use_audio_in_video: bool = False
    max_dataset_retries: int = 32

class LMMDatasetOmni(Dataset):
    def __init__(
        self, *, 
        annotation_paths: list[str], 
        processor: Qwen2_5OmniProcessor, 
        use_audio_in_video: bool = DataArgumentsOmni.use_audio_in_video,
        max_dataset_retries: int = DataArgumentsOmni.max_dataset_retries,
        root_path: str = "",
        **kwargs
    ):
        super().__init__()
        self.handles = []
        for annotation_path in annotation_paths:
            assert annotation_path.endswith('.jsonl'), f"Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
            logger.warning(f'Load {annotation_path}. Please ensure its last line stores the seek indices...')
            seeks = json.loads(readlastline(annotation_path))
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            logger.warning(f'Successfully loaded {annotation_path}')
        
        # Get special token IDs for label masking
        if 'Qwen' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer(
                '<|im_start|>assistant\n<|im_end|>').input_ids
        else:
            raise NotImplementedError(f"Processor not implemented for {processor.__class__.__name__}")
        
        self.processor = processor
        self.use_audio_in_video = False
        self.root_path = root_path
        self.max_dataset_retries = max(1, max_dataset_retries)
    
    def load_conversation(self, index):
        annotation_path, seek = self.handles[index]
        with open(annotation_path) as f:
            f.seek(seek)
            line = f.readline()
        line = json.loads(line)
        return line
    
    # def _clip_video(self, video_path, video_start, video_end, tmp_path):
    #     subprocess.run([
    #         'ffmpeg', '-y', '-loglevel', 'error',
    #         '-ss', str(video_start),
    #         '-t', str(video_end - video_start),
    #         '-i', video_path,
    #         '-map', '0',      # copy all streams (video + audio)
    #         '-c', 'copy',     # no re-encode → keeps fps, codec, timestamps
    #         tmp_path
    #     ], check=True)
    #     return os.path.abspath(tmp_path)
    
    def _clip_video(self, video_path, video_start, video_end, tmp_path, fps=2):
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-ss', str(video_start),
            '-t', str(video_end - video_start),
            '-i', video_path,
            '-r', str(fps),              # set output fps
            '-map', '0:v:0',             # select only the video stream
            '-c:v', 'libx264',           # re-encode video to ensure fps
            '-c:a', 'copy',             # keep original audio
            tmp_path
        ], check=True)
        return os.path.abspath(tmp_path)
    
    

    
    def _has_audio(self, video_path: str) -> bool:
        container = av.open(video_path)
        return any(s.type == 'audio' for s in container.streams)

    def preprocess_conversation(self, conversation):
        video_ratios = []
        instruction = ""
        video_path = conversation[0]['content'][1]['video']
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        has_audio = self._has_audio(video_path)
        
        # import pdb; pdb.set_trace()
        
        new_conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": [{"type": "text", "text": instruction}]}
        ]
        
        # Collect video clipping tasks
        clip_tasks = []
        tmp_files = []
        for idx, message in enumerate(conversation):
            if message['role'] == 'user':
                video_item = message['content'][1]
                if video_item['video_end'] - video_item['video_start'] <= 0:
                    continue  # skip invalid segments
                    
                tmp_path = os.path.join(tempfile.gettempdir(), f"{video_hash}_{idx}_{os.getpid()}.mp4")
                tmp_files.append(tmp_path)
                clip_tasks.append((video_path, video_item['video_start'], video_item['video_end'], tmp_path))
                video_ratios.append(video_item['video_end'] - video_item['video_start'])
        
        # Clip videos concurrently
        with ThreadPoolExecutor(max_workers=min(len(clip_tasks), 4)) as executor:
            futures = [executor.submit(self._clip_video, *task) for task in clip_tasks]
            [f.result() for f in futures]  # Wait for all to complete
        
        # Build conversation with clipped videos
        tmp_idx = 0
        for message in conversation:
            if message['role'] == 'user':
                video_item = message['content'][1]
                if video_item['video_end'] - video_item['video_start'] <= 0:
                    continue  # skip invalid segments
                new_conversation.append({"role": "user", "content": [{"type": "video", "video": f"file://{tmp_files[tmp_idx]}"}]})
                tmp_idx += 1
            else:
                text_item = message['content'][0]
                new_conversation.append({"role": "assistant", "content": [{"type": "text", "text": text_item['text']}]})

        video_ratios = [i / sum(video_ratios) for i in video_ratios]
        return new_conversation, tmp_files, has_audio, video_ratios
    

    def getitem(self, index):
        """
        Process a single data sample.
        
        This method is much simpler than LMMDataset.getitem because:
        - No need to manually preprocess videos with decord
        - No need to handle streaming conversation logic
        - process_mm_info handles video/audio extraction
        - Qwen Omni processor handles all multimodal preprocessing
        
        The processor automatically:
        - Extracts frames from videos at appropriate FPS
        - Extracts audio from videos (if use_audio_in_video=True)
        - Resizes and normalizes visual inputs
        - Processes audio into appropriate format
        """
        raw_conversation = self.load_conversation(index)
        # import pdb; pdb.set_trace()

        conversation, tmp_files, has_audio, video_ratios = self.preprocess_conversation(raw_conversation)
        try:
            audios, images, videos = process_mm_info(conversation, video_ratios=video_ratios, use_audio_in_video=self.use_audio_in_video and has_audio)
            texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            inputs = self.processor(text=texts, audio=audios, images=images, videos=videos, 
                                   return_tensors="pt", padding=True, use_audio_in_video=self.use_audio_in_video and has_audio)
            
            # import pdb; pdb.set_trace()
            
            # Create labels for training (mask everything except assistant responses)
            input_ids = inputs.input_ids
            labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
            im_start_idxs = (input_ids == self.im_start_id).nonzero()
            im_end_idxs = (input_ids == self.im_end_id).nonzero()
            
            # import pdb; pdb.set_trace()
            if input_ids.shape[1] > 30000:
                # import pdb; pdb.set_trace()
                print("Video path: ", raw_conversation)
            
            for (sample_idx, im_start_idx), (sample_idx, im_end_idx) in zip(im_start_idxs, im_end_idxs):
                if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                    labels[sample_idx, im_start_idx+3:im_end_idx+1] = input_ids[sample_idx, im_start_idx+3:im_end_idx+1]
            
            inputs['labels'] = labels
            return inputs
        finally:
            # Clean up tmp files
            for tmp_file in tmp_files:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

    def __getitem__(self, index):
        max_tries = self.max_dataset_retries
        for _ in range(max_tries):
            try:
                return self.getitem(index)
            except KeyboardInterrupt:
                # User interrupted - don't retry, just re-raise
                logger.info("Interrupted by user - exiting")
                raise
            except Exception as e:
                traceback.print_exc()
                logger.warning(f"Failed {_}-th try to get item {index}: {e}")
                index = random.randint(0, self.__len__() - 1)
                logger.warning(f"Retrying to get item {index}")
        raise Exception(f"Failed to get item after {max_tries} retries")

    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1
        return batched_inputs[0]

    def __len__(self):
        return len(self.handles)

if __name__ == "__main__":
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
    
    processor = Qwen2_5OmniProcessor.from_pretrained('Qwen/Qwen2.5-Omni-7B', padding_side='right')
    
    dataset = LMMDatasetOmni(
        annotation_paths=[
            # "/home/qua/code/reaction/livecc/data/reaction_clean/livecc_reactions_clean.jsonl"
            "/orcd/scratch/seedfund/001/multimodal/qua/reaction_data/livecc_reaction_stream_no_music_10000.jsonl"
            # '/orcd/scratch/orcd/002/qua/data/reaction_data/output_conversation_rewritten.jsonl',
            
            # "/orcd/scratch/seedfund/001/multimodal/qua/reaction_data/livecc_reaction_rewritten_stream_10000.jsonl"
        ], 
        processor=processor,
        use_audio_in_video=True,
    )
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=dataset.data_collator)
    
    # Test a sample
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    for sample in dataloader:
        print(f"Batch keys: {sample.keys()}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")

