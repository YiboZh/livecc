# NOTE: Some parts were borrowed from qwen_vl_utils. We modified them for better use in LiveCC.
# Feel free to contact joyachen@u.nus.edu for any problems. Thank you!

import os, torch
import numpy as np
import decord # NOTE: import decord should be after torch, otherwise seg fault
from transformers import logging
from torchvision import transforms

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord+'
os.environ['VIDEO_MAX_PIXELS'] = str(int(os.environ.get('VIDEO_MAX_PIXELS', 24576 * 28 * 28))) # increase this for streaming. 24576 * 28 * 28 = 19267584
import qwen_vl_utils.vision_process as qwen_vp

IMAGE_PATCH_SIZE = int(os.environ.get('IMAGE_PATCH_SIZE', 14))
SPATIAL_MERGE_SIZE = getattr(qwen_vp, "SPATIAL_MERGE_SIZE", 2)
IMAGE_FACTOR = getattr(qwen_vp, "IMAGE_FACTOR", IMAGE_PATCH_SIZE * SPATIAL_MERGE_SIZE)
setattr(qwen_vp, "IMAGE_FACTOR", IMAGE_FACTOR)

def _resolve_pixels(attr_name: str, fallback: int) -> int:
    """Fetch constant from qwen_vl_utils with env override fallback."""
    value = getattr(qwen_vp, attr_name, fallback)
    env_value = os.environ.get(attr_name)
    if env_value is not None:
        value = int(env_value)
    setattr(qwen_vp, attr_name, value)
    return value

VIDEO_MIN_PIXELS = _resolve_pixels(
    "VIDEO_MIN_PIXELS",
    getattr(qwen_vp, "VIDEO_MIN_TOKEN_NUM", 128) * IMAGE_FACTOR * IMAGE_FACTOR,
) # follow qwen2vl paper
VIDEO_MAX_PIXELS = _resolve_pixels(
    "VIDEO_MAX_PIXELS",
    getattr(qwen_vp, "VIDEO_MAX_TOKEN_NUM", 768) * IMAGE_FACTOR * IMAGE_FACTOR,
)
VIDEO_TOTAL_PIXELS = _resolve_pixels(
    "VIDEO_TOTAL_PIXELS",
    int(getattr(qwen_vp, "MODEL_SEQ_LEN", 8192) * IMAGE_FACTOR * IMAGE_FACTOR * 0.9),
)
qwen_vp.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', getattr(qwen_vp, "FPS_MAX_FRAMES", 768))) # decrease this for efficiency 
from qwen_vl_utils.vision_process import (
    FORCE_QWENVL_VIDEO_READER, FPS_MAX_FRAMES, FRAME_FACTOR, FPS,
    smart_nframes, smart_resize
)

logger = logging.get_logger(__name__)

logger.warning(f'{__name__}: {FORCE_QWENVL_VIDEO_READER=}, {FPS_MAX_FRAMES=}, {VIDEO_MIN_PIXELS=}, {VIDEO_TOTAL_PIXELS=}')

def _read_video_decord_plus(ele: dict, strict_fps: bool = False, drop_last: bool = True, return_pts: bool = False):
    """read video using decord.VideoReader. can handle more cases compared to _read_video_decord.

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
        sample_fps
        clip_pts if return_pts=True
    """
    video_path = ele["video"]
    remote_loader = ele.get('remote_loader')
    if os.path.exists(video_path):
        vr = decord.VideoReader(video_path, num_threads=2)
    elif remote_loader is not None:
        vr = decord.VideoReader(remote_loader(video_path), num_threads=2)
    else:
        raise ValueError(f'video_path {video_path} not found')
    video_start = ele.get('video_start', None)
    video_end = ele.get('video_end', None)
    video_fps = vr.get_avg_fps()
    clip_idxs, clip_pts = None, None
    if video_start is not None or video_end is not None:
        vr.get_frame_timestamp(0)
        video_pts = vr._frame_pts[:,1]
        first_ts = float(video_pts[0])
        last_ts = float(video_pts[-1])
        orig_start, orig_end = video_start, video_end
        adjusted = False
        if video_start is None:
            video_start = first_ts
        else:
            if video_start < first_ts:
                video_start = first_ts
            elif video_start > last_ts:
                logger.warning(f"{video_path}: requested video_start {video_start:.3f} out of range [{first_ts:.3f}, {last_ts:.3f}], clamping.")
                video_start = last_ts
                adjusted = True
        if video_end is None:
            video_end = last_ts
        else:
            if video_end < first_ts:
                video_end = first_ts
            elif video_end > last_ts:
                logger.warning(f"{video_path}: requested video_end {video_end:.3f} out of range [{first_ts:.3f}, {last_ts:.3f}], clamping.")
                video_end = last_ts
                adjusted = True
        if video_end < video_start:
            logger.warning(f"{video_path}: video_end {video_end:.3f} < video_start {video_start:.3f}, snapping to start timestamp.")
            video_end = video_start
            adjusted = True
        if orig_start is not None or orig_end is not None:
            if adjusted:
                logger.warning(f"{video_path}: adjusted clip window from [{orig_start}, {orig_end}] to [{video_start}, {video_end}].")
        clip_mask = (video_start <= video_pts) & (video_pts <= video_end)
        clip_idxs = clip_mask.nonzero()[0]
        clip_pts = video_pts[clip_idxs]
        if clip_idxs.size == 0:
            logger.warning(f"{video_path}: no frames found in adjusted clip window, forcing closest available frame.")
            nearest_idx = int(np.clip(np.searchsorted(video_pts, video_start), 0, len(video_pts) - 1))
            clip_idxs = np.array([nearest_idx], dtype=int)
            clip_pts = np.array([video_pts[nearest_idx]], dtype=video_pts.dtype)
        total_frames = len(clip_idxs)
    else:
        total_frames = len(vr)

    if total_frames == 0:
        raise ValueError(f'video_path {video_path} contains no frames')

    pad_count = 0
    sampled_frames = 0
    if not strict_fps:
        if total_frames >= FRAME_FACTOR:
            try:
                nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
            except ValueError as exc:
                logger.warning(f'smart_nframes failed for {video_path} ({exc}), fallback to total_frames={total_frames}')
                nframes = total_frames
        else:
            nframes = total_frames
        nframes = max(int(nframes), 1)
        nframes_idxs = np.linspace(0, max(total_frames - 1, 0), nframes).round().astype(int)
        if clip_idxs is None:
            clip_idxs = nframes_idxs
        else:
            clip_idxs = clip_idxs[nframes_idxs]
        if clip_pts is not None:
            clip_pts = clip_pts[nframes_idxs]
        sampled_frames = len(clip_idxs)
        pad_count = (FRAME_FACTOR - sampled_frames % FRAME_FACTOR) % FRAME_FACTOR
        if pad_count and sampled_frames:
            pad_value = clip_idxs[-1]
            clip_idxs = np.concatenate([clip_idxs, np.full(pad_count, pad_value, dtype=int)])
            if clip_pts is not None:
                clip_pts = np.concatenate([clip_pts, np.full(pad_count, clip_pts[-1], dtype=clip_pts.dtype)])
    else:
        if clip_pts is None: # no video_start/video_end
            vr.get_frame_timestamp(0)
            clip_pts = vr._frame_pts[:,1]
            clip_idxs = np.arange(len(clip_pts))
        expected_timestamps = np.arange(clip_pts[0], clip_pts[-1] + 1e-6, 1 / FPS)
        if len(expected_timestamps) > FPS_MAX_FRAMES:
            if drop_last:
                expected_timestamps = expected_timestamps[:FPS_MAX_FRAMES]
            else:
                expected_timestamps = expected_timestamps[np.linspace(0, len(expected_timestamps) - 1, FPS_MAX_FRAMES).round().astype(int)]
        expected_idxs_for_clip_pts = (expected_timestamps[:, None] <= clip_pts).argmax(axis=1)
        clip_pts = clip_pts[expected_idxs_for_clip_pts].tolist()
        clip_idxs = clip_idxs[expected_idxs_for_clip_pts].tolist()
        sampled_frames = len(clip_idxs)
        while len(clip_idxs) % FRAME_FACTOR != 0:
            clip_idxs.append(clip_idxs[-1])
            clip_pts.append(clip_pts[-1])
        pad_count = len(clip_idxs) - sampled_frames

    if isinstance(clip_idxs, np.ndarray):
        clip_idx_list = clip_idxs.tolist()
    else:
        clip_idx_list = list(clip_idxs)
    clip = torch.from_numpy(vr.get_batch(clip_idx_list).asnumpy()).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = sampled_frames / max(total_frames, 1e-6) * video_fps
    video_metadata = dict(
        fps=video_fps,
        frames_indices=clip_idx_list,
        total_num_frames=total_frames,
        video_backend="decord+",
        padded_frames=pad_count,
    )
    if return_pts:
        return clip, sample_fps, clip_pts
    return clip, video_metadata, sample_fps

from qwen_vl_utils.vision_process import VIDEO_READER_BACKENDS
_video_reader_backend = VIDEO_READER_BACKENDS['decord+'] = _read_video_decord_plus

def _spatial_resize_video(video: torch.Tensor, nframes: int = None):
    if not nframes:
        nframes, _, height, width = video.shape
    else:
        height, width = video.shape[2:]
    max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=max_pixels,
    )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    ).float() # need float?
    return video

def get_smart_resized_video_reader(video_path: str, max_pixels: int = None):
    video_reader = decord.VideoReader(video_path)
    nframes = min(len(video_reader), FPS_MAX_FRAMES)
    height, width, _ = video_reader.next().shape

    if max_pixels is None:
        max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=max_pixels,
    )
    video_reader = decord.VideoReader(video_path, num_threads=2)
    return video_reader, resized_height, resized_width

def get_smart_resized_clip(
    video_reader: decord.VideoReader, 
    resized_height: int,
    resized_width: int,
    timestamps: torch.Tensor, 
    video_pts: np.ndarray, 
    video_pts_index_from: int = 0, 
):
    while len(timestamps) % FRAME_FACTOR != 0:
        timestamps = torch.cat([timestamps, timestamps[-1:] + 1 / FPS])
    clip_idxs = []
    for timestamp in timestamps:
        while video_pts_index_from < len(video_pts) and video_pts[video_pts_index_from] < timestamp:
            video_pts_index_from += 1
        if video_pts_index_from >= len(video_pts):
            break
        clip_idxs.append(video_pts_index_from)
    while len(clip_idxs) % FRAME_FACTOR != 0:
        clip_idxs = clip_idxs[:-1]
        timestamps = timestamps[:-1]
    clip = torch.from_numpy(video_reader.get_batch(clip_idxs).asnumpy()).permute(0, 3, 1 ,2) # thwc or cthw -> tchw
    # NOTE: windows OS may put channel first
    if (clip.shape[0] == 3) and (clip.shape[1] == len(clip_idxs)):
        clip = clip.transpose(0, 1)
    clip = transforms.functional.resize(
        clip,
        [resized_height, resized_width],
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    return clip, timestamps, clip_idxs
