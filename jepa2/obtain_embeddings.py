import torch
import pickle
import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
from tqdm import tqdm

import sys

#hf_repo = "facebook/vjepa2-vitg-fpc64-384"
hf_repo = "facebook/vjepa2-vitl-fpc64-256"
video_path = sys.argv[1]
output_path = sys.argv[2]
block_size = 64

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModel.from_pretrained(hf_repo, torch_dtype=torch.float16).to(device)
processor = AutoVideoProcessor.from_pretrained(hf_repo)
model.eval()

vr = VideoDecoder(video_path)
total_frames = len(vr)

total_frames = int(total_frames * 0.1)

num_blocks = -(-total_frames // block_size)
print(f"Total frames: {total_frames}, blocks: {num_blocks}")

all_embeddings = []

for start in tqdm(range(0, total_frames, block_size), total=num_blocks, unit="block"):
    end = min(start + block_size, total_frames)
    frame_idx = np.arange(start, end)

    if len(frame_idx) < block_size:
        frame_idx = np.pad(frame_idx, (0, block_size - len(frame_idx)), mode="edge")

    frames = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        pixel_values = inputs["pixel_values_videos"]
        encoder_outputs = model.encoder(pixel_values_videos=pixel_values)

    embedding = encoder_outputs.last_hidden_state.mean(dim=1).float().cpu()
    all_embeddings.append(embedding)

all_embeddings = torch.cat(all_embeddings, dim=0)  # (num_blocks, hidden_dim)
print(f"\nFinal embeddings: {all_embeddings.shape}")

with open(output_path, "wb") as f:
    pickle.dump({"embeddings": all_embeddings.numpy(), "video_path": video_path, "block_size": block_size}, f)

print(f"Saved to {output_path}")
