import os
import cv2
import torch
from options.test_options import TestOptions
from models import create_model
from util import util
from PIL import Image
import numpy as np

def process_video(video_path, output_video):
    # Extract frames
    os.makedirs('temp_frames', exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = f'temp_frames/frame_{i:04d}.jpg'
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        i += 1
    cap.release()

    # Load CycleGAN model
    opt = TestOptions().parse()
    opt.dataroot = 'temp_frames'
    opt.name = 'photo2sketch'
    opt.model = 'test'
    opt.no_dropout = True
    opt.checkpoints_dir = './checkpoints'
    opt.results_dir = './results'
    opt.gpu_ids = [-1]  # CPU mode
    model = create_model(opt)
    model.setup(opt)

    # Process frames
    out_frames = []
    for f in frames:
        img = util.load_image(f, opt)
        model.set_input({'A': img, 'A_paths': f})
        model.test()
        visuals = model.get_current_visuals()
        fake_img = util.tensor2im(visuals['fake'])
        sketch_path = f'results/frame_{os.path.basename(f)}'
        util.save_image(fake_img, sketch_path)
        out_frames.append(sketch_path)

    # Combine into video
    frame = cv2.imread(out_frames[0])
    height, width, _ = frame.shape
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    for img in out_frames:
        out.write(cv2.imread(img))
    out.release()

    print(f"Sketch video saved at {output_video}")