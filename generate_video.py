import cv2
import os

image_folder = "outputs/photo2sketch/test_latest/images"
output_video = "outputs/sketch_video.mp4"

os.makedirs("outputs", exist_ok=True)

images = sorted([img for img in os.listdir(image_folder) if img.endswith("_fake.png")])
if not images:
    print("No frames found!")
else:
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for img in images:
        out.write(cv2.imread(os.path.join(image_folder, img)))
    out.release()

print(f"Sketch video saved at {output_video}")