import os
import cv2
import torch
import numpy as np
import time
from flask import Flask, render_template, Response, request, send_file, url_for
from models.networks import define_G
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename
import threading
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Load CycleGAN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading CycleGAN model...")

netG = define_G(input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks',
                norm='instance', use_dropout=False, init_type='normal', init_gain=0.02)
checkpoint_path = 'checkpoints/photo2sketch/latest_net_G.pth'
netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
netG.to(device)
netG.eval()
print("Model loaded successfully.")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = netG(image)
    output = output.squeeze().cpu().detach()
    output = (output + 1) / 2
    output = output.numpy().transpose(1, 2, 0)
    return cv2.cvtColor((output * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# ========== Live Camera ==========

camera = cv2.VideoCapture(0)

def generate():
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.resize(frame, (256, 256))
        sketch = transform_frame(frame)
        combined = np.hstack((frame, sketch))
        _, buffer = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ========== Upload Route ==========

# Global dict to store progress and output filename per job
jobs = {}

@app.route('/upload_sketch', methods=['POST'])
def upload_sketch():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    file_ext = filename.split('.')[-1].lower()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # === IMAGE Upload ===
    if file_ext in ['jpg', 'jpeg', 'png']:
        img = cv2.imread(filepath)
        img = cv2.resize(img, (256, 256))
        sketched = transform_frame(img)
        sketch_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sketch.jpg')
        cv2.imwrite(sketch_path, sketched)
        return send_file(sketch_path, mimetype='image/jpeg')

    # === VIDEO Upload ===
    elif file_ext in ['mp4', 'avi', 'mov']:
        job_id = str(uuid.uuid4())
        jobs[job_id] = {'percent': 0, 'done': False, 'output_filename': None}
        
        def process_video():
            cap = cv2.VideoCapture(filepath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 10
            output_filename = f'sketched_{int(time.time())}_{job_id}.mp4'
            output_path = os.path.join(app.config['STATIC_FOLDER'], output_filename)
            frame_width, frame_height = 256, 256
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0
            frame_idx = 0
            last_sketch = None
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (256, 256))
                if frame_idx % 5 == 0:
                    last_sketch = transform_frame(frame)
                out.write(last_sketch)
                processed_frames += 1
                jobs[job_id]['percent'] = int((processed_frames / total_frames) * 100)
                frame_idx += 1
            cap.release()
            out.release()
            jobs[job_id]['percent'] = 100
            jobs[job_id]['done'] = True
            jobs[job_id]['output_filename'] = output_filename
        threading.Thread(target=process_video, daemon=True).start()
        return render_template("progress.html", job_id=job_id)
    else:
        return "Unsupported file type", 400

@app.route('/progress/<job_id>')
def get_progress(job_id):
    job = jobs.get(job_id)
    if not job:
        return {'percent': 0, 'done': False}
    return {'percent': job['percent'], 'done': job['done']}

@app.route('/result/<job_id>')
def get_result(job_id):
    job = jobs.get(job_id)
    if not job or not job['done']:
        return "Not ready", 404
    return render_template("video_display.html", video_filename=job['output_filename'])

# ========== MAIN ==========

if __name__ == '__main__':
    app.run(debug=True)