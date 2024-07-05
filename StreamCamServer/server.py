import cv2
import base64
import numpy as np
import threading
import queue
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval, BlipForQuestionAnswering
from PIL import Image
import requests
import matplotlib.pyplot as plt

app = Flask(__name__)
socketio = SocketIO(app)
frame_queue = queue.Queue()
connected = False

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    global connected
    connected = True
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    global connected
    connected = False
    cv2.destroyAllWindows()
    print("Client disconnected")

@socketio.on('video_frame')
def handle_video_frame(data):
    # Decode base64 image
    img_data = base64.b64decode(data.split(',')[1])
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if connected:
        frame_queue.put(frame)

def display_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            text_corpus = []
            text_inputs = processor(text=text_corpus, return_tensors="pt", padding=True, truncation=True)
            image_inputs = processor(images=frame, return_tensors="pt")

            with torch.no_grad():
                outputs = retrieval_model(**image_inputs, **text_inputs)
                itm_scores = outputs.itm_score  # ITM scores for each pair

            # Debug: Print itm_scores and corresponding text
            print("ITM Scores:")
            for score, text in zip(itm_scores, text_corpus):
                print(f"{text}: {score}")

            # Select the best match based on the highest score for the match class (index 1)
            best_idx = itm_scores[:, 1].argmax().item()
            best_match = text_corpus[best_idx]
    cv2.destroyAllWindows()

if __name__ == '__main__':
    retrieval_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    display_thread = threading.Thread(target=display_frames)
    display_thread.start()
    socketio.run(app, host='0.0.0.0', port=5000)
