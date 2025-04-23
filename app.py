import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from flask import Flask, request, render_template, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLO model
try:
    model = YOLO("best.pt")  # Ensure this model is trained on the 9 classes
    logger.info("YOLO model loaded successfully")
    logger.info(f"Model class names: {model.names}")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise

# Define color map for the 9 classes using RGB values, matching the model's class names
COLOR_MAP = {
    "Blight": sv.Color(255, 0, 0),      # Red
    "Brown Spot": sv.Color(0, 0, 255),  # Blue
    "False Smut": sv.Color(0, 255, 0),  # Green
    "Healthy": sv.Color(255, 255, 0),   # Yellow
    "Leaf Smut": sv.Color(128, 0, 128), # Purple
    "Rice blast": sv.Color(255, 165, 0),# Orange
    "Stem Rot": sv.Color(255, 0, 255),  # Magenta
    "Tungro": sv.Color(0, 255, 255),    # Cyan
    "Background": sv.Color(255, 255, 255)# White
}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process an image for crop disease detection."""
    try:
        logger.info(f"Processing image: {image_path}")
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Failed to load image")
            return None, "Error: Could not load image", {}

        # Resize image for consistent processing
        target_width, target_height = 1280, 720
        image = cv2.resize(image, (target_width, target_height))
        logger.info("Image resized successfully")

        # Run YOLO detection
        results = model(image)[0]
        detections = sv.Detections.from_ultralytics(results)
        logger.info("YOLO detection completed")

        # Log detections and count diseases
        disease_counts = {}
        if len(detections) > 0:
            for class_id in detections.class_id:
                class_name = results.names[class_id]
                disease_counts[class_name] = disease_counts.get(class_name, 0) + 1
            logger.info(f"Detected diseases: {disease_counts}")
        else:
            logger.info("No diseases detected")

        # Assign colors to detections based on class names
        if len(detections) > 0:
            colors = []
            for class_id in detections.class_id:
                class_name = results.names[class_id]
                color = COLOR_MAP.get(class_name, sv.Color(128, 128, 128))  # Default to gray if class not in COLOR_MAP
                colors.append(color)
        else:
            colors = []

        # Annotate image with bounding boxes and labels
        annotated_image = image.copy()
        if len(detections) > 0:
            # Draw bounding boxes and labels for each detection with its corresponding color
            for detection_idx, (xyxy, color) in enumerate(zip(detections.xyxy, colors)):
                # Create a single-detection object for this iteration
                single_detection = sv.Detections(
                    xyxy=np.array([xyxy]),
                    class_id=np.array([detections.class_id[detection_idx]]),
                    confidence=np.array([detections.confidence[detection_idx]])
                )

                # Initialize annotators with the specific color for this detection
                box_annotator = sv.BoxAnnotator(color=color)
                label_annotator = sv.LabelAnnotator(
                    color=color,
                    text_color=sv.Color(255, 255, 255),  # White text for better contrast
                    text_position=sv.Position.TOP_LEFT
                )

                # Annotate bounding box
                annotated_image = box_annotator.annotate(
                    scene=annotated_image,
                    detections=single_detection
                )

                # Annotate label with class name and confidence
                labels = [f"{results.names[single_detection.class_id[0]]}: {single_detection.confidence[0]:.2f}"]
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=single_detection,
                    labels=labels
                )

        # Add disease count overlay on the image
        y_offset = 30
        for disease, count in disease_counts.items():
            text = f"{disease}: {count}"
            cv2.putText(
                annotated_image, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            y_offset += 30

        # Save the annotated image with a unique filename (using timestamp)
        timestamp = int(time.time())
        output_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{timestamp}.jpg')
        cv2.imwrite(output_filename, annotated_image)
        logger.info(f"Annotated image saved as: {output_filename}")

        # Return the relative path for URL generation
        output_path = f'uploads/output_{timestamp}.jpg'
        return output_path, None, disease_counts

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None, f"Error: {str(e)}", {}

@app.route('/favicon.ico')
def favicon():
    """Serve a favicon to eliminate 404 errors."""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and display results."""
    error = None
    output_image = None
    disease_summary = {}

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            error = 'No file part'
            logger.error(error)
        else:
            file = request.files['file']
            if file.filename == '':
                error = 'No selected file'
                logger.error(error)
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"File uploaded: {filepath}")

                # Process the image
                output_path, processing_error, disease_counts = process_image(filepath)
                if processing_error:
                    error = processing_error
                else:
                    output_image = url_for('static', filename=output_path)
                    disease_summary = disease_counts

    return render_template('index.html', error=error, output_image=output_image, disease_summary=disease_summary)

if __name__ == '__main__':
    # Disable Flask auto-reloading to prevent interruptions
    app.run(debug=True, use_reloader=False)