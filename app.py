from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import base64
import os
from werkzeug.utils import secure_filename
import pdf2image
from datetime import datetime
import tempfile
import zipfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf_to_images(pdf_path):
    """Convert PDF to list of PIL Images"""
    try:
        images = pdf2image.convert_from_path(pdf_path, dpi=300)
        return images
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return []

def find_document_contour(image):
    """Find the largest rectangular contour in the image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edged = cv2.Canny(blurred, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Find the largest rectangular contour
    for contour in contours:
        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If we have a quadrilateral
        if len(approx) == 4:
            return approx.reshape(4, 2)
    
    return None

def order_points(pts):
    """Order points in top-left, top-right, bottom-right, bottom-left order"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    
    return rect

def four_point_transform(image, pts):
    """Apply perspective transform to get bird's eye view"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width and height of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def optimize_for_recognition(image):
    """Optimize image for OCR recognition"""
    # Convert to PIL Image for easier manipulation
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Convert to grayscale for ink saving
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.5)
    
    # Create dark background version
    # Convert back to RGB for background processing
    rgb_image = pil_image.convert('RGB')
    
    # Create a mask for the document (white/light areas)
    gray_array = np.array(pil_image)
    mask = gray_array > 180  # Threshold for light areas
    
    # Create dark background
    rgb_array = np.array(rgb_image)
    rgb_array[~mask] = [40, 40, 40]  # Dark gray background
    
    optimized_image = Image.fromarray(rgb_array)
    
    return optimized_image

def process_image(image_data, filename):
    """Process uploaded image for passport recognition"""
    try:
        # Convert to OpenCV format
        if filename.lower().endswith('.pdf'):
            # Handle PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
            
            # Convert PDF to images
            images = pdf_to_images(temp_path)
            os.unlink(temp_path)
            
            if not images:
                return None, "Failed to convert PDF"
            
            # Process first page (assuming passport is on first page)
            pil_image = images[0]
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            # Handle regular images
            pil_image = Image.open(io.BytesIO(image_data))
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        original_image = cv_image.copy()
        
        # Find document contour
        contour = find_document_contour(cv_image)
        
        if contour is not None:
            # Apply perspective transform
            transformed = four_point_transform(cv_image, contour)
        else:
            # If no contour found, use original image
            transformed = cv_image
        
        # Optimize for recognition
        optimized = optimize_for_recognition(transformed)
        
        # Convert results to base64 for web display
        def image_to_base64(img):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'original': image_to_base64(original_image),
            'processed': image_to_base64(optimized),
            'contour_found': contour is not None
        }, None
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_data = file.read()
        
        # Process the image
        result, error = process_image(file_data, filename)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'success': True,
            'original_image': result['original'],
            'processed_image': result['processed'],
            'contour_found': result['contour_found'],
            'filename': filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<image_type>')
def download_image(image_type):
    # This would be implemented to serve the processed images
    # For now, return a placeholder response
    return jsonify({'message': f'Download {image_type} image'})

if __name__ == '__main__':
    app.run(debug=True)