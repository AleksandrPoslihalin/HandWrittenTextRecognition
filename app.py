from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from main import text_segment_and_recogn  # Замените main на имя вашего скрипта

UPLOAD_FOLDER = 'uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_recognize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Извлекаем дополнительные параметры
        pen_color = request.form.get('pen_color')
        paper_type = request.form.get('paper_type')
        recognized_text = text_segment_and_recogn(file_path, pen_color, paper_type)
        return jsonify({'recognized_text': recognized_text})

if __name__ == '__main__':
    app.run(debug=True)
