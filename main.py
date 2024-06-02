from flask import Flask, request, jsonify, send_file, redirect
from flask_cors import CORS
import os, threading, webbrowser
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)
    
model_path = f'{os.getcwd()}/model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

@app.route('/')
def index():
    return send_file(f'index.html')

@app.route('/converter')
def converter():
    return send_file(f'convert.html')
@app.route('/merger')
def merger():
    return send_file(f'merger.html')

@app.route('/about')
def about():
    return send_file(f'about.html')

@app.route('/ai_chat')
def ai_chat():
    return send_file(f'chat.html')

@app.errorhandler(404)
def page_not_found(e):
    return redirect("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']
    
    inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.2,
        top_p=0.2,
        top_k=50,
        no_repeat_ngram_size=2,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove user input from the response
    response = response.replace(user_input, '').strip()
    
    return jsonify({"response": response})

def convert_image(input_file_path, output_format):
    try:
        output_folder = os.path.dirname(input_file_path)
        file_name, file_extension = os.path.splitext(os.path.basename(input_file_path))
        output_file_path = os.path.join(output_folder, f"{file_name}_converted.{output_format}")
        im = Image.open(input_file_path)
        im.save(output_file_path, output_format.upper())
        return output_file_path
    except Exception as e:
        print(e)

@app.route('/merge_files', methods=['POST'])
def merge_files():
    video_path = request.form['videoPath']
    audio_path = request.form['audioPath']
    output_path = request.form['outputPath']
    output_file_name = request.form['outputFileName']

    if not (video_path and audio_path and output_path and output_file_name):
        return jsonify({'message': 'Please provide all necessary information.'}), 400
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ffmpeg_cmd = 'ffmpeg'
    merge_command = f'{ffmpeg_cmd} -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac {os.path.join(output_path, output_file_name)}'
    os.system(merge_command)
    
    output_message = f"Files merged successfully. Output saved as {os.path.join(output_path, output_file_name)}"
    return jsonify({'message': output_message})


@app.route('/convert', methods=['POST'])
def convert():
    data = request.json
    input_path = data['input_path']
    output_format = data['output_format']
    
    if not os.path.isfile(input_path):
        return jsonify({"error": "Invalid file path"}), 400
    
    output_file_path = convert_image(input_path, output_format)
    if output_file_path:
        return jsonify({"message": "File converted successfully", "output_file_path": output_file_path})
    else:
        return jsonify({"error": "Conversion failed"}), 500

def start_server():
    app.run(port=6969)

def open_browser():
    while True:
        try:
            webbrowser.open('http://127.0.0.1:6969/')
            break
        except:
            continue

if __name__ == '__main__':
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    open_browser()