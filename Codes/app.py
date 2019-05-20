from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import os
import cv2

UPLOAD_FOLDER = '/content/ano_pred_cvpr2018/uploads/01'
ALLOWED_EXTENSIONS = set(['jpg', 'png'])
frames_path = 'frames'
npy_path = '/content/npy'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # upload file
run_with_ngrok(app)  # Start ngrok when app is run

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    print (request.method)
    del_files_name = os.listdir(UPLOAD_FOLDER)
    for dels in del_files_name:
        os.remove(UPLOAD_FOLDER + '/' + dels)

    del_files_name = os.listdir(frames_path)
    for dels in del_files_name:
        os.remove(frames_path + '/' + dels)
        
    if request.method == 'GET':
        
        
        
        #return render_template('result.html')
        return render_template('index.html', value='hi')
        
        
    if request.method == 'POST':
        print(request.files)
        
        uploaded_files = request.files.getlist("file[]")
        
        for _file in uploaded_files:        
            filename = secure_filename(_file.filename)
            _file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        !python inference.py --dataset  ped2    \
                    --test_folder  ../uploads      \
                    --gpu  0    \
                    --snapshot_dir    checkpoints/pretrains/ped2    \
                    --evaluate compute_auc
        
        os.system('ffmpeg -f image2 -r 20 -i ./frames/%04d.jpg -vcodec libx264 -acodec aac -y static/video.mp4')
		
        return render_template('result.html')
    

if __name__ == '__main__':
    app.run()