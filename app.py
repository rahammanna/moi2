from concurrent.futures.thread import _worker
import os, time, subprocess, sys, json, wave, re
from flask import Flask, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from vosk import Model, KaldiRecognizer,  SpkModel

WORKDIR = '/home/rahammanna/moi2/'
UPLOAD_FOLDER = WORKDIR + 'content/'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'aac'}
WAVFILE = UPLOAD_FOLDER + 'dia.wav'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            recode_file(filename)
            response = processing()
            return render_template('index.html', response=response)
    return render_template('index.html')

def cosine_dist(x, y):
        nx = np.array(x)
        ny = np.array(y)
        return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)

def recode_file(filename):
    try:
        os.remove(WAVFILE)
        # ffmpeg -y -i $inputfile -ar 48000 -ac 1 -f wav $wavfile
        command = ['ffmpeg', '-y', '-i', UPLOAD_FOLDER + filename, '-ar', '48000', '-ac', '1', '-f', 'wav', WAVFILE]
        process = subprocess.Popen(args=command, stdout=subprocess.PIPE)
        #filename = process.stdout.read()
        process.wait()
        os.remove(UPLOAD_FOLDER + filename)
    except Exception:
        raise

def processing():
    model_path = WORKDIR + "vosk-model-small-ru-0.22"
    spk_model_path = WORKDIR + "vosk-model-spk-0.4"
    model = Model(model_path)
    spk_model = SpkModel(spk_model_path)
    wf = wave.open(WAVFILE, "rb")
    spk_sig = [-1.110417,0.09703002,1.35658,0.7798632,-0.305457,-0.339204,0.6186931,-0.4521213,0.3982236,-0.004530723,0.7651616,0.6500852,-0.6664245,0.1361499,0.1358056,-0.2887807,-0.1280468,-0.8208137,-1.620276,-0.4628615,0.7870904,-0.105754,0.9739769,-0.3258137,-0.7322628,-0.6212429,-0.5531687,-0.7796484,0.7035915,1.056094,-0.4941756,-0.6521456,-0.2238328,-0.003737517,0.2165709,1.200186,-0.7737719,0.492015,1.16058,0.6135428,-0.7183084,0.3153541,0.3458071,-1.418189,-0.9624157,0.4168292,-1.627305,0.2742135,-0.6166027,0.1962581,-0.6406527,0.4372789,-0.4296024,0.4898657,-0.9531326,-0.2945702,0.7879696,-1.517101,-0.9344181,-0.5049928,-0.005040941,-0.4637912,0.8223695,-1.079849,0.8871287,-0.9732434,-0.5548235,1.879138,-1.452064,-0.1975368,1.55047,0.5941782,-0.52897,1.368219,0.6782904,1.202505,-0.9256122,-0.9718158,-0.9570228,-0.5563112,-1.19049,-1.167985,2.606804,-2.261825,0.01340385,0.2526799,-1.125458,-1.575991,-0.363153,0.3270262,1.485984,-1.769565,1.541829,0.7293826,0.1743717,-0.4759418,1.523451,-2.487134,-1.824067,-0.626367,0.7448186,-1.425648,0.3524166,-0.9903384,3.339342,0.4563958,-0.2876643,1.521635,0.9508078,-0.1398541,0.3867955,-0.7550205,0.6568405,0.09419366,-1.583935,1.306094,-0.3501927,0.1794427,-0.3768163,0.9683866,-0.2442541,-1.696921,-1.8056,-0.6803037,-1.842043,0.3069353,0.9070363,-0.486526]
    #spk_sig =[-0.435445, 0.877224, 1.072917, 0.127324, -0.605085, 0.930205, 0.44148, -1.20399, 0.069384, 0.538427, 1.226569, 0.852291, -0.806415, -1.157439, 0.313101, 1.332273, -1.628154, 0.402829, 0.472996, -1.479501, -0.065581, 1.127467, 0.897095, -1.544573, -0.96861, 0.888643, -2.189499, -0.155159, 1.974215, 0.277226, 0.058169, -1.234166, -1.627201, -0.429505, -1.101772, 0.789727, 0.45571, -0.547229, 0.424477, -0.919078, -0.396511, 1.35064, -0.02892, -0.442538, -1.60219, 0.615162, 0.052128, -0.432882, 1.94985, -0.704909, 0.804217, 0.472941, 0.333696, 0.47405, -0.214551, -1.895343, 1.511685, -1.284075, 0.623826, 0.034828, -0.065535, 1.604209, -0.923321, 0.502624, -0.288166, 0.536349, -0.631745, 0.970297, 0.403614, 0.131859, 0.978622, -0.5083, -0.104544, 1.629872, 1.730207, 1.010488, -0.866015, -0.711263, 2.359106, 1.151348, -0.426434, -0.80968, -1.302515, -0.444948, 0.074877, 1.352473, -1.007743, 0.318039, -1.532761, 0.145248, 3.59333, -0.467264, -0.667231, -0.890853, -0.197016, 1.546726, 0.890309, -0.7503, 0.773801, 0.84949, 0.391266, -0.79776, 0.895459, -0.816466, 0.110284, -1.030472, -0.144815, 1.087008, -1.448755, 0.776005, -0.270475, 1.223657, 1.09254, -1.237237, 0.065166, 1.487602, -1.409871, -0.539695, -0.758403, 0.31941, -0.701649, -0.210352, 0.613223, 0.575418, -0.299141, 1.247415, 0.375623, -1.001396]
    rec = KaldiRecognizer(model, wf.getframerate() * wf.getnchannels(), spk_model)
    #rec = KaldiRecognizer(model, wf.getframerate() * wf.getnchannels())
    rec.SetSpkModel(spk_model)
    text = ''
    text_array = []
    xvector = []
    speaker = []
    frames = []
    wf.rewind()  
    for i in range(3000):
        data = wf.readframes(4000)
        datalen=len(data);
        if datalen == 0:
            res = json.loads(rec.FinalResult())
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text_array.append(res['text'])
            text += res['text']
            if 'spk' in res:
                xvector.append(res['spk'])
                speaker.append(cosine_dist(spk_sig, res['spk']))
                frames.append(res['spk_frames'])
        if datalen == 0:
            break
    response = {"text": text, "text_array": text_array, "xvector": xvector, "speaker": speaker, "frames": frames}
    return response