from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)

# Pasta onde os arquivos de áudio serão armazenados
AUDIOS = 'uploads'
os.makedirs(AUDIOS, exist_ok=True) 
app.config['AUDIOS'] = AUDIOS

# Pasta onde os arquivos de texto transcritos serão armazenados
TEXTO = 'transcricao'
os.makedirs(TEXTO, exist_ok=True) 
app.config['TEXTO'] = TEXTO

# Extensões permitidas
EXTENSOES_PERMITIDAS = {'wav', 'mp3', 'm4a', 'flac'}

def verifica_arq(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSOES_PERMITIDAS

@app.route('/upload', methods=['POST'])
def upload_file():
    # Verifica se a parte 'file' está na requisição
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # Se nenhum arquivo for selecionado
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    if file and verifica_arq(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['AUDIOS'], filename)
        file.save(filepath)

        return jsonify({"message": "Arquivo salvo com sucesso", "filename": filename}), 200

    return jsonify({"error": "Tipo de arquivo nao suportado"}), 400

@app.route('/transcrever', methods=['POST'])
def traduzir_arquivo():
    # Recebe o nome do arquivo previamente carregado
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "Nome do arquivo não fornecido"}), 400

    # Caminho completo para o arquivo de áudio
    filepath = os.path.join(app.config['AUDIOS'], filename)

    # Carrega o modelo Whisper
    model = whisper.load_model("medium")

    try:
        # Carrega o áudio
        audio = whisper.load_audio(filepath)
        audio = whisper.pad_or_trim(audio)

        # Cria o log-Mel spectrogram e move para o mesmo dispositivo que o modelo
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Define as opções de decodificação com o idioma português
        decode_options = whisper.DecodingOptions(language='pt')
        
        # Realiza a transcrição
        result = model.decode(mel, decode_options)
        transcription_text = result.text

        # Caminho completo para salvar a transcrição
        transcription_filename = f"{os.path.splitext(filename)[0]}.txt"
        transcription_filepath = os.path.join(app.config['TEXTO'], transcription_filename)

        # Salve a transcrição em um arquivo de texto
        with open(transcription_filepath, 'w', encoding='utf-8') as f:
            f.write(transcription_text)

        # Retorne o caminho do arquivo de transcrição como resposta JSON
        return jsonify({"message": "Transcrição realizada com sucesso", "transcription_file": transcription_filepath}), 200


    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/teste', methods=['GET'])
def testar():
    return jsonify({"message": "Transcrição realizada com sucesso"}), 200

if __name__ == '__main__':
    app.run(debug=True)
