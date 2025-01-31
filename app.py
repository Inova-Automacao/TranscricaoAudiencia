from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from pytube import YouTube
import whisper
import tempfile
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('API_KEY')

app = Flask(__name__)

# Configuração do banco de dados PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:eZaheKwKIeHknKAqDJqCfmQiCWKCLNAF@monorail.proxy.rlwy.net:55261/railway'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Definição do modelo
class Transcription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)

# Criação das tabelas
with app.app_context():
    db.create_all()

def verify_api_key():
    api_key = request.headers.get('X-API-KEY')
    if api_key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 403

# Rota para upload e transcrição
@app.route('/upload', methods=['POST'])
def upload():
    auth_error = verify_api_key()

    if auth_error:
        return auth_error
    # Verifica se a parte 'file' está na requisição
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # Se nenhum arquivo for selecionado
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    try:
        # Salva o arquivo em um diretório temporário
        temp_audio_path = save_temporary_file(file)

        # Carrega o modelo Whisper
        model = whisper.load_model("base")

        # Realiza a transcrição
        result = model.transcribe(temp_audio_path, language='pt')
        transcription_text = result['text']

        # Salva a transcrição no banco de dados
        transcription = Transcription(text=transcription_text)
        db.session.add(transcription)
        db.session.commit()

        return jsonify({"message": "Transcrição realizada com sucesso", "transcription": transcription_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def save_temporary_file(file):
    # Cria um arquivo temporário para salvar o conteúdo do arquivo enviado
    temp_audio = tempfile.NamedTemporaryFile(delete=False)

    # Salva os dados do arquivo na versão temporária
    file.save(temp_audio)
    temp_audio.close()

    # Retorna o caminho do arquivo temporário
    return temp_audio.name

# Rota GET para obter uma transcrição pelo ID
@app.route('/transcricao/<int:transcription_id>', methods=['GET'])
def get_transcription(transcription_id):

    auth_error = verify_api_key()

    if auth_error:
        return auth_error
    transcription = Transcription.query.get(transcription_id)

    if not transcription:
        return jsonify({"error": "Transcrição não encontrada"}), 404

    return jsonify({
        "id": transcription.id,
        "text": transcription.text
    }), 200

# Rota GET para obter todos os IDs de transcrições
@app.route('/transcricoes', methods=['GET'])
def get_all_transcription_ids():     
    transcription_ids = [transcription.id for transcription in Transcription.query.all()]
    return jsonify({"transcription_ids": transcription_ids}), 200

def download_audio(youtube_url):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    temp_audio = tempfile.NamedTemporaryFile(delete=False)
    stream.stream_to_buffer(temp_audio)
    temp_audio.close()
    return temp_audio.name

# Rota POST para enviar link do youtube
@app.route('/uploadlink', methods=['POST'])
def upload_link():

    auth_error = verify_api_key()

    if auth_error:
        return auth_error
    url_yt = request.form.get('url_yt')

    try:
        # Salva o arquivo em um diretório temporário
        temp_audio_path = download_audio(url_yt)

        # Carrega o modelo Whisper
        model = whisper.load_model("base")

        # Realiza a transcrição
        result = model.transcribe(temp_audio_path, language='pt')
        transcription_text = result['text']

        # Salva a transcrição no banco de dados
        transcription = Transcription(text=transcription_text)
        db.session.add(transcription)
        db.session.commit()

        # Remove o arquivo de áudio temporário (opcional)
        # O arquivo será deletado automaticamente ao fechar o NamedTemporaryFile

        return jsonify({"message": "Transcrição realizada com sucesso", "transcription": transcription_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
