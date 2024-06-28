from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import whisper
import tempfile

app = Flask(__name__)

# Configuração do banco de dados PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:hbZbvyApKVUTDWtpoNUMRxNQOrjjPRJO@monorail.proxy.rlwy.net:27291/railway'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Definição do modelo
class Transcription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)

# Criação das tabelas
with app.app_context():
    db.create_all()

# Rota para upload e transcrição
@app.route('/upload', methods=['POST'])
def upload():
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

if __name__ == '__main__':
    app.run(debug=True)
