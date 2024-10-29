from typing import List, Tuple
from google.cloud import bigquery
import pandas as pd
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel
from flask import Flask, request, jsonify

# Initialize Vertex AI
vertexai.init(project="dataton-2024-team-11-cofares", location="europe-west1")

# Initialize models
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
generative_multimodal_model = GenerativeModel("gemini-1.5-flash-002")

class productos():
    def __init__(self):
        self.id = int()

def gen_conv_KNN(conv: List[Tuple[str, str]], veto: List[productos]) -> str:
    prompt = "Necesidades:\ncliente: "
    usuario = True
    for x in conv:
        if usuario:
            prompt += "Cliente: " + x + "\n"
            usuario = False
        else:
            prompt += "Farmacéutico " + x + "\n"
    if len(veto) > 0:
        prompt += "\nProductos vetados:\n"
        for producto in veto:
            prompt += f"- Producto ID: {producto.nombre}\n"
    return prompt

def query_template(data):
    return "SELECT base.id, distance FROM VECTOR_SEARCH(TABLE DataHub.Muestras_Cofares_Limpio_Emb,'ml_generate_embedding_result', ((select " + data + " as ml_generate_embedding_result)),top_k => 5,distance_type => 'COSINE');"

def buscar_cercanos(data, veto: List[str]) -> pd.DataFrame:
    client = bigquery.Client()
    embeddings = embedding_model.get_embeddings([data])
    # Ejecutar la consulta
    emb = query_template(str(embeddings[0].values))

    query_job = client.query(emb)
    results = query_job.result()

    # Convertir los resultados a un DataFrame
    df = results.to_dataframe()

    # Calculate similarity (1 - distance)
    df['similarity'] = 1 - df['distance']
    
    # Sort by similarity
    for n,x in enumerate(df["id"]):
        if x in veto:
            df.drop(n)
    df = df.sort_values('similarity', ascending=False)
    return df

def get_info(df):
    contexto = ""
    client = bigquery.Client()
    consulta = ""
    for x in range(len(df)):
        consulta += "'" + df["id"][x] + "',"
    consulta = consulta[:-1]

    query = "SELECT nombre_material,concat('nombre ', ifnull(nombre_material,''), '.Nombre_largo ', ifnull(nombre_material_largo,''),'.Marca propia ', es_marca_propia,'.Nombre proveedor ',ifnull(nombre_proveedor,''),'.Nombres matrículas: ', ifnull(nombre_matricula_nivel0,''), ', ', ifnull(nombre_matricula_nivel1,''), ', ', ifnull(nombre_matricula_nivel2,''), ', ', ifnull(nombre_matricula_nivel3,''), ', ', ifnull(nombre_matricula_nivel4,''), ', ', ifnull(nombre_matricula_nivel5,''), '. Información ', ifnull(txt_mas_informacion_del_producto,''), '. Instrucciones ', ifnull(txt_instrucciones_de_uso,''), '. Composición ', ifnull(txt_composicion,'')) AS content FROM `DataHub.Muestras_Cofares_Limpio` where id in (" + consulta + ")"
    
    query_job = client.query(query)
    results = query_job.result().to_dataframe()
    
    productos = "\n".join(results["nombre_material"])
    contexto += str(results["content"])
    client.close()
    return contexto, productos

def respuesta_promt(consulta,contexto):
    prompt = "Eres un ayudante de farmacia, tienes que conseguir hacer la pregunta correcta para saber que producto es mejor, teniendo en cuenta los productos relevantes y el contexto anterior, contesta únicamente con la pregunta que harías al usuario. La pregunta nunca puede ser sobre si le interesa tal o cual producto"
    prompt += "\ncontexto " + contexto
    prompt += "\nconsulta " + str(consulta)
    return generative_multimodal_model.generate_content(prompt)

def buscar_producto(consulta: List[str], veto: List[int]) -> Tuple[List[float], str]:
    # Generar el prompt para el sistema RAG
    data = gen_conv_KNN(consulta, veto)

    cercanos = buscar_cercanos(data, veto)
    contexto, productos = get_info(cercanos)
    res = "\nProductos encontrados:" + productos
    
    respuesta = respuesta_promt(consulta, contexto)
    res += respuesta.text
    return res

# Inicializa Flask
app = Flask(__name__)

# Modifica la función main para que sea un endpoint
@app.route('/consulta', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400
            
        user_input = data.get('message')
        if not user_input:
            return jsonify({'error': 'El mensaje está vacío'}), 400

        respuesta = buscar_producto(user_input, [])
        return jsonify({'response': respuesta})
        
    except Exception as e:
        print(f"Error: {str(e)}")  # Para logging
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def hello():
    return "¡Bienvenido a FarmaCIA!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)