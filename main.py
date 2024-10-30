from typing import List, Tuple
from google.cloud import bigquery
import pandas as pd
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
from google.cloud import storage
import base64
import numpy as np
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
    return "with VectorSearchResults AS (SELECT base.id, distance FROM VECTOR_SEARCH(TABLE DataHub.embeddings_cervantes,'embed', ((select " + data + " as embed)),top_k => 5,distance_type => 'COSINE')) SELECT search.id, search.distance, DC.nombre_material, DC.txt_mas_informacion_del_producto, DC.ids_imagenes, concat('nombre ', ifnull(DC.nombre_material,''),'.Nombre_largo ', ifnull(DC.nombre_material_largo,''),'.Marca propia ', DC.es_marca_propia,'.Nombre proveedor ', ifnull(DC.nombre_proveedor,''),'.Nombres matrículas: ',ifnull(DC.nombre_matricula_nivel0,''), ', ',ifnull(DC.nombre_matricula_nivel1,''), ', ',ifnull(DC.nombre_matricula_nivel2,''), ', ',ifnull(DC.nombre_matricula_nivel3,''), ', ',ifnull(DC.nombre_matricula_nivel4,''), ', ',ifnull(DC.nombre_matricula_nivel5,''),'. Información ', ifnull(DC.txt_mas_informacion_del_producto,''),'. Instrucciones ', ifnull(DC.txt_instrucciones_de_uso,''),'. Composición ', ifnull(DC.txt_composicion,'')) AS content FROM DataHub.Datos_Cofares AS DC JOIN  VectorSearchResults AS search ON DC.codigo_material = search.id"

def get_image_from_storage(image_id: str) -> str:
    try:
        # Limpiar el ID de la extensión .jpg si existe
        image_id = image_id.replace('.jpg', '')
        
        storage_client = storage.Client()
        bucket = storage_client.bucket('dataton-2024-team-11-cofares-datastore')
        
        # Primero intentamos con .jpg
        blob = bucket.blob(f'reto_cofares/{image_id}.jpg')
        if not blob.exists():
            # Si no existe como jpg, buscamos la primera parte (quitando _P_ o _U_)
            base_id = image_id
            if '_P_' in image_id:
                base_id = image_id.split('_P_')[0]
            elif '_U_' in image_id:
                base_id = image_id.split('_U_')[0]
            
            # Intentamos con _P_1.webp
            blob = bucket.blob(f'reto_cofares/{base_id}_P_1.webp')
            if not blob.exists():
                # Intentamos con _U_1.webp
                blob = bucket.blob(f'reto_cofares/{base_id}_U_1.webp')
                if not blob.exists():
                    # Si tampoco existe como webp, intentamos el original como webp
                    blob = bucket.blob(f'reto_cofares/{image_id}.webp')
                    if not blob.exists():
                        print(f"No se encontró ninguna imagen para el {image_id}")
                        return None
        
        # Descarga la imagen en memoria
        image_data = blob.download_as_bytes()
        
        # Convierte a base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determina el tipo MIME basado en la extensión
        mime_type = 'image/jpeg' if blob.name.endswith('.jpg') else 'image/webp'
        return f"data:{mime_type};base64,{base64_image}"
    
    except Exception as e:
        print(f"Error al cargar la imagen {image_id}: {str(e)}")
        return None

def buscar_cercanos(data, veto: List[str]) -> Tuple[List[dict], str]:
    
    client = bigquery.Client()
    embeddings = embedding_model.get_embeddings([data])

    emb = query_template(str(embeddings[0].values))
    query_job = client.query(emb)
    df = query_job.result().to_dataframe()

    # Obtener el contexto de todos los productos
    contexto = str(df["content"].str.cat(sep=' '))

    # Calculate similarity (1 - distance)
    df['similarity'] = 1 - df['distance']
    
    # Filtrar productos vetados y ordenar
    df = df[~df['id'].isin(veto)]
    df = df.sort_values('similarity', ascending=False)

    productos_encontrados = []
    for _, row in df.iterrows():
        # Procesar las IDs de imágenes y obtener las imágenes
        imagenes = []
        try:
            # Convertir la serie de ids_imagenes a una lista plana
            if isinstance(row["ids_imagenes"], (list, np.ndarray)):
                image_ids = [row["ids_imagenes"][0]]
            else:
                # Si es un string, procesarlo
                image_ids = str(row["ids_imagenes"]).replace("[", "").replace("]", "").replace("'", "").replace(".jpg", "").split(",")
            
            # Limpiar y procesar cada ID
            image_ids = [id.strip() for id in image_ids if id.strip()]
            
            # Obtener las imágenes
            imagenes = [get_image_from_storage(img_id) for img_id in image_ids]
            # Filtra las imágenes que no se pudieron cargar
            imagenes = [img for img in imagenes if img is not None]
        except Exception as e:
            print(f"Error procesando imágenes: {e}")
            imagenes = []
        
        producto_info = {
            "id": row["id"],
            "Nombre": row["nombre_material"],
            "Descripcion": row["txt_mas_informacion_del_producto"],
            "Similitud": row["similarity"],
            "Imagenes": imagenes if imagenes else ["no image"]
        }
        productos_encontrados.append(producto_info)

    client.close()
    return productos_encontrados, contexto

def get_info(df):
    contexto = ""
    client = bigquery.Client()
    consulta = ""
    for x in range(len(df)):
        consulta += "'" + df["id"][x] + "',"
    consulta = consulta[:-1]

    query = "SELECT nombre_material, concat('nombre ', ifnull(nombre_material,''), '.Nombre_largo ', ifnull(nombre_material_largo,''),'.Marca propia ', es_marca_propia,'.Nombre proveedor ',ifnull(nombre_proveedor,''),'.Nombres matrículas: ', ifnull(nombre_matricula_nivel0,''), ', ', ifnull(nombre_matricula_nivel1,''), ', ', ifnull(nombre_matricula_nivel2,''), ', ', ifnull(nombre_matricula_nivel3,''), ', ', ifnull(nombre_matricula_nivel4,''), ', ', ifnull(nombre_matricula_nivel5,''), '. Información ', ifnull(txt_mas_informacion_del_producto,''), '. Instrucciones ', ifnull(txt_instrucciones_de_uso,''), '. Composición ', ifnull(txt_composicion,'')) AS content FROM `DataHub.Muestras_Cofares_Limpio` where id in (" + consulta + ")"
    
    query_job = client.query(query)
    results = query_job.result().to_dataframe()
    
    contexto += str(results["content"])
    client.close()
    
    return contexto

def respuesta_promt(consulta: List[str], contexto: str):
    prompt = "Eres un ayudante de farmacia, tienes que conseguir hacer la pregunta correcta para saber que producto es mejor, teniendo en cuenta los productos relevantes y el contexto anterior, contesta únicamente con la pregunta que harías al usuario. La pregunta nunca puede ser sobre si le interesa tal o cual producto"
    prompt += "\ncontexto " + contexto
    prompt += "\nconversación:\n"
    
    # Procesar cada mensaje de la conversación
    for i, mensaje in enumerate(consulta):
        if i % 2 == 0:
            prompt += f"Usuario: {mensaje}\n"
        else:
            prompt += f"Asistente: {mensaje}\n"
    
    return generative_multimodal_model.generate_content(prompt)

def buscar_producto(consulta: List[str], veto: List[int]) -> Tuple[List[dict], str]:
    # Generar el prompt para el sistema RAG
    data = gen_conv_KNN(consulta, veto)

    productos_encontrados, contexto = buscar_cercanos(data, veto)
    
    respuesta = respuesta_promt(consulta, contexto).text

    return productos_encontrados, respuesta

# Inicializa Flask
app = Flask(__name__)
CORS(app) # Permitir solicitudes desde cualquier origen

# Modifica la función main para que sea un endpoint
@app.route('/consulta', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400
            
        user_input = data.get('message')
        pass_input = data.get('password')
        veto_input = data.get('veto', [])  # Si no se proporciona veto, usar lista vacía

        # Comprobar contraseña
        if not hashlib.sha256(pass_input.encode()).hexdigest() == "bcbe984a9013368372962bd78a4c16b1a2f67c38cb400ed8a9a3a2173712374c":
            return jsonify({'Password': 'La contraseña es incorrecta'}), 400
        
        # Comprobar contraseña con mensaje vacío
        if not user_input and hashlib.sha256(pass_input.encode()).hexdigest() == "bcbe984a9013368372962bd78a4c16b1a2f67c38cb400ed8a9a3a2173712374c":
            return jsonify({'Password': 'La contraseña es correcta'}), 200

        productos_encontrados, respuesta = buscar_producto(user_input, veto_input)
        
        # Estructura de respuesta deseada
        response = {
            "Recomendaciones": productos_encontrados,
            "Respuesta": respuesta
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)