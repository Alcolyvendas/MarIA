import json
import ast
import os
from openai import AsyncAzureOpenAI
import chainlit as cl
import pickle
import pandas as pd

# Añadidas nuevas -------------------------------------------------------------------------------

import io
import scipy.sparse as sps
import numpy as np

from scipy.sparse import csr_matrix
from datetime import datetime

from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Pipeline.utils import create_mapping, get_mapped_sessions_to_recommend, get_items_to_exclude, predict_my_purchase

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import load_npz
from scipy.sparse import coo_matrix

import nltk
from nltk.stem.porter import PorterStemmer
# -----------------------------------------------------------------------------------------------

cl.instrument_openai()

client = AsyncAzureOpenAI()

# marIA. Datos de entrada, para encodear/decodear -----------------------------------------------
# Estos datos de entrada tiene que venir encodeados previamente(Susana), para que se pueda utilizar en la aplicación

def cargar_pkls():
    # Cargar el DataFrame resultante desde el archivo pickle

    with open('./dataEncoderDecoder_item_id.pkl', 'rb') as file:
        data = pickle.load(file)            
        
    digested_items = data['result']
    X_sparse = data['X_sparse']
    cv = data['cv']            
    
    # Convertir la matriz dispersa a una matriz densa si es necesario
    X = X_sparse.toarray()

    return digested_items, cv, X

result, cv, X = cargar_pkls()
#similarity = None

# Esto genera un candidato a dar al recomendador, todas las preguntas podrían irse acumulando en la misma sesión
# para ir acumulando las queries e ir pasandolas todas cada vez que hay una nueva!

def compute_query_similarity(query, X, cv, threshold=0.1):
    """
    Calcula la similitud entre la consulta y todos los ítems existentes usando una matriz dispersa guardada.
    
    Args:
    - query: La consulta para calcular similitudes.
    - X: La matriz de características (muestras x características).
    - cv: El CountVectorizer o TF-IDF Vectorizer utilizado para transformar la consulta.
    - sparse_matrix_filename: El nombre del archivo que contiene la matriz dispersa de similitud.
    - threshold: El umbral de similitud para considerar valores relevantes.
    
    Returns:
    - sparse_matrix: La matriz dispersa de similitudes.
    - query_similarity: La similitud de la consulta con todos los ítems.
    """

    ps = PorterStemmer()
    
    def stem(text):
        y = []

        for i in text.split():
            y.append(ps.stem(i))

        return " ".join(y)
    
    query_stemmed = stem(query.lower())
    #print(f"query_stemmed:{query_stemmed}")
    
    # Transformar la consulta en un vector
    query_vector = cv.transform([query_stemmed]).toarray()    
    #print("query_vector",query_vector)

    # Cargar la matriz dispersa desde el archivo
    #sparse_matrix = load_npz(sparse_matrix_filename)
    
    # Calcular la similitud entre la consulta y todos los ítems
    query_similarity = cosine_similarity(query_vector, X).flatten()
    #print("query_similarity",query_similarity)

    # Filtrar similitudes que superan el umbral
    data = []
    rows = []
    cols = []
    for i, sim in enumerate(query_similarity):
        if sim > threshold:
            data.append(sim)
            rows.append(0)  # Fila 0 para la consulta
            cols.append(i)
    
    # Crear la matriz dispersa de similitudes para la consulta
    query_sparse_matrix = coo_matrix((data, (rows, cols)), shape=(1, X.shape[0]))
    #print("query_sparse_matrix",query_sparse_matrix)

    return query_sparse_matrix, query_similarity

def get_top_items(query_sparse_matrix, query_similarity, top_n=5):
    """
    Obtiene los índices de los ítems más similares desde la matriz dispersa.
    
    Args:
    - query_sparse_matrix: La matriz dispersa de similitudes para la consulta.
    - query_similarity: La similitud de la consulta con todos los ítems.
    - top_n: Número de ítems más similares a recuperar.
    
    Returns:
    - items_list: Lista de los ítems más similares.
    """
    # Convertir la matriz dispersa a densa si es necesario
    dense_similarity = query_sparse_matrix.toarray().flatten()
    
    # Obtener los índices de los ítems más similares
    item_indices = np.argsort(-dense_similarity)[:top_n]
    items_list = [(i, dense_similarity[i]) for i in item_indices]
    
    return items_list

# Esto genera posibles candidatos a dar a la sesión que alimenta el recomendador, todas las preguntas se van acumulando en la misma sesión
# para ir acumulando las queries e ir pasandolas todas cada vez que hay una nueva!
def encoding2recomender(yourQuery, session_id=1):
    
    #print("yourQuery",yourQuery)

    query_sparse_matrix, query_similarity = compute_query_similarity(yourQuery, X, cv)
    #print("query_sparse_matrix",query_sparse_matrix)

    items_list = get_top_items(query_sparse_matrix, query_similarity)    
    #print("items_list",items_list)

    candidates_query = None
    # Crear la salida en formato CSV con solo el primer ítem
    if items_list:  # Verificar que hay elementos en items_list
        candidates = []
        # Crear DataFrame con todos los ítems
        for item_index in items_list:
            item_id = result.iloc[item_index[0]].item_id
            candidates.append({
                'session_id': session_id,
                'item_id': item_id,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            })
        
        # Convertir el resultado a un DataFrame
        candidates_query = pd.DataFrame(candidates)
    else:
        # Devolver un DataFrame vacío si items_list está vacío
        candidates_query = pd.DataFrame(columns=['session_id', 'item_id', 'date'])        
    
    return candidates_query 

# Fase de decodificación!
# Si esto es la salida del recomendador, hay de decodearlo para devolverlo al usuario (respuesta del chat)
# Siempre será una sesión nada más, pero para ver variedad!

def decoding2chat(prediccion_df):
    """
    Añade la columna 'item_descrip' a prediccion_df basándose en el DataFrame result.
    
    Parameters:
    prediccion_df (pd.DataFrame): DataFrame con columnas 'session_id', 'item_id', 'rank'.
    result (pd.DataFrame): DataFrame con las columnas 'item_id', 'item_descrip', 'item_keywords'.
    
    Returns:
    pd.DataFrame: prediccion_df con la nueva columna 'item_descrip'.
    """

    # Crear un diccionario para la búsqueda rápida
    item_descrip_dict = result.set_index('item_id')['item_descrip'].to_dict()

    # Función para obtener la descripción del item
    def obtener_descripcion(item_id):
        return item_descrip_dict.get(item_id, '')

    # Aplicar la función a cada item_id en prediccion_df
    prediccion_df['item_descrip'] = prediccion_df['item_id'].apply(obtener_descripcion)

    #return prediccion_df
    #joined_df = decoded_prediccion_df.groupby('session_id')['item_descrip'].apply(lambda x: ', '.join(x)).reset_index()
    listed_df = prediccion_df.groupby('session_id')['item_descrip'].apply(list).reset_index()
    
    return listed_df        

candidates_df = pd.DataFrame(columns=['session_id', 'item_id', 'date'])

def getCandidates():
    return candidates_df

def setCandidates(candidates):
    global candidates_df
    candidates_df = candidates

def get_fashion_recommendation(yourQuery):
    
    print("[get_fashion_recommendation].yourQuery",yourQuery)

    candidates_query = encoding2recomender(yourQuery)
    #print("candidates_query",candidates_query)
    candidates_session = getCandidates()
    eureka = False
    # Buscar el primer item_id de candidates_query que no esté en la sesión
    for index, row in candidates_query.iterrows():
        if row['item_id'] not in candidates_session['item_id'].values:
            # Crear un nuevo candidato para agregar a la sesión, candidates_session
            new_candidate = pd.DataFrame([{
                'session_id': row['session_id'],
                'item_id': row['item_id'],
                'date': row['date']
            }])
            # Añadir nuevo candidato a la sesión
            setCandidates(pd.concat([candidates_session, new_candidate], ignore_index=True))
            eureka = True
            break  # Solo añadimos el primer item_id que no existe en la sesión y luego salimos del bucle

    data = {
        'recomendaciones': "No fue posible obtener recomendaciones con la información que me das."
    }

    if eureka:
        current_session = getCandidates()
        print("Tras añadir el nuevo candidato, la sesión queda ahora como ...")
        print(current_session)

        results_recommendings = recommend2me(current_session.copy())
        #print("results_recommendings",results_recommendings)

        decoded_prediccion_df = decoding2chat(results_recommendings)
        
        #print("decoded_prediccion_df",decoded_prediccion_df)
        #print("Resultados",decoded_prediccion_df.iloc[0]['item_descrip'])
        data = {
            'recomendaciones': decoded_prediccion_df.iloc[0]['item_descrip']
        }
    else:
        print("No hay nada añadido nuevo a la sesión!")

    return json.dumps(data)


def recommend2me(test_sessions_df):
    
    # Carga de parámetros del modelo
    #       W_sparse: matriz de similaridad de items
    #       item_features_df: dataframe de características de items
    #       candidate_items_df: dataframe de candidatos a items
    #       unique_interactions: número de interacciones únicas
    #       purch_weight: peso de las compras
    #       view_weight: peso de las visualizaciones

    W_sparse = None
    item_features_df = None 
    candidate_items_df = None
    unique_interactions = None
    purch_weight = None
    view_weight = None

    # ----------------------------------------------------------------------------------------------------------------
    # Función para cargar el modelo
    #       folder_path: ruta de la carpeta donde se encuentra el archivo PKL
    #       file_name: nombre del archivo PKL
    def load_model(folder_path, file_name=None):

        if file_name is None:
             file_name = 'RP3betaRecommender.pkl' 

        # Cargar los parámetros del archivo PKL
        with open(folder_path + file_name, 'rb') as f:
            parameters = pickle.load(f)

        # Recuperar el buffer de NPZ
        npz_data = parameters['npz_data']

        #print ("(1)Dato cargado!")

        # Leer la matriz NPZ desde el buffer de bytes
        npz_buffer = io.BytesIO(npz_data)
        W_sparse = sps.load_npz(npz_buffer)

        # Recuperar otros parámetros
        candidate_items_df = parameters['candidate_items_df']
        #print ("(2)Dato cargado!")
        item_features_df = parameters['item_features_df']
        #print ("(3)Dato cargado!")
        unique_interactions = parameters['unique_interactions']
        #print ("(4)Dato cargado!")
        purch_weight = parameters['purch_weight']
        #print ("(5)Dato cargado!")
        view_weight = parameters['view_weight']

        #print (f"Datos del modelo {file_name[:-4]} cargados!")

        return W_sparse, item_features_df, candidate_items_df, unique_interactions, purch_weight, view_weight

    W_sparse, item_features_df, candidate_items_df, unique_interactions, purch_weight, view_weight = load_model("./","RP3betaRecommender.pkl")  

    # ----------------------------------------------------------------------------------------------------------------

    # Funciones auxiliares para la preparación de los datos de entrada -----------------------------------------------
    def create_csr_matrix(df, M, N):
        return sps.csr_matrix((df['score'].values,
                              (df['session_id'].values, df['item_id'].values)),
                             shape=(M, N))

    def split_dataframes_test(
            item_features_df, candidate_items_df, test_sessions_df,
            unique_interactions=True, view_weight=1, purch_weight=1,
    ):
        test_sessions_df['score'] = view_weight
        test_sessions_df = test_sessions_df.sort_values(by=['session_id', 'date'], ascending=[True, True]).reset_index(drop=True)

        item_mapping = create_mapping(item_features_df['item_id'])
        test_session_mapping = create_mapping(test_sessions_df['session_id'])

        mapped_test_sessions_df = get_mapped_sessions_to_recommend(test_sessions_df, test_session_mapping)

        recommendable_items = candidate_items_df['item_id'].values
        items_to_ignore = get_items_to_exclude(item_features_df, recommendable_items)

        mapped_items_to_ignore = [item_mapping[item] for item in items_to_ignore]

        test_sessions_df['session_id'] = test_sessions_df['session_id'].map(test_session_mapping)
        test_sessions_df['item_id'] = test_sessions_df['item_id'].map(item_mapping)

        if unique_interactions:
            test_sessions_df.drop_duplicates(subset=['session_id', 'item_id'], inplace=True, keep='last')

        return test_sessions_df, test_session_mapping, item_mapping, mapped_items_to_ignore, mapped_test_sessions_df


    def get_URM_test(
            item_features_df, candidate_items_df, test_sessions_df,
            unique_interactions=True, view_weight=1, purch_weight=1, 
    ):
        test_sessions_df, test_session_mapping, \
        item_mapping, mapped_items_to_ignore, mapped_test_sessions_df = \
            split_dataframes_test(
                item_features_df, candidate_items_df, test_sessions_df,
                unique_interactions=unique_interactions, view_weight=view_weight, purch_weight=purch_weight, 
            )

        URM_test_views = create_csr_matrix(test_sessions_df, len(test_session_mapping), len(item_mapping))

        return URM_test_views, mapped_items_to_ignore, mapped_test_sessions_df, test_session_mapping, item_mapping

    # Carga del modelo -------------------------------------------------------------------------------------------


    # Inicializar una URM_train vacia 
    URM_train_dummy = csr_matrix((0, 0), dtype=np.float32)

    # Crear una instancia de RP3betaRecommender
    rp3_recommender = RP3betaRecommender(URM_train_dummy)

    if (W_sparse is None):
        # Esta parte nunca va a hacer falta realmente, 
        # porque siempre se va a cargar el modelo desde el archivo pkl

        # Cargar el modelo desde el archivo zip
        folder_path = "./"  # El directorio actual
        file_name = "RP3betaRecommender.zip"

        # Llamar al método load_model para cargar los datos en la instancia, la matriz npz
        rp3_recommender.load_model(folder_path, file_name)
    else:
        rp3_recommender.set_W_sparse(W_sparse)
        #print("Cargados desde el fichero pkl!")


    if candidate_items_df is None:
        candidate_items_df = pd.read_csv('./datasets/candidate_items.csv')
        item_features_df = pd.read_csv('./datasets/item_features.csv')

    # ----------------------------------------------------------------------------------------------------------------

    # Preparacion de los datos de entrada para el recomendador -------------------------------------------------------
    if unique_interactions is None:
        unique_interactions = True
        purch_weight = 1
        view_weight = 0.2


    URM_test_views, mapped_items_to_ignore, mapped_test_sessions_df, test_session_mapping, item_mapping = get_URM_test(
        test_sessions_df=test_sessions_df,
        candidate_items_df=candidate_items_df,
        item_features_df=item_features_df,
        unique_interactions=unique_interactions,
        purch_weight=purch_weight,
        view_weight=view_weight,
    )

    # ----------------------------------------------------------------------------------------------------------------

    # Predicciendo con los datos de entrada que se tienen ------------------------------------------------------------
    rp3_recommender.set_URM_train(URM_test_views)
    rp3_recommender.set_items_to_ignore(mapped_items_to_ignore)

    # Recomendando ---------------------------------------------------------------------------------------------------
    # Top 10 mejores items a recomendar!
    cutoff = 10

    df_recommendings = predict_my_purchase(
        models=[rp3_recommender],
        session_ids=mapped_test_sessions_df,
        add_item_score=False,
        cutoff=cutoff, # Los 100 mejores resultados originalmente
    )
    
    mapped_results = df_recommendings[0]
    
    #print("mapped_results",mapped_results)

    def save_results(prediction_df, item_mapping, session_mapping, save_path="./", cutoff=10, output_format="console"):
        """
        Save the prediction results in the specified format (csv, json, etc.)

        Parameters:
        - prediction_df: DataFrame containing the predictions.
        - item_mapping: Dictionary mapping item IDs to original item IDs.
        - session_mapping: Dictionary mapping session IDs to original session IDs.
        - save_path: Directory where the results will be saved.
        - cutoff: Number of top items to rank.
        - output_format: The format in which to save the results ("csv", "json","console").
        # console: do nothing, just return it!

        Returns:
        - The modified prediction DataFrame.
        """

        # Preparing the DataFrame
        prediction_df = prediction_df[['session_id', 'item_id']]
        num_sessions = len(np.unique(prediction_df['session_id']))
        rank_col = list(range(1, cutoff + 1)) * num_sessions
        prediction_df['rank'] = rank_col

        # Create inverse mappings
        inv_item_map = {v: k for k, v in item_mapping.items()}
        inv_session_map = {v: k for k, v in session_mapping.items()}

        # Map to original IDs
        prediction_df['item_id'] = prediction_df['item_id'].map(inv_item_map)
        prediction_df['session_id'] = prediction_df['session_id'].map(inv_session_map)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Current date and time for file naming
        now = datetime.now()

        # Determine the file format and save accordingly
        if output_format == "csv":
            final_path = os.path.join(save_path, f'results_{now:%Y_%m_%d_at_%H_%M_%S}.csv')
            prediction_df.to_csv(final_path, index=False)
            print(f"Resultados salvados en {final_path}")
        elif output_format == "json":
            final_path = os.path.join(save_path, f'results_{now:%Y_%m_%d_at_%H_%M_%S}.json')
            prediction_df.to_json(final_path, orient="records", lines=True)
            print(f"Resultados salvados en {final_path}")
        #elif output_format == "console":
        #    print("No hacer nada, se devuelve y ya está!")
        #else:
        #    raise ValueError(f"Unsupported output format: {output_format}")

        return prediction_df


    results = save_results(
        prediction_df=mapped_results,
        item_mapping=item_mapping,
        session_mapping=test_session_mapping,
        save_path='./', cutoff = cutoff,
    )
    
    #print("Resultados",results )

    return results


# -----------------------------------------------------------------------------------------------

MAX_ITER = 5

def get_current_weather(location, unit):
    unit = unit or "Fahrenheit"
    weather_info = {
        "location": location,
        "temperature": "60",
        "unit": unit,
        "forecast": ["windy"],
    }
    return json.dumps(weather_info)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fashion_recommendation",
            "description": "Get a list of recommended items for a given query or queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "yourQuery": {
                        "type": "string",
                        "description": "The query about fashion/clothes the user wants a recommendation from, e.g. birthday's wife just right now",
                    }
                },
                "required": ["yourQuery"],
            },
        }
    }
]



#@cl.on_chat_start
#def start_chat():

    # Define el prompt del sistema
    #SYSTEM_PROMPT = "Eres un asistente de compras personales especializado en moda llamada MarIA, dedicado a ayudar a los usuarios a encontrar prendas y accesorios que se adapten perfectamente a su estilo, preferencias y necesidades. Ofrece recomendaciones detalladas y sugerencias de combinaciones de atuendos, teniendo en cuenta las tendencias actuales y la información proporcionada por el usuario. Mantén un tono elegante, amigable y profesional en todo momento, asegurándote de que la experiencia del usuario sea sofisticada y personalizada."


    
    #cl.user_session.set(
    #    "message_history",
    #    [{"role": "assistant", "content": SYSTEM_PROMPT}],
    #)


@cl.on_chat_start
async def on_chat_start():

    # Define el prompt del sistema
    SYSTEM_PROMPT = "Eres un asistente de compras personales especializado en moda llamada MarIA, dedicada a ayudar a los usuarios a encontrar prendas y accesorios que se adapten perfectamente a su estilo, preferencias y necesidades. Ofrece recomendaciones detalladas y sugerencias de combinaciones de atuendos, teniendo en cuenta las tendencias actuales y la información proporcionada por el usuario. Mantén un tono elegante, amigable y profesional en todo momento, asegurándote de que la experiencia del usuario sea sofisticada y personalizada."
    #cl.css('styles.css')

    cl.user_session.set(
        "message_history",
        [{"role": "assistant", "content": SYSTEM_PROMPT}],
    )
    init_message = "¡Hola! Soy MarIA, tu asistente de compras especializada en moda y accesorios. Voy a asistirte para ayudarte a encontrar la prenda perfecta."
    await cl.Message(content=init_message).send()

""" @cl.set_starters
async def set_starters():
    init_message = "¡Hola! Soy MarIA, tu asistente de compras especializada en moda y accesorios. Voy a asistirte para ayudarte a encontrar la prenda perfecta."
    
    return [
        cl.Starter(
            label="MarIA",
            message= init_message,
            icon="./public/avatars/avatar1.jpg",
            ),
 
        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            icon="https://www.svgrepo.com/show/412769/learn.svg",
            )
    ] """



#@cl.set_starters
#async def set_starters():
#    return [
#        cl.Starter(
#            label="Morning routine ideation",
#            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
#            icon="https://www.svgrepo.com/show/499853/idea.svg",
#            ),
 
#        cl.Starter(
#            label="Explain superconductors",
#            message="Explain superconductors like I'm five years old.",
#            icon="https://www.svgrepo.com/show/412769/learn.svg",
#            ),
#        cl.Starter(
#            label="Python script for daily email reports",
#            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
#            icon="https://www.svgrepo.com/show/507435/terminal.svg",
#            ),
#        cl.Starter(
#            label="Text inviting friend to wedding",
#            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
#            icon="https://www.svgrepo.com/show/477084/write-document.svg",
#            )
#        ]

@cl.step(type="tool")
async def call_tool(tool_call_id, name, arguments, message_history):
    arguments = ast.literal_eval(arguments)

    current_step = cl.context.current_step
    current_step.name = name
    current_step.input = arguments

    if name == "get_current_weather":
        function_response = get_current_weather(
            location=arguments.get("location"),
            unit=arguments.get("unit"),
        )
    elif name == "get_fashion_recommendation":
        function_response = get_fashion_recommendation(
            yourQuery=arguments.get("yourQuery")
        )
        
    current_step.output = function_response
    current_step.language = "json"

    message_history.append(
        {
            "role": "function",
            "name": name,
            "content": function_response,
            "tool_call_id": tool_call_id,
        }
    )

async def call_gpt4(message_history):
    settings = {
        "model": "gpt-4",
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    tool_call_id = None
    function_output = {"name": "", "arguments": ""}

    final_answer = cl.Message(content="", author="Answer")
    #final_answer = cl.Message(content="", author="Answer", icon="./imagenes/your-custom-avatar.png")

    async for part in stream:
        try:
            new_delta = part.choices[0].delta
            tool_call = new_delta.tool_calls and new_delta.tool_calls[0]
            function = tool_call and tool_call.function
            if tool_call and tool_call.id:
                tool_call_id = tool_call.id

            if function:
                if function.name:
                    function_output["name"] = function.name
                else:
                    function_output["arguments"] += function.arguments
            if new_delta.content:
                if not final_answer.content:
                    await final_answer.send()
                await final_answer.stream_token(new_delta.content)
        except Exception as e:
            print(f"An error occurred: {e}")

    if tool_call_id:
        await call_tool(
            tool_call_id,
            function_output["name"],
            function_output["arguments"],
            message_history,
        )

    if final_answer.content:
        await final_answer.update()

    return tool_call_id


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    #message_history = []
    message_history.append({"role": "user", "content": message.content})

    cur_iter = 0

    while cur_iter < MAX_ITER:
        tool_call_id = await call_gpt4(message_history)
        if not tool_call_id:
            break

        cur_iter += 1
