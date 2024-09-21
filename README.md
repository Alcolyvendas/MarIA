# Chatbot de Recomendaciones de Moda usando Azure OpenAI y Chainlit

Este proyecto forma parte de nuestro Trabajo Fin de Máster, y como tal ha sido desarrollado con fines educativos y de investigación. Este implementa un chatbot de recomendaciones de moda personalizado, **MarIA**, utilizando Azure OpenAI, Chainlit y varios algoritmos de recomendación populares. El bot ayuda a los usuarios a encontrar artículos de ropa que se ajusten a su estilo y preferencias al proporcionar sugerencias basadas en las consultas de los usuarios. Soporta respuestas en streaming y utiliza la función de llamada de Azure OpenAI para obtener recomendaciones de moda.

## Características

- **Recomendaciones de Moda Personalizadas**: El chatbot analiza las consultas de los usuarios y proporciona recomendaciones de moda.
- **Recomendaciones Basadas en Sesiones**: Las consultas se acumulan dentro de una sesión para refinar el proceso de recomendación.
- **Llamadas a Funciones con Azure OpenAI**: El bot utiliza el mecanismo de llamada a funciones de OpenAI para obtener recomendaciones de moda.
- **Respuestas en Streaming**: El chatbot transmite sus respuestas en tiempo real.
- **Integración con Sistema de Recomendación Personalizado**: El chatbot interactúa con un motor de recomendación de moda personalizado que utiliza datos pre-codificados y similitud coseno.
- **Procesamiento de Audio**: La aplicación soporta el procesamiento de audio para la expansión en reconocimiento de voz (versión app_speech.py).

## Tecnologías

- **Python**: Lenguaje principal para la aplicación.
- **Chainlit**: Utilizado para crear la interfaz interactiva del chatbot.
- **Azure OpenAI**: Proporciona las capacidades de procesamiento del lenguaje natural y llamadas a funciones.
- **Pandas**: Para la manipulación y preprocesamiento de datos.
- **Scikit-learn**: Utilizado para calcular la similitud coseno entre consultas y artículos.
- **NLTK**: Toolkit de procesamiento de lenguaje natural para el preprocesamiento de texto.
- **Pickle**: Para cargar modelos y datos preprocesados.

## Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/Alcolyvendas/fashion-recommendation-chatbot.git
   cd fashion-recommendation-chatbot
   ```

2. **Instalar dependencias**:
   Asegúrate de tener Python 3.8 o superior instalado. Ejecuta el siguiente comando para instalar las bibliotecas necesarias:
   ```bash
   pip install -r requirements.txt o pip install -r requirements_speech.txt
   ```

3. **Configurar Azure OpenAI**:
   Necesitarás tus credenciales de Azure OpenAI para conectarte al servicio. Configura las variables de entorno necesarias o agrégalas a un archivo `.env`:
   - `AZURE_OPENAI_API_KEY`
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_DEPLOYMENT_ID`
     
   (versión app_speech.py, igualmente necesitarás las credenciales del servicio Azure Speech)
   - `AZURE_SPEECH_KEY`
   - `AZURE_SPEECH_REGION`
   - `AZURE_SPEECH_ENDPOINT`     

4. **Asegúrate de que FFmpeg esté instalado**:
   Si se requiere procesamiento de audio, instala FFmpeg y asegúrate de que esté disponible en el PATH de tu sistema, el chainlit através del micrófono captura el audio en formato `ogg`, pero Azure, no lo admite, así pues hay que convertirlo a formato `wav`.

## Uso

1. **Ejecutar el chatbot**:
   Usa el siguiente comando para iniciar el chatbot:
   ```bash
   chainlit run app.py -w o chainlit run app_speech.py -w
   ```

2. **Interactuar con el bot**:
   Abre la interfaz de Chainlit y comienza a interactuar con **MarIA**, tu asistente de moda. Puedes pedir recomendaciones de moda escribiendo consultas como:
   - "Necesito un vestido para una boda."
   - "¿Qué debería usar para una reunión de negocios?"

   El chatbot responderá con recomendaciones personalizadas basadas en las consultas de tu sesión.

## Funciones Principales
   A continuación, te doy una breve explicación que describe el propósito de cada función y cómo se integran en el flujo de trabajo general del sistema.

- **`get_fashion_recommendation`**: Esta función principal que procesa las consultas de la sesión de un usuario, para devolver recomendaciones de moda personalizadas y obtiene recomendaciones de moda basadas en ellas.
    - **Parámetros**:
      - `yourQuery`: la consulta ingresada por el usuario.
      
    - **Proceso**:
      - Utiliza `encoding2recomender` para generar una lista de artículos candidatos basados en la consulta.
      - Compara los candidatos generados con los artículos ya presentes en la sesión actual.
      - Si se encuentra un nuevo candidato, se agrega a la sesión.
      - Si se agrega un nuevo artículo a la sesión, el sistema usa `recommend2me` para generar recomendaciones finales y las decodifica con `decoding2chat`.
      - Si no se encuentran nuevos candidatos, devuelve un mensaje indicando que no se pudieron hacer recomendaciones basadas en la información proporcionada.
    
    - **Salida**: Un JSON con las recomendaciones decodificadas o un mensaje de error si no se encontraron recomendaciones.
  
- **`encoding2recomender`**: Codifica la nueva consulta para la sesión, genera artículos candidatos para el recomendador basados en la nueva consulta obtenida.
    - **Parámetros**:
      - `yourQuery`: la consulta del usuario, que se transforma en una matriz dispersa para calcular su similitud con los artículos disponibles.
      - `session_id`: un identificador para la sesión actual (predeterminado es 1).
      
    - **Proceso**:
      - Calcula la similitud entre la consulta y los artículos disponibles usando `compute_query_similarity`.
      - Extrae los artículos más relevantes usando `get_top_items`.
      - Devuelve un DataFrame con los artículos candidatos, incluyendo `session_id` y la `fecha` de la recomendación.
      
    - **Salida**: Un `DataFrame` que contiene las columnas `session_id`, `item_id` y `date`.

- **`compute_query_similarity`**: Calcula la similitud entre consultas de usuarios y descripciones de artículos almacenados utilizando similitud coseno para obtener los artículos candidatos vistos para `get_fashion_recommendation`.

- **`decoding2chat`**: Toma las predicciones del modelo y añade las descripciones de esos artículos, haciéndolos más comprensibles en una interfaz de chat, decodifica los artículos recomendados para devolverlos en un formato amigable para el usuario en el chat.
    - **Parámetros**:
      - `prediccion_df`: lista resultado con las columnas `session_id`, `item_id` y `rank`, representando los artículos recomendados.
      
    - **Proceso**:
      - Añade una columna `item_descrip` a la lista resultado usando la función `obtener_descripcion`, mapeando el `item_id` a su descripción.
      - Devuelve la lista con las descripciones agrupadas por la sesión.
    
    - **Salida**: Lista de descripciones de los artículos recomendados.
  
- **`call_gpt4`**: Maneja la comunicación con el modelo Azure OpenAI para procesar la entrada del usuario y generar respuestas.
  
- **`recommend2me`**: Realiza el proceso de recomendación en sí, utilizando el modelo `RP3betaRecommender`.

    - **Parámetros**:
      - `test_sessions_df`: Lista de todas las interacciones de la sesión actual.
    
    - **Proceso**:
      - Carga el modelo de recomendación preentrenado desde un archivo `PKL`.
      - Prepara los datos de entrada creando matrices dispersas (`csr_matrix`) y mapeos entre sesiones y artículos.
      - Usa el `RP3betaRecommender` para predecir los artículos más relevantes.
      - Devuelve una lista con las predicciones de los artículos recomendados.
    
    - **Salida**: Listade de los artículos recomendados por el modelo.

---

Cada vez que un usuario hace una consulta, éstas se acumulan dentro de la misma sesión, permitiendo que se pasen juntas al recomendador cuando se hace una nueva consulta. El sistema asegura que solo se agreguen nuevos artículos a la sesión, evitando recomendaciones duplicadas. Aunque esto se controla dentro del sistema, también está controlado explícitamente por un parámetro de configuración en el algoritmo de recomendación elegido.

El sistema está optimizado para ofrecer recomendaciones de moda, pero puede ser **mejorado** ajustando los datos de entrada y el modelo de recomendación. De hecho, debido a su estrategia modular, el algoritmo actual podría ser reemplazado por uno mejor o más óptimo.

---

## Recomendador

Para la construcción del módulo de recomendación, se ha partido de tres de los modelos de referencia del trabajo https://dl.acm.org/doi/10.1145/3556702.3556829 de un grupo participante de la competición RecSys 2022 incorporando nuevas adaptaciones y trabajos específicos para los objetivos de este trabajo

Puedes actualizar la lógica del recomendador modificando los archivos `app.py` o `app_speech.py`.

## Personalización

- **Modificar Recomendaciones**: Para ajustar la lógica de recomendación, puedes modificar las funciones `get_fashion_recommendation` y `compute_query_similarity`.
- **Añadir Nuevas Funciones**: Puedes añadir nuevas funciones en `app.py` o `app_speech.py` para manejar otros tipos de consultas o interacciones de los usuarios.

---


# Fashion Recommendation Chatbot using Azure OpenAI and Chainlit

This project implements a personalized fashion recommendation chatbot, **MarIA**, using Azure OpenAI, Chainlit, and various popular recommendation algorithms. The bot helps users find clothing items that fit their style and preferences by providing suggestions based on user queries. It supports streaming responses and utilizes Azure OpenAI's function calling to fetch fashion recommendations.

## Features

- **Personalized Fashion Recommendations**: The chatbot analyzes user queries and provides fashion recommendations.
- **Session-Based Recommendations**: Queries are accumulated within a session to refine the recommendation process.
- **Function Calling with Azure OpenAI**: The bot utilizes OpenAI's function calling mechanism to fetch fashion recommendations.
- **Streaming Responses**: The chatbot streams its responses in real-time.
- **Integration with Custom Recommender System**: The chatbot interacts with a custom fashion recommendation engine that uses pre-encoded data and cosine similarity.
- **Audio Processing**: The application supports audio processing for future expansion into speech recognition (version app_speech.py).

## Technologies

- **Python**: Core language for the application.
- **Chainlit**: Used to create the interactive chatbot interface.
- **Azure OpenAI**: Provides the natural language processing and function calling capabilities.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: Used for calculating cosine similarity between queries and items.
- **NLTK**: Natural language toolkit for text preprocessing.
- **Pickle**: To load preprocessed models and data.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Alcolyvendas/fashion-recommendation-chatbot.git
   cd fashion-recommendation-chatbot
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8 or higher installed. Run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt or pip install -r requirements_speech.txt
   ```

3. **Set up Azure OpenAI**:
   You'll need your Azure OpenAI credentials to connect to the service. Set the necessary environment variables or add them to a `.env` file:
   - `AZURE_OPENAI_API_KEY`
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_DEPLOYMENT_ID`
     
   (version app_speech.py)
   - `AZURE_SPEECH_KEY`
   - `AZURE_SPEECH_REGION`
   - `AZURE_SPEECH_ENDPOINT`     

5. **Ensure FFmpeg is installed**:
   If audio processing is required, install FFmpeg and ensure it is available in your system's PATH.

## Usage

1. **Run the chatbot**:
   Use the following command to start the chatbot:
   ```bash
   chainlit run app.py -w or chainlit run app_speech.py -w
   ```

2. **Interact with the bot**:
   Open the Chainlit interface and start interacting with **MarIA**, your fashion assistant. You can ask for fashion recommendations by typing queries such as:
   - "I need a dress for a wedding."
   - "What should I wear for a business meeting?"

   The chatbot will respond with personalized recommendations based on your query/session.

## Main Functions
  Let me give you a brief explanation that clearly describes the purpose of each function and how they integrate into the overall system workflow.

- **`get_fashion_recommendation`**: This main function processes a user's query/session to return personalized fashion recommendations, fetches fashion recommendations based on user queries.
    - **Parameters**:
      - `yourQuery`: the user's input query.
      
    - **Process**:
      - Uses `encoding2recomender` to generate a list of candidate items based on the query.
      - Compares the generated candidates with the items already present in the current session.
      - If a new candidate is found, it is added to the session.
      - If a new item is added to the session, the system uses `recommend2me` to generate final recommendations and decodes them with `decoding2chat`.
      - If no new candidates are found, it returns a message indicating that no recommendations could be made based on the provided information.
    
    - **Output**: A JSON with the decoded recommendations or an error message if no recommendations were found.
  
- **`encoding2recomender`**: Encodes your query/session, generates candidate items for the recommender based on a given user query.
    - **Parameters**:
      - `yourQuery`: the user query, which is transformed into a sparse matrix to calculate its similarity to available items.
      - `session_id`: an identifier for the current session (default is 1).
      
    - **Process**:
      - It calculates the similarity between the query and available items using `compute_query_similarity`.
      - Extracts the most relevant items using `get_top_items`.
      - Returns a DataFrame with the recommended items, including `session_id` and the recommendation `date`.
      
    - **Output**: A `DataFrame` containing the columns `session_id`, `item_id`, and `date`.

- **`compute_query_similarity`**: Calculates the similarity between user queries and stored item descriptions using cosine similarity to get the candidates/viewed items for -get_fashion_recommendation-.
  
- **`decoding2chat`**: Takes the model's predictions and adds item descriptions, making them more understandable in a chat interface, decodes the recommended items to return them in a user-friendly format in the chat.
    - **Parameters**:
      - `prediccion_df`: a `DataFrame` with the columns `session_id`, `item_id`, and `rank`, representing the recommended items.
      
    - **Process**:
      - Adds an `item_descrip` column to the DataFrame using the `obtener_descripcion` function, which maps an `item_id` to its corresponding description from the `result` dataset.
      - Returns a `DataFrame` with the descriptions grouped by `session_id`.
    
    - **Output**: A `DataFrame` containing a list of item descriptions for each session.
  
- **`call_gpt4`**: Handles communication with the Azure OpenAI model to process user input and generate responses.
  
- **`recommend2me`**: Performs the recommendation process itself, utilizing the `RP3betaRecommender` model.

    - **Parameters**:
      - `test_sessions_df`: a `DataFrame` containing the interactions of the current session.
    
    - **Process**:
      - Loads the pre-trained recommendation model from a `PKL` file.
      - Prepares input data by creating sparse matrices (`csr_matrix`) and mappings between sessions and items.
      - Uses the `RP3betaRecommender` to predict the most relevant items.
      - Returns a `DataFrame` with the predictions, including the `session_id` and recommended items.
    
    - **Output**: A `DataFrame` containing the items recommended by the model.

---

Each time a user makes a query, the queries can be accumulated within the same session, allowing them to be passed together to the recommender when a new query is made. The system ensures that only new items are added to the session, avoiding duplicate recommendations. Although this is controlled within the system, it is also explicitly controlled by a configuration parameter in the chosen recommendation algorithm.

The system is optimized for offering fashion recommendations but can be **improved** by adjusting the input data and the recommendation model. In fact, due to its modular strategy, the current algorithm could be replaced by a better or more optimal one..

--- 


## Recommender

For the construction of the recommendation module, we started from three reference models from the work https://dl.acm.org/doi/10.1145/3556702.3556829 by a participating group in the RecSys 2022 competition, incorporating new adaptations and specific developments to meet the objectives of this project. You can update the recommender logic by modifying the `app.py` or `app_speech.py` files.

## Customization

- **Modifying Recommendations**: To adjust the recommendation logic, you can modify the `get_fashion_recommendation` and `compute_query_similarity` functions.
- **Adding New Functions**: You can add new functions in the `app.py` or `app_speech.py` to handle other types of user queries or interactions.
