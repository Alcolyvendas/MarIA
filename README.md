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


## Custom Recommender

The project uses a custom graph-based recommender, **RP3betaRecommender**, to generate fashion recommendations based on preprocessed item data. You can update the recommender logic by modifying the `app.py` file or the utility functions in `Pipeline.utils`.

## Customization

- **Modifying Recommendations**: To adjust the recommendation logic, you can modify the `get_fashion_recommendation` and `compute_query_similarity` functions.
- **Adding New Functions**: You can add new functions in the `app.py` or `app_speech.py` to handle other types of user queries or interactions.
