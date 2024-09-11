# Streaming chatbot with Azure OpenAI functions

This chatbot utilizes OpenAI's function calling feature to invoke appropriate functions based on user input and stream the response back.

On top of the standard chat interface, the UI exposes the particular function called along with its arguments, as well as the response from the function.

**The current configuration defines one OpenAI function that can be called**:
- `get_fashion_recommendation`: returns fashion recommendation for a given session.
  - Any change could be done in `app.py` or `app_speech.py`


# Chatbot con funciones de Azure OpenAI

Este chatbot utiliza la función de llamadas de OpenAI para invocar consultas basadas en la entrada del usuario y transmitir la respuesta de vuelta.

Además de la interfaz de chat estándar, la interfaz de usuario expone una función particular junto con sus consulta o sesión, así como la respuesta de la función.

**La configuración actual define una función de OpenAI que llama**:
- `get_fashion_recommendation`: devuelve una recomendación de moda para una sesión o consulta dada.
  - Cualquier cambio se puede realizar en `app.py` o `app_speech.py`.
