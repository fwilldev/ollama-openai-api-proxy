# ollama-openai-api-proxy

## Overview
`ollama-openai-api-proxy` is a lightweight application which proxies [ollama/ollama](https://github.com/ollama/ollama) compatible API requests to OpenAI's API.

One of the main purposes is to leverage OpenAI's API with AI Assistant Applications like Jetbrains AI Assistant.

The application uses the official [openai/openai-python](https://github.com/openai/openai-python) library and OpenAI's [Responses API](https://platform.openai.com/docs/api-reference/responses). 

## Current Endpoints

| Ollama Endpoint  | Action                                                                                                           |
|------------------|------------------------------------------------------------------------------------------------------------------|
| GET `/api/tags`  | Returns Model IDs compatible with OpenAI's API                                                                   |
| POST `/api/chat` | Proxies the Ollama compatible request to OpenAI's Response API and provides a Ollama compatible Stream response. |

## Installation

### OpenAI API Key

Either way you setup the application, you need to create an `.env` file with your `OPENAI_API_KEY`:

For example with:

```bash
  echo "OPENAI_API_KEY=your-api-key" > .env
```

### Docker

Use `docker compose` to run this application in a container:

```bash
   docker compose up --build -d
```

### Run from source

To set up the application, ensure that you have Python 3.9 and `virtualenv` installed. Follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ollama-openai-api-proxy
   ```

2. Create a virtual environment:
   ```bash
   virtualenv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Install FastAPIs CLI:
```bash
    pip install "fastapi[standard]"
```
5. Run:
```bash
    fastapi dev main.py
```

## Usage

### Jetbrains AI Assistant

You can use this proxy as an offline model for Jetbrains AI Assistant.

1. Go to `Settings`
2. Got to `Tools` --> `AI Assistant`
3. Under `Models` --> `Enable Ollama` and change the port to `8000`
4. Now in the AI Chat, you can select the model under `Ollama`

## TODOs

- Proxy more Ollama Endpoints to OpenAI
- Small frontend with:
  - Usage of given API Key
  - Costs of given API Key

## License

See [LICENSE](https://github.com/fwilldev/ollama-openai-api-proxy/blob/main/LICENSE)
