from abc import ABC, abstractmethod
import requests


class Model(ABC):
    """
    Parent class for any models.
    """
    def __init__(self, model_name: str, model_url:str, api_key:str) -> None:
        self.model_name = model_name
        self.model_url = model_url
        self.api_key = api_key

    @abstractmethod
    def generate_response(self, system_message: str, user_message:str) -> str:
        pass


class YandexModel(Model):
    """
    Class for YandexGPT
    """
    def __init__(self, model_name: str, model_url: str, api_key: str,
                 yandex_folder: str, temperature: float = 0.5, max_tokens: int = 2000) -> None:
        super().__init__(model_name, model_url, api_key)
        self.folder = yandex_folder
        self.temperature = temperature
        self.max_tokens = max_tokens


    def generate_response(self, system_message: str, user_message:str) -> str:
        """
        Obtains response from Yandex GPT through Yandex Cloud API
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder,
        }
        payload = {
            "modelUri": f"gpt://{self.folder}/{self.model_name}",
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature if self.temperature > 0 else 0.1,
                "maxTokens": self.max_tokens
            },
            "messages": [
                {
                    "role": "system",
                    "text": system_message
                },
                {
                    "role": "user",
                    "text": f"Вопрос: {user_message}"
                }
            ]}

        response_content = ""
        try:
            with requests.post(self.model_url, headers=headers, json=payload) as response:
                result = response.json()
                response_content = result["result"]["alternatives"][0]["message"]["text"]
        except Exception as e:
            print(f"Error in Yandex GPT request: {e}")
            response_content = None
        return response_content

