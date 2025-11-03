from focus_group import FocusGroup
from model import YandexModel


if __name__ == "__main__":
    MODEL_NAME = "yandexgpt"
    YC_FOLDER_ID = "your_folder_id"
    YC_API_KEY = "your_api_key"
    MODEL_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    model = YandexModel(MODEL_NAME, MODEL_URL, YC_API_KEY, YC_FOLDER_ID)
    focus_group = FocusGroup(model_for_preprocessing=model, use_auto_preprocessing=True, interviews_path="interviews",
                             avatars_path="avatars", iterations=3)
    print(focus_group.add_agents(model))
    try:
        focus_group.conduct_interview("text.txt", "log.txt")
    except Exception as e:
        print(e)