from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import requests
import json


class Agent:
    """
    This class represents a single focus group participant with personality parameters
    and language model access to generate personalized responses.
    """

    def __init__(self, avatar_for_agent, model, tokenizer, calibration_path):
        self.name = avatar_for_agent.get("name", "")
        self.age = avatar_for_agent.get("age", None)
        self.profession = avatar_for_agent.get("profession", "")
        self.sentiment = avatar_for_agent.get("sentiment", 0.5)
        self.traits = avatar_for_agent.get("traits", "")
        self.concerns = avatar_for_agent.get("concerns", "")
        self.key_phrases = avatar_for_agent.get("key_phrases", "")
        self.communication_style = avatar_for_agent.get("communication_style", "")
        self.economic_view = avatar_for_agent.get("economic_view", "")
        self.trust_in_institutions = avatar_for_agent.get("trust_in_institutions", 0.5)
        self.knowledge_level = avatar_for_agent.get("knowledge_level", "")
        self.financial_behavior = avatar_for_agent.get("financial_behavior", "")
        self.policy_priority = avatar_for_agent.get("policy_priority", "")
        self.cb_functions_understanding = avatar_for_agent.get("cb_functions_understanding", "")
        self.cb_perception = avatar_for_agent.get("cb_perception", "")
        self.cb_trust_factors = avatar_for_agent.get("cb_trust_factors", "")
        self.information_sources = avatar_for_agent.get("information_sources", "")
        self.emotional_tone = avatar_for_agent.get("emotional_tone", "")
        self.model = model
        self.tokenizer = tokenizer
        self.calibration_path = calibration_path

    def calibrate(self, calibration_path):
        calibration_files = [f for f in os.listdir(self.calibration_path) if
                             os.path.isfile(os.path.join(self.calibration_path, f))]
        prompts = []
        for calibration_file in calibration_files:
            with open(calibration_file, 'r', encoding='utf-8') as file:
                prompt = """Как ты оцениваешь этот текст с учетом следующих факторов: 
                обоснованность принимаемого решения, полнота объяснений, доступность языка документа, 
                способность документа вызвать доверие к описываемому решению? 
                Формат ответа: одно слово (негативно либо нейтрально либо позитивно).
                Не добавляй в ответ дополнительной информации. Текст: """
                prompt += file.read()
                prompts.append(prompt)
        calibration_results = []
        for prompt in prompts:
            calibration_results.append(self.generate_response(prompt))
        return calibration_results

    def generate_response(self, prompt):
        """
        This method generates a text response to the given prompt, personalized
        based on the agent's traits and profile.
        """
        full_prompt = (
            f"Ты — участник фокус-группы по имени {self.name}, {self.age} лет, "
            f"{self.profession}. Твой характер отличается следующими чертами ({self.traits}), "
            f"и тебя беспокоит следующее: {self.concerns}. "
            f"Ты выражаешь мысли через {self.communication_style}, и твои ключевые фразы — это: {self.key_phrases}. "
            f"Твоя точка зрения на экономику: {self.economic_view}. "
            f"Уровень доверия к институтам: {self.trust_in_institutions}, а уровень знаний — {self.knowledge_level}. "
            f"Ты ведёшь себя так: {self.financial_behavior}, приоритет в политике — {self.policy_priority}. "
            f"Ты понимаешь функции ЦБ как: {self.cb_functions_understanding}, но воспринимаешь его как: {self.cb_perception}, "
            f"доверяешь только при наличии: {self.cb_trust_factors}. "
            f"Ты получаешь информацию из: {self.information_sources}. "
            f"Твой эмоциональный тон: {self.emotional_tone}. "
            f"Вопрос, на который ты отвечаешь: {prompt}"
            f"Если вопрос содержит указания на другие мнения, учти их при формулировке ответа, если они согласуются с твоими убеждениями."
        )
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=200, temperature=self.sentiment)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class FocusGroup:
    """
    This class creates & manages a group of agents and provides tools to interact with them as a unit.
    """

    def __init__(self, interviews_path: str = "interviews",
                 avatars_path: str = "avatars",
                 result_path: str = "result.txt",
                 model_for_preprocessing: str = "deepseek/deepseek-r1",
                 api_key_preprocessing: str = "",
                 use_auto_preprocessing: bool = True,
                 model_for_agent=None,
                 tokenizer_for_agent=None,
                 iterations: int = 2, ):
        """
        This method initializes all agents in the focus group from a list of config dicts.
        The resulting number of agents will be equal to the number of downloaded interviews.
        Parameters:
        interviews_path -> path to the directory with the interviews ("/interviews" by default)
        avatars_path -> path to the directory where ready-to-use avatars will be stored in json format ("/avatars")
        result_path -> pathto the file we want to store the results of interview in ("/result.txt" by default)
        model_for_preprocessing -> name of the model in OpenRouter (DeepSeek R1 by default)
        api_key_preprocessing -> API key for OpenRouter
        use_auto_preprocessing -> whether we like our interviews preprocessed automatically (via OpenRouter API, True by default), if false, avatars are downloaded manually
        model_for_agent -> any pretrained model downloaded locally via transformers (qwen, mistral etc.)
        tokenizer_for_agent -> tokenizer for pretrained model
        iterations -> NB! -> parameter that sets number of times the questions will be processed by each agent
        """
        self.interviews_path = interviews_path
        self.avatars_path = avatars_path
        self.result_path = result_path
        self.model_for_preprocessing = model_for_preprocessing
        self.api_key_preprocessing = api_key_preprocessing
        self.use_auto_preprocessing = use_auto_preprocessing
        self.model_for_agent = model_for_agent
        self.tokenizer_for_agent = tokenizer_for_agent
        self.iterations = iterations

        if self.use_auto_preprocessing:  # if we want avatars created automatically
            self.make_avatars()
        avatar_files = [f for f in os.listdir(self.avatars_path) if os.path.isfile(os.path.join(self.avatars_path, f))]
        avatars = []
        for avatar_file in avatar_files:
            with open(os.path.join(self.avatars_path, avatar_file), "r", encoding="utf-8") as file:
                avatar = json.load(file)
                avatars.append(avatar)
        self.avatars = avatars
        self.agents = []
        for i, avatar_for_agent in enumerate(avatars):
            agent = self.create_agent(avatar_for_agent, self.model_for_agent, self.tokenizer_for_agent)
            self.agents.append(agent)

    def make_avatars(self):

        def process_content(content):
            """cleans the content of the automatically generated tags"""
            return content.replace('<think>', '').replace('</think>', '')

        SYSTEM_PROMPT = """Ты — профессиональный аналитик социологических и экономических исследований.\n
                        Твоя задача — анализировать текстовые расшифровки интервью с респондентами и извлекать ключевые характеристики в строго заданном JSON-формате.\n
                        **Правила обработки:**\n
                        1. Входные данные: Строка (str) с расшифровкой интервью на русском языке.\n
                        2. Анализируй только явно упомянутую информацию и логические выводы из ответов.\n
                        3. Количественные оценки (sentiment, trust) вычисляй по шкале 0.0-1.0 на основе:\n
                        - Лексического анализа (эмоциональные маркеры, модальность)\n
                        - Косвенных индикаторов (уверенность/неуверенность в суждениях)\n
                        4. Для текстовых полей (traits, concerns и др.) используй:\n
                        - Цитаты из интервью (если точно соответствуют)\n
                        - Краткие обобщения (3-5 слов на пункт)\n
                        5. Уровень знаний определяй по:\n
                        - Использованию терминологии\n
                        - Глубине объяснений\n
                        - Упоминанию образования/опыта\n
                        **Требуемый JSON-формат:**\n
                        {\n
                        "name": "Имя из текста (англ. форма)",\n
                        "age": "Цифра (возраст)",\n
                        "profession": "Основной род занятий (до 3 слов)",\n
                        "sentiment": "Float (0.0-1.0) общий позитив тональности",\n
                        "traits": "Черты характера через запятую (4-5 ключевых)",\n
                        "concerns": "Главные экономические тревоги через запятую",\n
                        "key_phrases": "Дословные цитаты (2-3 реплики, разделенные '|')",\n
                        "communication_style": "Стилистика речи (2-3 характеристики)",\n
                        "economic_view": "Суть взглядов на экономику (1 предложение)",\n
                        "trust_in_institutions": "Float (0.0-1.0) доверие к гос.институтам",\n
                        "knowledge_level": "Оценка компетентности (формат: 'Уровень' ('пояснение'))",\n
                        "financial_behavior": "Паттерны поведения (2-3 ключевых)",\n
                        "policy_priority": "Главный экономический приоритет (1 пункт)",\n
                        "cb_functions_understanding": "Понимание функций ЦБ (3 пункта через запятую)",\n
                        "cb_perception": "Отношение к ЦБ (1 предложение)",\n
                        "cb_trust_factors": "Факторы доверия к ЦБ (2-3 пункта)",\n
                        "information_sources": "Источники инфо через запятую",\n
                        "emotional_tone": "Доминирующая эмоция (1-2 слова)"\n
                        }\n
                        **Критические инструкции:**\n
                        - Если данные отсутствуют: используй null (кроме текстовых полей - оставляй пустую строку).\n
                        - Для оценок (sentiment/trust): 0.8+ = явное доверие/позитив, 0.6-0.79 = умеренное, <0.6 = скепсис.\n
                        - knowledge_level: "Низкий", "Базовый", "Средний", "Средне-высокий", "Высокий".\n
                        - В key_phrases включай ТОЛЬКО дословные цитаты (макс. 7 слов).\n
                        - В cb_perception укажи когнитивную глубину (глубокое/поверхностное понимание).\n
                        - Учитывай контекстные противоречия (напр., "доверяю, но не разбираюсь" → умеренное доверие)."""

        headers = {
            "Authorization": f"Bearer {self.api_key_preprocessing}",
            "Content-Type": "application/json"
        }
        txt_files = [f for f in os.listdir(self.interviews_path) if f.endswith('.txt')]
        interviews = []
        if not txt_files:
            print("В папке нет .txt файлов!")
            return
        for file_name in txt_files:
            file_path = os.path.join(self.interviews_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                interviews.append(content)
        for interview in interviews:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": interview}]
            data = {
                "model": self.model_for_preprocessing,
                "messages": messages
            }
            try:
                response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                         headers=headers,
                                         json=data,
                                         timeout=30
                                         )
                response_json = response.json()
                content_str = response_json['choices'][0]['message']['content']
                avatar_data = json.loads(content_str)
                filename = f"{avatar_data['name']}.json"
                with open(os.path.join(self.avatars_path, filename), 'w') as f:
                    json.dump(avatar_data, f)
            except Exception as e:
                print(f"Ошибка API: {str(e)}")
                continue

    def create_agent(self, avatar_for_agent, model, tokenizer):
        """
        This function creates a single Agent instance based on config and model.
        """
        return Agent(avatar_for_agent, model, tokenizer)

    def get_group_number(self):
        """
        This method returns the total number of agents in the group.
        """
        return len(self.agents)

    def calibrate_all(self, calibration_path=None):
        """
        This method is used for agents' calibration on the small dataset of documents.
        """
        calibration_results = dict()
        if calibration_path is not None:
            for agent in self.agents:
                calibration_result = agent.calibrate(self.calibration_path)
                calibration_results[agent.name] = calibration_result
        return calibration_results

    def respond_all(self, prompt):
        """
        This method collects responses from all agents to a given prompt.
        """
        responses = []
        for agent in self.agents:
            responses.append((agent.name, agent.generate_response(prompt)))
        return responses

    def start_dialog(self):
        """
        This function interacts with the user, accepts questions, sends them to all agents,
        and prints & saves responses until user types 'stop'.
        """
        print(f"Your synthetic focus group consists of {len(self.agents)} respondents.")
        while True:
            user_input = input("Введите вопрос к фокус-группе (или 'stop' для выхода): ")
            if user_input.lower().strip() == "stop":
                print("Диалог завершён.")
                break
            if self.iterations == 1:
                responses = self.respond_all(user_input)
                with open(self.result_path, 'a', encoding='utf-8') as file:
                    file.write(f"Вопрос группе: {user_input}\n")
                    for response in responses:
                        file.write(f"{response[0]} ответил: {response[1]}\n")
            if self.iterations > 1:
                responses = self.respond_all(user_input)
                with open(self.result_path, 'a', encoding='utf-8') as file:
                    file.write(f"Вопрос группе: {user_input}\n")
                    summary = []
                    for response in responses:
                        file.write(f"{response[0]} ответил: {response[1]}\n")
                        summary.append(response[1])
                    for i in range(self.iterations - 1):
                        repeat_input = f"Ответь на вопрос: {user_input}. При формулировке ответа учти следующие мнения: {' '.join(summary)}"
                        file.write(f"Вопрос задан группе повторно: {repeat_input}\n")
                        new_responses = self.respond_all(repeat_input)
                        for response in new_responses:
                            file.write(f"{response[0]} ответил: {response[1]}\n")
                            summary.append(response[1])
                            if len(" ".join(summary)) > 32000:
                                summary = summary[
                                          self.get_group_number():]  # clean the summary if its size is larger than the model's context window


if __name__ == "__main__":
    MODEL = "Qwen/Qwen3-8B"
    API = "#########################################################################"
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    group = FocusGroup(use_auto_preprocessing=False, model_for_agent=model, tokenizer_for_agent=tokenizer, iterations=1)
    group.start_dialog()
