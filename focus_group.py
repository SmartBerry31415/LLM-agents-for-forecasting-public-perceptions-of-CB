from agent import Agent
from model import Model
import os
import json
import concurrent.futures

class FocusGroup:
    """
    This class creates & manages a group of agents and provides tools to interact with them as a unit.
    """
    def __init__(self, model_for_preprocessing: Model | None,
                 use_auto_preprocessing: bool = False,
                 interviews_path: None | str = "interviews",
                 avatars_path: str = "avatars",
                 iterations: int = 2):
        self.model_for_preprocessing = model_for_preprocessing
        self.use_auto_preprocessing = use_auto_preprocessing
        self.interviews_path = interviews_path
        self.avatars_path = avatars_path
        self.iterations = iterations
        self.agents = []
        if self.use_auto_preprocessing:
            self._conduct_auto_preprocessing()


    def _conduct_auto_preprocessing(self):
        """
            Creates avatars from given interviews (in a format of text files - .txt)
        """
        system_prompt = """Ты — профессиональный аналитик социологических и экономических исследований.\n
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
                                - Учитывай контекстные противоречия (напр., "доверяю, но не разбираюсь" → умеренное доверие). Интервью: """
        txt_files = [f for f in os.listdir(self.interviews_path) if f.endswith('.txt')]
        interviews = []
        if not txt_files:
            print("There are no .txt files in specified directory!")
            return
        for file_name in txt_files:
            file_path = os.path.join(self.interviews_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                interviews.append(content)
        for interview in interviews:
            response = self.model_for_preprocessing.generate_response(system_prompt, interview)
            avatar_data = json.loads(response)
            filename = f"{avatar_data['name']}.json"
            try:
                with open(os.path.join(self.avatars_path, filename), 'w', encoding='utf-8') as f:
                    json.dump(avatar_data, f)
            except Exception as e:
                print(f"Error occured while making an avatar: {str(e)}")
                continue


    def add_agents(self, model: Model):
        """
        This function creates Agent instances based on given model.
        """
        try:
            avatar_files = [f for f in os.listdir(self.avatars_path) if os.path.isfile(os.path.join(self.avatars_path, f))]
            avatars = []
            for avatar_file in avatar_files:
                with open(os.path.join(self.avatars_path, avatar_file), "r", encoding="utf-8") as file:
                    avatar = json.load(file)
                    avatars.append(avatar)
            for avatar in avatars:
                self.agents.append(Agent(avatar, model))
            return f"Successfully added {len(self.agents)} agents."
        except Exception as e:
            return f"Error occured while adding agents: {str(e)}."


    def calibrate_all(self, calibration_path=None, max_tries: int = 3) -> dict:
        """Calibrate all agents"""
        if not calibration_path:
            return None
        calibration_results = {}
        for agent in self.agents:
            try:
                agent_results = agent.calibrate(calibration_path, max_tries)
                calibration_results[agent.name] = {
                    "profile": {
                        "age": agent.age,
                        "profession": agent.profession,
                        "trust_level": agent.trust_in_institutions
                    },
                    "responses": agent_results
                }
            except Exception as e:
                calibration_results[agent.name] = {"error": str(e)}
        return calibration_results


    def respond_all(self, request):
        """
        This method collects responses from all agents to a given prompt.
        """
        iteration_number = 0
        all_responses = []
        iteration_context = request
        all_responses.append(("Модератор", iteration_context, iteration_number))
        iteration_responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_agent = {executor.submit(agent.generate_response, request): agent for agent in self.agents}
            for future in concurrent.futures.as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    response = future.result()
                    response_tuple = (agent.name, response, iteration_number)
                    iteration_responses.append(response_tuple)
                    all_responses.append(response_tuple)
                except Exception as e:
                    print(f"Error with agent {agent.name}: {e}")
                    error_tuple = (agent.name, f"Error: {str(e)}", iteration_number)
                    iteration_responses.append(error_tuple)
                    all_responses.append(error_tuple)
        for iteration in range(2, self.iterations + 1):
            prev_iteration_responses = [resp for (name, resp, it) in iteration_responses if it == iteration - 2]
            group_opinions = ". ".join([resp[:500] for resp in prev_iteration_responses])
            iteration_context = f"Iteration {iteration}"
            all_responses.append(("SYSTEM", iteration_context, iteration_number))
            iteration_responses = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                future_to_agent = {executor.submit(agent.generate_response, request, group_opinions): agent for agent in self.agents}
                for future in concurrent.futures.as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        response = future.result()
                        response_tuple = (agent.name, response, iteration_number)
                        iteration_responses.append(response_tuple)
                        all_responses.append(response_tuple)
                    except Exception as e:
                        print(f"Error with agent {agent.name}: {e}")
                        error_tuple = (agent.name, f"Error: {str(e)}", iteration_number)
                        iteration_responses.append(error_tuple)
                        all_responses.append(error_tuple)
        return all_responses


    def conduct_interview(self, text_file: str | None, log_file: str = "logs.txt"):
        """
        Interactive interview
        """
        print(f"Your synthetic focus group consists of {len(self.agents)} respondents.")
        print(f"The interview will be conducted with {self.iterations} iterations.")
        print("Interview starts now. To terminate the interview, enter END")
        if text_file:
            with open(text_file, "r", encoding='utf-8') as t_f:
                text = t_f.read()
        request = input("Enter your question for the focus group: ")
        with open(log_file, "a", encoding="utf-8") as l_f:
            while request != "END":
                print("Wait...")
                if text_file:
                    request += f"Текст, с которым ты работаешь: {text}"
                responses = self.respond_all(request)
                for (name, resp, _) in responses:
                    result = name + ": " + resp
                    print(result)
                    l_f.write(result + '\n')
                request = input("Enter your question for the focus group: ")
        print("Interview terminated. Results saved to:", log_file)


    def conduct_auto_interview(self, text_file: str | None, interview_file: str, log_file: str = "logs.txt"):
        """
        Automatic interview (put interview questions to a text file).
        """

        print(f"Your synthetic focus group consists of {len(self.agents)} respondents.")
        print(f"The interview will be conducted with {self.iterations} iterations automatically.")
        print("Interview starts now.")
        if text_file:
            with open(text_file, "r", encoding='utf-8') as t_f:
                text = t_f.read()
        with open(interview_file, "r", encoding='utf-8') as q_f:
            interview_questions = q_f.readlines()
        for request in interview_questions:
            if text_file:
                request += f"Текст, с которым ты работаешь: {text}"
            responses = self.respond_all(request)
            with open(log_file, "a", encoding="utf-8") as l_f:
                for (name, resp, _) in responses:
                    result = name + ": " + resp
                    print(result)
                    l_f.write(result + '\n')
        print("Interview terminated. Results saved to:", log_file)

