from model import Model
import os
import re



class Agent:
    """
    This class represents a single focus group participant with personality parameters
    and language model access to generate personalized responses.
    """
    def __init__(self, avatar_for_agent: dict[str, str], model: Model):
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
        self.system_prompt = self._generate_system_prompt()
        self.model = model
        self.response_history = []
        self.amenable_to_influence = self._calculate_amenability(self.trust_in_institutions)

    def _generate_system_prompt(self):
        """
        Generates system prompt
        """
        system_prompt = (
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
        )
        return system_prompt

    def _calculate_amenability(self, trust: float) -> float:
        """
        Calculate influence amenability based on trust_in_institutions
        - Close to 0 or 1: low influence (0)
        - 0.3-0.5 range: high influence (1)
        """
        return trust / 0.6 if trust < 0.3 else 1.0 if trust <= 0.5 else (1.0 - trust)


    def generate_response(self, request: str, group_opinions: str | None = None):
        """
        Obtains response from a given model
        """
        request_prompt = f"Вопрос, на который ты отвечаешь: {request}"
        if group_opinions:
            influence_context = (
                f"\nКонтекст обсуждения: Другие участники выразили мнения: {group_opinions}. "
                f"Твоя восприимчивость к мнениям других: {self.amenable_to_influence} "
                f"(0 - невосприимчив, 1 - очень восприимчив). Учти это при формировании ответа."
            )
            request_prompt += influence_context
        response_content = self.model.generate_response(self.system_prompt, request_prompt)
        response_content = re.sub(r'\n', ' ', response_content)
        self.response_history.append(response_content)
        return response_content


    def calibrate(self, calibration_path: str, max_tries: int = 3) -> dict:
        """
        Calibrate agent using documents in specified directory
        """
        VALID_RESPONSES = ['негативно', 'нейтрально', 'позитивно']
        if not os.path.exists(calibration_path):
            raise FileNotFoundError(f"Calibration directory not found: {calibration_path}")
        calibration_results = []
        file_list = sorted(os.listdir(calibration_path))
        for i, filename in enumerate(file_list):
            file_path = os.path.join(calibration_path, filename)
            if not os.path.isfile(file_path) or not filename.endswith(".txt"):
                continue
            with open(file_path, 'r', encoding='utf-8') as file:
                document = file.read()
                user_prompt = (
                    "Оцени этот текст с учетом следующих факторов:\n"
                    "Обоснованность принимаемого решения\n"
                    "Полнота объяснений\n"
                    "Доступность языка документа\n"
                    "Способность документа вызвать доверие\n\n"
                    "ТВОЙ ОТВЕТ ДОЛЖЕН БЫТЬ ТОЛЬКО ОДНИМ СЛОВОМ ИЗ СПИСКА:\n"
                    "['негативно', 'нейтрально', 'позитивно']\n\n"
                    "НЕ ДОБАВЛЯЙ НИКАКИХ ДРУГИХ СЛОВ, КОММЕНТАРИЕВ ИЛИ ЗНАКОВ ПРЕПИНАНИЯ.\n"
                    f"Текст документа:\n{document}"
                )
                for _ in range(max_tries):
                    response = self.model.generate_response(self.system_prompt, user_prompt)
                    clean_response = None if response is None else response.strip().lower()
                    if clean_response in VALID_RESPONSES:
                        break
                calibration_results.append({
                    "document": filename,
                    "response": response.strip().lower()
                })
        return calibration_results