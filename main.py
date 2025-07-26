from transformers import AutoModelForCausalLM, AutoTokenizer
from os import listdir, makedirs
from os.path import isfile, join, exists
import json

MODEL = "Qwen/Qwen3-8B"


class Agent:
    """
    This class represents a single focus group participant with personality parameters
    and language model access to generate personalized responses.
    """

    def __init__(self, config, model, tokenizer):
        self.name = config.get("name", "")
        self.age = config.get("age", None)
        self.profession = config.get("profession", "")
        self.sentiment = config.get("sentiment", 0.5)
        self.traits = config.get("traits", "")
        self.concerns = config.get("concerns", "")
        self.key_phrases = config.get("key_phrases", "")
        self.communication_style = config.get("communication_style", "")
        self.economic_view = config.get("economic_view", "")
        self.trust_in_institutions = config.get("trust_in_institutions", 0.5)
        self.knowledge_level = config.get("knowledge_level", "")
        self.financial_behavior = config.get("financial_behavior", "")
        self.policy_priority = config.get("policy_priority", "")
        self.cb_functions_understanding = config.get("cb_functions_understanding", "")
        self.cb_perception = config.get("cb_perception", "")
        self.cb_trust_factors = config.get("cb_trust_factors", "")
        self.information_sources = config.get("information_sources", "")
        self.emotional_tone = config.get("emotional_tone", "")
        self.model = model
        self.tokenizer = tokenizer

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
        )
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=200, temperature=self.sentiment)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def create_agent(config, model, tokenizer):
    """
    This function creates a single Agent instance based on config and model.
    """
    return Agent(config, model, tokenizer)


class FocusGroup:
    """
    This class manages a group of agents and provides tools to interact with them as a unit.
    """

    def __init__(self, configs, model, tokenizer):
        """
        This method initializes all agents in the focus group from a list of config dicts.
        """
        self.agents = []
        for i, config in enumerate(configs):
            agent = create_agent(config, model, tokenizer)
            self.agents.append(agent)

    def get_agent(self, index):
        """
        This method retrieves an agent by index from the group.
        """
        if 0 <= index < len(self.agents):
            return self.agents[index]
        raise IndexError("Агент с таким индексом не найден.")

    def respond_all(self, prompt):
        """
        This method collects responses from all agents to a given prompt.
        """
        responses = []
        for agent in self.agents:
            responses.append(agent.generate_response(prompt))
        return responses

    def __len__(self):
        """
        This method returns the total number of agents in the group.
        """
        return len(self.agents)


def dialog(focus_group):
    """
    This function interacts with the user, accepts questions, sends them to all agents,
    and prints & saves responses until user types 'stop'.
    """
    folder = "results"
    if not exists(folder):
        makedirs(folder)

    file_path = join(folder, "interview.txt")

    with open(file_path, "w", encoding="utf-8") as file:
        while True:
            user_input = input("Введите вопрос к фокус-группе (или 'stop' для выхода): ")
            if user_input.lower().strip() == "stop":
                print("Диалог завершён.")
                break

            print(f"\nQuestion: {user_input}")
            file.write(f"Question: {user_input}\n")

            for agent in focus_group.agents:
                try:
                    response = agent.generate_response(user_input)
                except Exception as e:
                    response = f"⚠️ Ошибка: {str(e)}"
                formatted = f"{agent.name.capitalize()}: {response}"
                print(formatted)
                file.write(formatted + "\n")

            file.write("\n\n")
            print("\n\n")


def load_configs_from_folder(folder="params"):
    """
    This function reads JSON configuration files from the given folder
    and returns a list of config dictionaries.
    """
    param_files = [f for f in listdir(folder) if isfile(join(folder, f))]
    configs = []
    for param_file in param_files:
        path = join(folder, param_file)
        with open(path, "r", encoding="utf-8") as file:
            config = json.load(file)
            configs.append(config)
    return configs


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    configs = load_configs_from_folder("params")
    group = FocusGroup(configs, model, tokenizer)
    dialog(group)
