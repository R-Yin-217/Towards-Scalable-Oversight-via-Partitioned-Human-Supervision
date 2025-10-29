import argparse
import copy
import json
import time
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
import pandas
from tqdm import tqdm

from prompt import get_init_archive

client = openai.OpenAI()

from utils import format_multichoice_question, random_id, bootstrap_confidence_interval

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False



@backoff.on_exception(backoff.expo, (openai.RateLimitError,), max_tries=10)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.5, max_retries=10):
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg},
            ],
            temperature=temperature,
            max_completion_tokens=12000,  
            stop=None,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content

        if content == "":
            print(f"[Warning] Empty content on attempt {attempt+1}")
        else:
            try:
                json_dict = json.loads(content)
                return json_dict
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode failed on attempt {attempt+1}: {e}")
                print(f"[Content]:\n{content[:300]}...\n")  

        time.sleep(2)  

    raise ValueError("Failed to get valid JSON response from model after retries.")

class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gpt-5-nano', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        # construct system prompt
        if(args.data_filename == 'Dataset/EDINET_Bench_16.csv' or args.data_filename == 'Dataset/EDINET_Bench_Extended_16.csv'):
            options = "['金融(除く銀行)', '商社・卸売', '不動産', '素材・化学', '情報通信・サービスその他', '建設・資材', '電機・精密', '小売', '食品', '機械', '自動車・輸送機', '運輸・物流', '鉄鋼・非鉄', '銀行', '医薬品', '電気・ガス・エネルギー資源']"
            output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}.  You must classify the company into exactly one of these categories: {options}." for key in self.output_fields}
        elif(args.data_filename == 'Dataset/Medical_Abstract_5.csv'):
            options = "['general pathological conditions', 'neoplasms', 'digestive system diseases', 'nervous system diseases', 'cardiovascular diseases']"
            output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}.  You must classify the company into exactly one of these categories: {options}." for key in self.output_fields}
        else:
            raise ValueError("You got the wrong Dataset for prompt!")
            output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1, debug=False) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            if(debug):
                print(f"[Prompt]:\n{prompt}")
                print(f"[System Prompt]:\n{system_prompt}")
                print(f"[Success] Model raw JSON response:\n{response_json}")
                print("\n")
                exit()
                
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
            
        except Exception as e:
            if "maximum context length" in str(e):
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self) -> None:
        pass


def evaluate(args):
    # json
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_{args.model}.json")
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            output = json.load(json_file)
    else:
        output = []
        
    archive = get_init_archive(args.prompt_name)
 

    for solution in archive:
        print(f"============{solution['name']}=================")
        try:
            if(args.method == 'comp'):  
                comp_label = True
            else:
                comp_label = False
            acc_list = evaluate_forward_fn(args, solution["code"], CL = comp_label)
        except Exception as e:
            print(f"During evaluating solution {solution['name']}:")
            print(e)
            continue
        
        print(f"acc: {bootstrap_confidence_interval(acc_list, seed = args.shuffle_seed)}")

        fitness_str = bootstrap_confidence_interval(acc_list)

        solution['dataset'] = args.data_filename
        solution['seed'] = args.shuffle_seed
        solution['method'] = args.method
        solution['numbber of samples'] = args.valid_size
        solution['fitness'] = fitness_str
        if 'code' in solution:
            del solution['code']
        
        output.append(solution)
        
        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(output, json_file, indent=4)


def evaluate_forward_fn(args, forward_str, CL=False):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1") # namely forward
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func) # add func(forward) to AgentSystem defined 

    # get dataset 
    df = pandas.read_csv(args.data_filename)
    if(args.data_filename == 'Dataset/EDINET_Bench_16.csv' or args.data_filename == 'Dataset/EDINET_Bench_Extended_16.csv'):
        options = df['Industry'].unique().tolist()
    elif(args.data_filename == 'Dataset/Medical_Abstract_5.csv'):
        options = df['Label'].unique().tolist()
    else:
        raise ValueError("You got the wrong Dataset for dataset!")
    
    # get indices
    LETTER_TO_INDEX = {}
    for i in range(len(options)):
        LETTER_TO_INDEX[options[i]] = i
  
    # shuffle dataset
    random.seed(args.shuffle_seed)
    examples = [row.to_dict() for _, row in df.iterrows()]
    random.shuffle(examples)

    
    if(args.type == 'mix'):
        examples = examples[args.original_size:args.original_size + args.valid_size] * args.n_repreat
    else:
        examples = examples[:args.valid_size] * args.n_repreat

    
    # use the same dataset for ordinary and comp label
    questions = [format_multichoice_question(options, example, args.data_filename) for example in examples] # do not forget the number of examples

    if(CL):
        answers = [LETTER_TO_INDEX[example['Comp Label']] for example in examples]
    else:
        if(args.data_filename == 'Dataset/EDINET_Bench_16.csv' or args.data_filename == 'Dataset/EDINET_Bench_Extended_16.csv'):
            answers = [LETTER_TO_INDEX[example['Industry']] for example in examples]
        elif(args.data_filename == 'Dataset/Medical_Abstract_5.csv'):
            answers = [LETTER_TO_INDEX[example['Label']] for example in examples]
        else:
            raise ValueError("You got the wrong Dataset for answers!")
    
    
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    task_queue = []
    for q in questions:
        taskInfo = Info('task', 'User', q, -1)
        task_queue.append(taskInfo)
    agentSystem = AgentSystem()

    acc_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))


    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, str):
                answer_str = res
            elif isinstance(res, list):
                answer_info = next((x for x in res if x.name == "answer"), None)
                answer_str = answer_info.content if answer_info else ""
            elif hasattr(res, "content"):
                answer_str = res.content
            else:
                answer_str = ""


            answer_str = answer_str.strip()
            if len(answer_str) > 1 and answer_str[1] == ')':
                answer_str = answer_str[0]

            if answer_str in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[answer_str]
            else:
                print(f"[Warning] Unrecognized answer format at q{q_idx}: {answer_str}")
                acc_list.append(0)
                continue

        except Exception as e:
            print(f"[Error] Exception at q{q_idx}: {e}")
            acc_list.append(0)
            continue


        if(CL):
            if predicted_idx != answers[q_idx]:
                acc_list.append(1)
            else:
                acc_list.append(0)
        else:
            if predicted_idx == answers[q_idx]:
                acc_list.append(1)
            else:
                acc_list.append(0)    
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # save and seed
    parser.add_argument('--save_dir', type=str, default='Results/')
    parser.add_argument('--expr_name', type=str, default="Exp_2")
    parser.add_argument('--shuffle_seed', type=int, default=1998)
    parser.add_argument('--n_repreat', type=int, default=1)
    
    # model
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--model',
                        type=str,
                        default='gpt-5-nano',
                        choices=['gpt-4.1-nano', 'gpt-4o-2024-05-13', 'gpt-5-nano'])
    parser.add_argument('--method', type=str, 
                        default='original',
                        choices=['original', 'comp'])
    parser.add_argument('--type', type=str,
                        default='original',
                        choices=['original', 'mix'])
    parser.add_argument('--prompt_name', type=str,
                        default='basic',
                        choices=['basic', 'COT'])
    
    # data
    parser.add_argument('--data_filename', type=str, 
                        default='Dataset/Medical_Abstract_5.csv',
                        choices=['Dataset/EDINET_Bench_16.csv', 'Dataset/Medical_Abstract_5.csv', 'Dataset/EDINET_Bench_Extended_16.csv'])
    parser.add_argument('--num_multiple', type=int, default=5)

    # validation size
    parser.add_argument('--original_size', type=int, default=0)
    parser.add_argument('--valid_size', type=int, default=1)
    
    # other defult value
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)

    args = parser.parse_args()
    
    evaluate(args)
