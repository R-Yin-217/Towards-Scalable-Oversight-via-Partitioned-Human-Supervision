import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
import pandas
from tqdm import tqdm

client = openai.OpenAI()

from utils import format_multichoice_question, random_id, bootstrap_confidence_interval

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, max_tokens=8000, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8
):
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature, max_tokens=8000, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict

def pick_random_other(row):
    correct = row['Answer']
    other_choices = [c for c in order if c != correct]
    return random.choice(other_choices)


class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gpt-4.1-nano', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        # construct system prompt

        if(args.data_filename == 'Dataset/GPQA_Extended_4.csv'):
            output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A, B, C or D." for key in self.output_fields}
        elif(args.data_filename == 'Dataset/MATH_MC_5.csv'):
            output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A, B, C, D, or E." for key in self.output_fields}
        elif(args.data_filename == 'Dataset/Medical_Abstract_5.csv'):
            options = "['general pathological conditions', 'neoplasms', 'digestive system diseases', 'nervous system diseases', 'cardiovascular diseases']"
            output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}.  You must classify the company into exactly one of these categories: {options}." for key in self.output_fields}
        else:
            raise ValueError("You got the wrong dataset for prompt!")
        
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
            # print(e)
            if "maximum context length" in str(e) and SEARCHING_MODE:
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


def search(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0
    
    if(args.method == 'comp'):  
        comp_label = True
    else:
        comp_label = False

    for solution in archive:
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"],CL = comp_label)
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue

        fitness_str = bootstrap_confidence_interval(acc_list, seed=args.shuffle_seed, CL = comp_label, K = args.num_multiple)
        solution['fitness'] = fitness_str

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)

            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            # Reflexion 1
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            # Reflexion 2
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            n -= 1
            continue

        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"], CL = comp_label)
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During evaluation:")
                print(e)
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
                continue
        if not acc_list:
            n -= 1
            continue

        fitness_str = bootstrap_confidence_interval(acc_list,  seed=args.shuffle_seed, CL = comp_label, K = args.num_multiple)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1

        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)


def evaluate(args):
    comp_label = False # use the correct label to evaluate
        
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    current_idx = 0
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if current_idx < len(eval_archive):
            current_idx += 1
            continue
        sol = archive[current_idx]
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        current_idx += 1
        try:
            acc_list = evaluate_forward_fn(args, sol["code"], CL = comp_label)
        except Exception as e:
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list,  seed=args.shuffle_seed, CL = comp_label, K = args.num_multiple)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)
            

def evaluate_forward_fn(args, forward_str, CL):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    # get dataset
    df = pandas.read_csv(args.data_filename)
    LETTER_TO_INDEX = {}
    if(args.data_filename in ['Dataset/GPQA_Extended_4.csv', 'Dataset/MATH_MC_5.csv']):
        # for multiple version
        for i in range(args.num_multiple):
            LETTER_TO_INDEX[chr(ord('A') + i)] = i
        options = None
    elif(args.data_filename == 'Dataset/Medical_Abstract_5.csv'):
        options = df['Label'].unique().tolist()
        for i in range(len(options)):
            LETTER_TO_INDEX[options[i]] = i
    else:
        ValueError("You got the wrong Dataset for options!")
        
    # set seed 0 for valid set
    random.seed(args.shuffle_seed)
    examples = [row.to_dict() for _, row in df.iterrows()]
    random.shuffle(examples)

    if SEARCHING_MODE:
        examples = examples[:args.valid_size] * args.n_repreat
    else:
        examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat

    questions = [format_multichoice_question(options, args.num_multiple, example, args.data_filename) for example in examples]

    if(CL):
        answers = [LETTER_TO_INDEX[example['Comp Label']] for example in examples]
    else:
        if(args.data_filename in ['Dataset/GPQA_Extended_4.csv', 'Dataset/MATH_MC_5.csv']):
            answers = [LETTER_TO_INDEX[example['Answer']] for example in examples]
        elif(args.data_filename == 'Dataset/Medical_Abstract_5.csv'):
            answers = [LETTER_TO_INDEX[example['Label']] for example in examples]
        else:
            raise ValueError("You got the wrong Dataset for answers!")
        
    print(f"problem length: {len(examples)}")
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
                
            if(args.data_filename in ['Dataset/GPQA_Extended_4.csv', 'Dataset/MATH_MC_5.csv']):
                answer_str = answer_str.upper()
            elif(args.data_filename == 'Dataset/Medical_Abstract_5.csv'):
                answer_str = answer_str.strip()
            else:
                raise ValueError("You have wrong datafile for check the answers!")

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
    parser.add_argument('--expr_name', type=str, default="Exp_3_test")
    parser.add_argument('--shuffle_seed', type=int, default=1998)
    parser.add_argument('--n_repreat', type=int, default=1)
    
    # model
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4.1-nano',
                        choices=['gpt-4.1-nano', 'gpt-4o-2024-05-13', 'gpt-5-nano'])
    parser.add_argument('--method', type=str, 
                        default='comp',
                        choices=['original', 'comp'])
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    
    # data
    parser.add_argument('--data_filename', type=str, 
                        default='Dataset/MATH_MC_5.csv',
                        choices=[ 'Dataset/GPQA_Extended_4.csv', 'Dataset/MATH_MC_5.csv', 'Dataset/Medical_Abstract_5.csv'])
    parser.add_argument('--num_multiple', type=int, default=5,
                        choices=[4, 5])

    # validation size
    parser.add_argument('--valid_size', type=int, default=128,
                        choices=[128, 1152, 88])
    parser.add_argument('--test_size', type=int, default=800,
                        choices=[800, 1000, 458])
    
    # other defult value
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    
    args = parser.parse_args()
    

    if(args.data_filename == 'Dataset/GPQA_Extended_4.csv'):
        from gpqa_prompt import get_init_archive, get_prompt, get_reflexion_prompt
    elif(args.data_filename == 'Dataset/MATH_MC_5.csv'):
        from math_prompt import get_init_archive, get_prompt, get_reflexion_prompt
    elif(args.data_filename == 'Dataset/Medical_Abstract_5.csv'):
        from med_prompt import get_init_archive, get_prompt, get_reflexion_prompt
    else:
        raise ValueError("You got the wrong dataset for utils!")
    
    # search
    SEARCHING_MODE = True
    search(args)

    # evaluate
    SEARCHING_MODE = False
    evaluate(args)