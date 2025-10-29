import json

Basic = {
    "name": "Basic",
    "code": """def forward(self, taskInfo):
    # Instruction for the Basic approach — just answer without thinking
    basic_instruction = ""

    if(args.model == 'gpt-5-nano'):
        args.temperature = 1.0

    # Only output 'answer', no 'thinking'
    basic_agent = LLMAgentBase(['answer'], 'Basic Agent', model=args.model, temperature=args.temperature)

    # Prepare the inputs for the agent
    basic_agent_inputs = [taskInfo]

    # Get the response from the agent
    (answer,) = basic_agent(basic_agent_inputs, basic_instruction)

    # Return only the final answer
    return answer
"""
}

COT = {
    "name": "Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    # Instruction for the Chain-of-Thought (CoT) approach
    # It is an important practice that allows the LLM to think step by step before solving the task.
    cot_instruction = "Please think step by step and then solve the task."

    # Instantiate a new LLM agent specifically for CoT
    # To allow LLM thinking before answering, we need to set an additional output field 'thinking'.

    if(args.model == 'gpt-5-nano'):
        args.temperature = 1.0
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=args.model, temperature=args.temperature)

    # Prepare the inputs for the CoT agent
    # The input should be a list of Info, and the first one is often the taskInfo
    cot_agent_inputs = [taskInfo]

    # Get the response from the CoT agent
    thinking, answer = cot_agent(cot_agent_inputs, cot_instruction)

    # Return only the final answer
    return answer
"""
}

COT_SC = {
    "name": "Self-Consistency with Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    # Instruction for step-by-step reasoning
    cot_instruction = "Please think step by step and then solve the task."
    N = 5 # Number of CoT agents

    # Initialize multiple CoT agents with a higher temperature for varied reasoning
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]

    # Majority voting function to select the most common answer
    from collections import Counter
    def majority_voting(answers):
        return Counter(answers).most_common(1)[0][0]
    
    possible_answers = []
    for i in range(N):
        thinking, answer = cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.append(answer.content)

    # Ensembling the answers from multiple CoT agents
    answer = majority_voting(possible_answers)
    return answer  
"""
          }

Reflexion = {
    "name": "Self-Refine (Reflexion)",
    "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning
    cot_initial_instruction = "Please think step by step and then solve the task."

    # Instruction for reflecting on previous attempts and feedback to improve
    cot_reflect_instruction = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # Instruction for providing feedback and correcting the answer
    critic_instruction = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output 'True' in 'correct'."
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent')
    
    N_max = 5 # Maximum number of attempts

    # Initial attempt
    cot_inputs = [taskInfo]
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        # Get feedback and correct status from the critic
        feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
        if correct.content == 'True':
            break
            
        # Add feedback to the inputs for the next iteration
        cot_inputs.extend([thinking, answer, feedback])

        # Reflect on previous attempts and refine the answer
        thinking, answer = cot_agent(cot_inputs, cot_reflect_instruction, i + 1)
    return answer
"""
}

LLM_debate = {
    "name": "LLM Debate",
    "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning
    debate_initial_instruction = "Please think step by step and then solve the task."

    # Instruction for debating and updating the solution based on other agents' solutions
    debate_instruction = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    
    # Initialize debate agents with different roles and a moderate temperature for varied reasoning
    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', temperature=0.8, role=role) for role in ['Biology Expert', 'Physics Expert', 'Chemistry Expert', 'Science Generalist']]

    # Instruction for final decision-making based on all debates and solutions
    final_decision_instruction = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)

    max_round = 2 # Maximum number of debate rounds
    all_thinking = [[] for _ in range(max_round)]
    all_answer = [[] for _ in range(max_round)]

    # Perform debate rounds
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, answer = debate_agents[i]([taskInfo], debate_initial_instruction)
            else:
                input_infos = [taskInfo] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                thinking, answer = debate_agents[i](input_infos, debate_instruction)
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    
    # Make the final decision based on all debate results and solutions
    thinking, answer = final_decision_agent([taskInfo] + all_thinking[max_round-1] + all_answer[max_round-1], final_decision_instruction)
    return answer
"""
}

Take_a_step_back = {
    "name": "Step-back Abstraction",
    "code": """def forward(self, taskInfo):
        # Instruction for understanding the principles involved in the task
        principle_instruction = "What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
        
        # Instruction for solving the task based on the principles
        cot_instruction = "Given the question and the involved principle behind the question, think step by step and then solve the task."
        
        # Instantiate LLM agents
        principle_agent = LLMAgentBase(['thinking', 'principle'], 'Principle Agent')
        cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')
        
        # Get the principles involved in the task
        thinking, principle = principle_agent([taskInfo], principle_instruction)

        # Use the principles to solve the task
        thinking, answer = cot_agent([taskInfo, thinking, principle], cot_instruction)
        return answer
"""
                    }

QD = {
    "name": "Quality-Diversity",
     "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning
    cot_initial_instruction = "Please think step by step and then solve the task."

    # Instruction for giving diverse answers
    qd_instruction = "Given previous attempts, try to come up with another interesting way to solve the task."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # Instruction for final decision-making based on collected reasoning and answers
    final_decision_instruction = "Given all the above solutions, reason over them carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    
    N_max = 3 # Maximum number of attempts

    # Initial attempt
    cot_inputs = [taskInfo]
    possible_answers = []
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    # Add the answer to the list of possible answers
    possible_answers.extend([thinking, answer])

    for i in range(N_max):
        # Reflect on previous attempts and generate another interesting answer
        cot_inputs.extend([thinking, answer])

        # Generate another interesting answer
        thinking, answer = cot_agent(cot_inputs, qd_instruction, i + 1)
        possible_answers.extend([thinking, answer])

    # Make the final decision based on all generated answers
    thinking, answer = final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    return answer
"""
      }

Role_Assignment = {
    "name": "Dynamic Assignment of Roles",
    "code": """def forward(self, taskInfo):
        # Instruction for step-by-step reasoning
        cot_instruction = "Please think step by step and then solve the task."
        expert_agents = [LLMAgentBase(['thinking', 'answer'], 'Expert Agent', role=role) for role in ['Physics Expert', 'Chemistry Expert', 'Biology Expert', 'Science Generalist']]

        # Instruction for routing the task to the appropriate expert
        routing_instruction = "Given the task, please choose an Expert to answer the question. Choose from: Physics, Chemistry, Biology Expert, or Science Generalist."
        routing_agent = LLMAgentBase(['choice'], 'Routing agent')

        # Get the choice of expert to route the task
        choice = routing_agent([taskInfo], routing_instruction)[0]

        if 'physics' in choice.content.lower():
            expert_id = 0
        elif 'chemistry' in choice.content.lower():
            expert_id = 1
        elif 'biology' in choice.content.lower():
            expert_id = 2
        else:
            expert_id = 3 # Default to Science Generalist

        thinking, answer = expert_agents[expert_id]([taskInfo], cot_instruction)
        return answer
"""
                   }

def get_init_archive(prompt_name):
    if(prompt_name == 'COT'):
        return [COT]
    else:
        return [Basic]
    
    



