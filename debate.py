import json
import argparse
import tqdm
import time
import logger
import pandas as pd
import autogen
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import autogen
from autogen import AssistantAgent, Agent, UserProxyAgent, ConversableAgent

class MultiAgentsDebate(AssistantAgent):
    def __init__(self, n_iters=2, **kwargs):
        super().__init__(**kwargs)
        self.register_reply([Agent,None],
                            reply_func=MultiAgentsDebate._reply_user,
                            position=0)
        self._n_iters = n_iters
    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        if messages is None:
            messages  = self._oai_messages[sender]
        user_question = messages[-1]['content']
        commander = AssistantAgent(
            name="Commander",
            max_consecutive_auto_reply=1,
            system_message="Your role is to initiate and lead the debate. Tell other agents think step-by-step.",
            is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config
        )
        score_agent = AssistantAgent(
            name="Scoring assistant",
            max_consecutive_auto_reply=1,
            system_message="Your role is to score a given text. Read the instructions and the text carefully. Then evaluate the text logically.",
            is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config
        )
        critics = AssistantAgent(
            name="Critics",
            system_message="""
            Your role is to play a Devil’s Advocate. Your logic has to be step-by-step. Critically review the score provided and assess whether the score is accurate. If you don’t think that the score is accurate, criticize the score. Try to criticize the score as much as possible.
            """,
            is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config
        )
        commander.initiate_chat(score_agent, message=user_question)
        time.sleep(2.5)
        final_score = None

        for _ in range(self._n_iters):
            commander.send(message="Check if the score is justified. Task description and the following source texts are as follows: " \
                                    + '\n' + user_question + '\n And the responses from Scoring assistant as follows: ' + '\n' \
                                    + commander._oai_messages[score_agent][1]['content'],
                           recipient=critics,
                           request_reply=True)
            time.sleep(2.5)
            
            feedback = commander._oai_messages[critics][-1]["content"]
            if feedback.find("NO_ISSUE") >= 0:
                break
            commander.send(
                message="Here is the feedback to your response. Please calculate the score again!\n"
                + feedback,
                recipient=score_agent,
                request_reply=True)
            time.sleep(2.5)
            
        final_score = score_agent._oai_messages[commander][-2]['content']
        return True, final_score
    
def set_config(model, key):
    config_list = [
        {
            
            'model' : model,
            'api_key' : key,

        }
    ]

    llm_config = {
        'timeout':600,
        'config_list' : config_list,
        'temperature' : 0,
        'seed': 453,
    }
    
    return llm_config

def set_userproxyAgent():
    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0
    )
    return user_proxy

import re

def normalize_string(s):
    s = ' '.join(s.split())
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    return s.lower().strip()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_fp', type=str, default='prompts/summeval/coh_detailed.txt')
    argparser.add_argument('--aspect', type=str, default='coherence')
    argparser.add_argument('--save_fp', type=str, default='results/gpt4_coh_detailed_openai.json')
    argparser.add_argument('--summeval_fp', type=str, default='data/summeval.json')
    argparser.add_argument('--key', type=str, default='sk-')
    argparser.add_argument('--model', type=str, default='gpt-4-1106-preview')

    args = argparser.parse_args()
    
    summeval = json.load(open(args.summeval_fp))
    
    data = pd.DataFrame(summeval, columns=summeval[0].keys())
    
    prompt = open(args.prompt_fp).read()
    aspect = args.aspect
    
    
    llm_config =set_config(args.model, args.key)
    
    user_proxy = set_userproxyAgent()
    multiAgents = MultiAgentsDebate(
            name="Calculating score through debate",
            llm_config=llm_config
    )
    
    results = []
    ignore = 0
    for idx in tqdm.tqdm(range(data.shape[0])):
        instance_dict = {}
        sampled_idx = data.iloc[idx]
        instance_dict['doc_id'] = sampled_idx['doc_id']
        instance_dict['source'] = sampled_idx['source']
        instance_dict['system_output'] = sampled_idx['system_output']
        cur_prompt = prompt.replace('{{Document}}', sampled_idx['source']).replace('{{Summary}}', sampled_idx['system_output'])
        instance_dict['human_score'] = sampled_idx['scores'][aspect.lower()]
        try:
            user_proxy.initiate_chat(multiAgents, message=cur_prompt)
            score = user_proxy._oai_messages[multiAgents][-1]['content']
            instance_dict[aspect.lower()] = score
            time.sleep(2.5)
            
            results.append(instance_dict)
            print('-'*50 + 'added result' + '-'*50)
            
            print('CLEAR!!!!')
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(3)
            else:
                ignore += 1
                print('ignored', ignore)
                break
    print('ignored total', ignore)
    with open(args.save_fp, 'w') as f:
        json.dump(results, f, indent=4)    