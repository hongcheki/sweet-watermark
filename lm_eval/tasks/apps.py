"""Measuring Coding Challenge Competence With APPS
https://arxiv.org/abs/2105.09938

APPS is a benchmark for code generation with 10000 problems. With three difficulty levels: introductory, interview and competition.
It can be used to evaluate the ability of language models to generate code from natural language specifications.

Homepage: https://github.com/hendrycks/apps
"""

import json
import random

from evaluate import load

from lm_eval.base import Task

_CITATION = """
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""


LEVELS = ["introductory", "interview", "competition"]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {apps-interview: Task, apps-competitoon: Task}
    """
    return {f"apps-{level}": create_task(level) for level in LEVELS}


def create_task(level):
    class APPS(GeneralAPPS):
        def __init__(self):
            super().__init__(level)

    return APPS


class GeneralAPPS(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "codeparrot/apps"
    DATASET_NAME = None

    def __init__(self, level):
        self.DATASET_NAME = level
        super().__init__(
            stop_words=["\nQUESTION", "\n---", "\nANSWER"],
            requires_execution=True,
        )
        self.few_shot = 2
        self.few_shot_pool = self._get_few_shot_pool()

    def _check_type(self, doc):
        try:
            input_outpout = json.loads(doc["input_output"])
            fn_name = (
                None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
            )
        except ValueError:
            fn_name = None
        
        if not fn_name:
            return "standard"
        else:
            return "call_based"

    def _get_few_shot_pool(self):
        pool = {
            'standard': [],
            'call_based': [],
        }

        # only consider samples where solutions exist
        for doc in self.dataset['train']:
            try:
                sols = json.loads(doc['solutions'])
            except:
                continue

            assert isinstance(sols, list)
        
            prompt = self._get_prompt(doc)
            prompt_with_sol = [
                prompt + sol
                for sol in sols
            ]

            pool[self._check_type(doc)].append(prompt_with_sol)

        print(
            "size of few-shot demonstration pool : "
            f"{len(pool['standard'])} (standard) "
            f"{len(pool['call_based'])} (call_based)"
        )
        print(f"and we do {self.few_shot}-shot")

        return pool

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def _get_prompt(self, doc):
        """Generate prompts for APPS
        Finetuning setup: prompt=question  with some starter code and function name if they exist.
        We also specify the type of the prompt, i.e. whether it is call-based or standard input-based.
        """
        starter_code = None if len(doc["starter_code"]) == 0 else doc["starter_code"]
        try:
            input_outpout = json.loads(doc["input_output"])
            fn_name = (
                None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
            )
        except ValueError:
            fn_name = None
        prompt = "\nQUESTION:\n"
        prompt += doc["question"]
        if starter_code:
            prompt += starter_code
        if not fn_name:
            call_format = "\nUse Standard Input format"
            prompt += call_format
        else:
            call_format = "\nUse Call-Based format"
            prompt += call_format
        prompt += "\nANSWER:\n"
        return prompt
    
    def get_prompt(self, doc):
        if self.few_shot > 0:
            pool = self.few_shot_pool[self._check_type(doc)]
            assert len(pool) >= self.few_shot

            prompt = "Implement answers to the following questions:"

            l1 = list(range(len(pool)))
            random.shuffle(l1)
            for i in l1[:self.few_shot]:
                sol_pool = pool[i]

                l2 = list(range(len(sol_pool)))
                random.shuffle(l2)

                prompt += sol_pool[l2[0]]

            prompt += self._get_prompt(doc)
        else:
            prompt = self._get_prompt(doc)

        return prompt

    def get_solutions(self, doc):
        # list of solutions which is str
        try:
            sols = json.loads(doc['solutions'])[0]
        except:
            #print("solution load errors :", doc['solutions'])
            sols = doc['solutions']
        return sols

    def get_full_data(self, doc):
        sol = self.get_solutions(doc)
        if sol:
            return self.get_prompt(doc) + sol
        else:
            return None

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return None

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for APPS)
        """
        #try:
        #    generation = generation.split("\nANSWER:", 1)[1]
        #except IndexError:
        #    # happens when prompts were very long and got truncated
        #    pass
        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences (not needed for APPS Task)
        """
        code_metric = load("codeparrot/apps_metric")
        results = code_metric.compute(
            predictions=generations, k_list=[1, 10, 100], level=self.DATASET_NAME
        )
        return results
