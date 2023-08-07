import sys
import json
import os
import logging
from pprint import pprint
from tqdm import tqdm
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
from OpenAttack.utils import visualizer, result_visualizer, get_language, language_by_name
from OpenAttack.text_process.tokenizer import Tokenizer, get_default_tokenizer
from OpenAttack.victim.base import Victim
from OpenAttack.attackers.base import Attacker
from qa.metric import AttackMetric, MetricSelector
from qa.attack_eval.utils import worker_process, worker_init, attack_process
from qa.tags import *

import multiprocessing as mp

logger = logging.getLogger(__name__)

class AttackEval:
    def __init__(self,
        attacker : Attacker,
        victim : Victim,
        language : Optional[str] = None,
        tokenizer : Optional[Tokenizer] = None,
        invoke_limit : Optional[int] = None,
        metrics : List[Union[AttackMetric, MetricSelector]] = []
    ):
        """
        `AttackEval` is a class used to evaluate attack metrics.

        Args:
            attacker: An attacker, must be an instance of :py:class:`.Attacker`.
            victim: A victim model, must be an instance of :py:class:`.Vicitm`.
            language: The language used for the evaluation. If is `None` then `AttackEval` will intelligently select the language based on other parameters.
            tokenizer: A tokenizer used for visualization.
            invoke_limit: Limit on the number of model invokes.
            metrics: A list of metrics. Each element must be an instance of :py:class:`.AttackMetric` or :py:class:`.MetricSelector` .
        """

        if language is None:
            lst = [attacker]
            if tokenizer is not None:
                lst.append(tokenizer)
            if victim.supported_language is not None:
                lst.append(victim)
            for it in metrics:
                if isinstance(it, AttackMetric):
                    lst.append(it)

            lang_tag = get_language(lst)
        else:
            lang_tag = language_by_name(language)
            if lang_tag is None:
                raise ValueError("Unsupported language `%s` in attack eval" % language)

        self._tags = { lang_tag }

        if tokenizer is None:
            self.tokenizer = get_default_tokenizer(lang_tag)
        else:
            self.tokenizer = tokenizer

        self.attacker = attacker
        self.victim = victim
        self.metrics = []
        for it in metrics:
            if isinstance(it, MetricSelector):
                v = it.select(lang_tag)
                if v is None:
                    raise RuntimeError("`%s` does not support language %s" % (it.__class__.__name__, lang_tag.name))
                self.metrics.append( v )
            elif isinstance(it, AttackMetric):
                self.metrics.append( it )
            else:
                raise TypeError("`metrics` got %s, expect `MetricSelector` or `AttackMetric`" % it.__class__.__name__)
        self.invoke_limit = invoke_limit
        
    @property
    def TAGS(self):
        return self._tags
    
    def __measure(self, data, adversarial_sample):
        ret = {}
        for it in self.metrics:
            value = it.after_attack(data, adversarial_sample)
            if value is not None:
                ret[it.name] = value
        return ret

    def __iter_dataset(self, dataset):
        for data in dataset:
            v = data
            for it in self.metrics:
                ret = it.before_attack(v)
                if ret is not None:
                    v = ret
            yield v
    
    def __iter_metrics(self, iterable_result):
        for data, result in iterable_result:
            adversarial_sample, attack_time, invoke_times = result

            ret = {
                "data": data,
                "success": adversarial_sample is not None,
                "result": adversarial_sample,
                "metrics": {
                    "Running Time": attack_time,
                    "Query Exceeded": self.invoke_limit is not None and invoke_times > self.invoke_limit,
                    "Victim Model Queries": invoke_times,
                    ** self.__measure(data, adversarial_sample)
                }
            }
            yield ret

    def ieval(self, dataset : Iterable[Dict[str, Any]], num_workers : int = 0, chunk_size : Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Iterable evaluation function of `AttackEval` returns an Iterator of result.

        Args:
            dataset: An iterable dataset.
            num_worers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.
        
        Yields:
            A dict contains the result of each input samples.

        """

        if num_workers > 0:
            ctx = mp.get_context("spawn")
            if chunk_size is None:
                chunk_size = num_workers
            with ctx.Pool(num_workers, initializer=worker_init, initargs=(self.attacker, self.victim, self.invoke_limit)) as pool:
                for ret in self.__iter_metrics(zip(dataset, pool.imap(worker_process, self.__iter_dataset(dataset), chunksize=chunk_size))):
                    yield ret
                   
        else:
            def result_iter():
                for data in self.__iter_dataset(dataset):
                    yield attack_process(self.attacker, self.victim, data, self.invoke_limit)
            for ret in self.__iter_metrics(zip(dataset, result_iter())):
                yield ret

    def eval(self, dataset: Iterable[Dict[str, Any]], total_len : Optional[int] = None, visualize : bool = False, progress_bar : bool = False, num_workers : int = 0, chunk_size : Optional[int] = None):
        """
        Evaluation function of `AttackEval`.

        Args:
            dataset: An iterable dataset.
            total_len: Total length of dataset (will be used if dataset doesn't has a `__len__` attribute).
            visualize: Display a pretty result for each data in the dataset.
            progress_bar: Display a progress bar if `True`.
            num_worers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.
        
        Returns:
            A dict of attack evaluation summaries.
        """

        if hasattr(dataset, "__len__"):
            total_len = len(dataset)
        
        def tqdm_writer(x):
            return tqdm.write(x, end="")
        
        if progress_bar:
            result_iterator = tqdm(self.ieval(dataset, num_workers, chunk_size), total=total_len)
        else:
            # adversarial_sample, attack_time, invoke_times
            result_iterator = self.ieval(dataset, num_workers, chunk_size)

        total_result = {}
        total_result_cnt = {}
        total_inst = 0
        success_inst = 0

        f = open("adv.json", mode="w")
        f.write("[")

        # Start For
        for i, res in enumerate(result_iterator):
            total_inst += 1
            success_inst += int(res["success"])

            if res.get("success", False):
                # print(">>> ", success_inst)
                adv_sample = {
                    "id": res["data"]["id"] + "-adv-" + str(success_inst),
                    "orig_id": res["data"]["id"],
                    "title": res["data"]["title"],

                    "context": res["result"]["context"],
                    "question": res["result"]["question"],
                    "answers": res["result"]["answers"]
                }
                f.write(json.dumps(adv_sample))
                f.write(",")

            
            for kw, val in res["metrics"].items():
                if val is None:
                    continue

                if kw not in total_result_cnt:
                    total_result_cnt[kw] = 0
                    total_result[kw] = 0
                total_result_cnt[kw] += 1
                total_result[kw] += float(val)

            # Use for adversarial training
            # if success_inst >= 8759: # 10% of the training set
            #     break

        # End for

        f.seek(f.tell() - 1, os.SEEK_SET)
        f.truncate()
        f.write("]")
        f.close()

        summary = {}
        summary["Total Attacked Instances"] = total_inst
        summary["Successful Instances"] = success_inst
        summary["Attack Success Rate"] = success_inst / total_inst
        for kw in total_result_cnt.keys():
            if kw in ["Succeed"]:
                continue
            if kw in ["Query Exceeded"]:
                summary["Total " + kw] = total_result[kw]
            else:
                summary["Avg. " + kw] = total_result[kw] / total_result_cnt[kw]
        
        if visualize:
            result_visualizer(summary, sys.stdout.write)
        return summary
    