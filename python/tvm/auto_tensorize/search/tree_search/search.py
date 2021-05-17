from tokenize import generate_tokens
import sys
import os
import json
from .generator_tree import GeneratorTreeNode
from .instance_tree import InstanceTreeNode


class EvaluateEntry(object):
    def __init__(
            self,
            level_1_gen,
            level_2_gen,
            level_1_choice,
            level_2_choice):
        self.level_1_gen = level_1_gen
        self.level_2_gen = level_2_gen
        self.level_1_choice = level_1_choice
        self.level_2_choice = level_2_choice


class EvaluateResult(object):
    def __init__(self, entry, result):
        self.level_1_gen = entry.level_1_gen
        self.level_2_gen = entry.level_2_gen
        self.level_1_choice = entry.level_1_choice
        self.level_2_choice = entry.level_2_choice
        self.result = result


def select_min(x, y):
    return (x, 0) if x < y else (y, 1)


def select_max(x, y):
    return (x, 0) if x > y else (y, 1)


def two_level_seach(
        level_1_generator_impl,
        level_2_generator_impl,
        evaluator_impl,
        level_1_type,
        level_2_type,
        search_trials=1000,
        expand_level_2_generator_per_step=1,
        expand_leaf_per_step=200,
        evaluate_batch=40,
        log_to_file=None,
        select_func=select_min):
    # prepare the generator tree root
    generator_tree_root = GeneratorTreeNode(
        level_1_generator_impl(), level_1_type)

    fout = open(os.devnull, "w") if log_to_file is None else open(
        log_to_file, "a")

    total_trial_counter = 0
    best_result = None
    best_choice = None

    def feedback(evaluate_results):
        nonlocal best_result
        nonlocal best_choice
        for res in evaluate_results:
            # level 1 node feedback
            generator_tree_root.feedback(res)
            child_id = generator_tree_root.get_child_id(
                # this instance has the same hash key as expected child
                InstanceTreeNode(res.level_1_choice, level_1_type)
            )
            child = generator_tree_root.get_child(child_id)
            # get the generator
            for grandson in child.children():
                # level 2 node feedback
                grandson.feedback(res)

            total_trial_counter += 1
            if best_result is None:
                best_result = res.result
                best_choice = (res.level_1_choice, res.level_2_choice)
            else:
                best_result, arg = select_func(best_result, res.result)
                best_choice = best_choice if arg == 0 else (
                    res.level_1_choice, res.level_2_choice)
            print(f"Trial {total_trial_counter} ", end="| ", flush=True)
            print(f"Current: {res.result}/{best_result} ", end="| ", flush=True)
            print(json.dumps({level_1_type: level_1_choice.to_json(),
                              level_2_type: level_2_choice.to_json()}), flush=True)

            # logging
            print(json.dumps({level_1_type: level_1_choice,
                              level_2_type: level_2_choice,
                              "result": res.result}), file=fout, flush=True)

    # search sketch
    steps = search_trials // (
        expand_level_2_generator_per_step * expand_leaf_per_step)
    to_evaluate_list = []
    print(f"Totally {steps} steps.")
    for st in range(steps):
        level_1_choice = generator_tree_root.generate_next()
        level_1_instance_node = InstanceTreeNode(level_1_choice, level_1_type)
        unique_node = generator_tree_root.append_child(level_1_instance_node)
        if not unique_node.is_leaf():
            for child in unique_node.children():
                for j in range(expand_leaf_per_step):
                    level_2_choice = child.generate_next()
                    # no need to attach level_2_choice to the tree
                    to_evaluate_list.append(
                        EvaluateEntry(
                            generator_tree_root._generator,
                            child._generator,
                            level_1_choice, level_2_choice))
                    total_trial_counter += 1

                    if len(to_evaluate_list) % evaluate_batch == 0:
                        evaluate_results = evaluator_impl(to_evaluate_list)
                        to_evaluate_list.clear()
                        feedback(evaluate_results)
        else:
            for i in range(expand_level_2_generator_per_step):
                gen = level_2_generator_impl(level_1_choice)
                level_2_generator_node = GeneratorTreeNode(
                    gen, level_2_type
                )
                unique_node.append_child(level_2_generator_node)
                for j in range(expand_leaf_per_step):
                    level_2_choice = level_2_generator_node.generate_next()
                    # no need to attach level_2_choice to the tree
                    to_evaluate_list.append(
                        EvaluateEntry(
                            generator_tree_root._generator,
                            level_2_generator_node._generator,
                            level_1_choice, level_2_choice))
                    total_trial_counter += 1

                    if len(to_evaluate_list) % evaluate_batch == 0:
                        evaluate_results = evaluator_impl(to_evaluate_list)
                        to_evaluate_list.clear()
                        feedback(evaluate_results)

    return best_choice, best_result
