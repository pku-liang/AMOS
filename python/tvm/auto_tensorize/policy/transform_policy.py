from functools import reduce
from ..utils import bi_product
import numpy as np
from ..tensorization_phases import TransformGenerator, TransformApplier


def all_fit(match_results):
    minimum_padding = -1
    chosen_match = None
    # choose match result according minimum padding
    for match_result in match_results:
        effective_volume = 1
        target_volume = 1
        for iv, tv_lst in match_result.axis_map.items():
            intrin_extent = int(iv.dom.extent)
            ext = reduce(
                lambda x, y:
                    x * int(y.dom.extent), tv_lst, 1)
            target_volume *= ext
            iterations = (ext + intrin_extent) - 1 // intrin_extent
            effective_volume *= iterations * intrin_extent
        padding_volume = effective_volume - target_volume
        if minimum_padding < 0:
            minimum_padding = padding_volume
            chosen_match = match_result
        else:
            if padding_volume < minimum_padding:
                minimum_padding = padding_volume
                chosen_match = match_result

    gen = TransformGenerator(chosen_match)
    record = gen.get(policy="random")
    # here is transform policy
    record.unfold_choice = (
        [1 for _ in record.unfold_choice[0]], record.unfold_choice[1])
    # app = TransformApplier(match_result, verbose=False)
    # new_state = app.apply(record)
    return chosen_match, record

def first_fit(match_results):
    for match_result in match_results:
        if not len(match_result.axis_map.values()):
            continue
        choices = bi_product(
            len(list(match_result.axis_map.values())[0]))
        # random permutation
        np.random.shuffle(choices)
        gen = TransformGenerator(match_result)
        record = gen.get(policy="random")
        for bit_vec in choices:
            if reduce(lambda x, y: x + y, bit_vec, 0) == 0:
                continue
            tmp_set = {}
            value_set = {}
            for ind, v in enumerate(bit_vec):
                if v:
                    for k, lst in match_result.axis_map.items():
                        if k not in tmp_set:
                            tmp_set[k] = set()
                            value_set[k] = 1
                        if hash(lst[ind]) not in tmp_set[k]:
                            tmp_set[k].add(hash(lst[ind]))
                            value_set[k] *= int(lst[ind].dom.extent)
            found = True
            for k, v in value_set.items():
                if v < int(k.dom.extent):
                    found = False
                    break
            if found:
                record.unfold_choice = (
                    bit_vec, record.unfold_choice[1])
                return match_result, record
    assert match_result is not None
    assert record is not None
    # return the last one searched
    return match_result, record


def default_score_func(*args, **kwargs):
    assert len(args) > 2
    value_map = args[0]
    intrin_op = args[1]
    target_op = args[2]
    total_volume = 1
    org_volume = 1
    for k, lst in value_map.items():
        ext = reduce(
                lambda x, y: 
                    x * int(y.dom.extent), lst, 1)
        intrin_extent = int(k.dom.extent)
        tiles = (ext + intrin_extent - 1) / intrin_extent
        total_volume *= tiles * intrin_extent
        org_volume *= ext
    return total_volume - org_volume


def best_fit(match_results, score_func=default_score_func):
    def helper2(args):
        match_result, bit_vec = args
        if reduce(lambda x, y: x + y, bit_vec, 0) == 0:
            return 1e10
        tmp_set = {}
        value_set = {}
        for ind, v in enumerate(bit_vec):
            if v:
                for k, lst in match_result.axis_map.items():
                    if k not in tmp_set:
                        tmp_set[k] = set()
                        value_set[k] = []
                    if hash(lst[ind]) not in tmp_set[k]:
                        tmp_set[k].add(hash(lst[ind]))
                        value_set[k].append(lst[ind])
        for intrin_op, target_op in match_result.main_op_map.items():
            pass
        score = score_func(value_set, intrin_op, target_op)
        return score

    def helper1(idx):
        match_result = match_results[idx]
        choices = bi_product(
            len(list(match_result.axis_map.values())[0]))
        args = [(match_result, choice) for choice in choices]
        score_lst = list(map(helper2, args))
        best_ind = np.argmin(score_lst)
        return (match_result, choices[best_ind], score_lst[best_ind])

    args = range(len(match_results))
    results = list(map(helper1, args))
    results = sorted(results, key=lambda x: x[2])
    assert len(results) > 0
    # choose the minimal one
    match_result, choice, score = results[0]
    gen = TransformGenerator(match_result)
    record = gen.get(policy="random")
    # here is transform policy
    record.unfold_choice = (
        choice, record.unfold_choice[1])
    return match_result, record
