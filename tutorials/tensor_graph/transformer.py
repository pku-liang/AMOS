from tvm import tensor_graph, tg, auto_tensorize as at
from collections import OrderedDict


TEST_CASES = OrderedDict()


def register_test(func):
    name = func.__name__
    prefix = "test"
    assert name[:len(prefix)] == prefix
    try:
        number = int(name[len(prefix):])

        def _inner(*args, **kwargs):
            print(func.__doc__)
            func(*args, **kwargs)
        assert number not in TEST_CASES, "Repeated test case number %d" % number
        TEST_CASES[number] = _inner
    except ValueError as e:
        print(e)
        print("Can't convert to number", name[len(prefix):])


@register_test
def test1():
    N = 5
    T = 512
    d_model = 10
    d_ff = 4
    num_blocks = 2
    num_heads = 2
    dtype="float16"
    out_dtype="float16"
    target = "cuda"

    model =  tensor_graph.testing.models.Transformer(num_blocks, num_heads, d_ff, d_model, dtype=dtype, out_dtype=out_dtype)
    model.eval()

    x = tensor_graph.core.GraphTensor([N, T, d_model], dtype=dtype, name="data")

    # get forward graph and tir graph
    fwd_graph = tensor_graph.core.make_fwd_graph(model, [x])
    tir_graph = tensor_graph.core.make_tir_graph(fwd_graph, inference=True)
    multi_graph = tg.make_tir_multi_graph(tir_graph)

    dispatch = tensor_graph.core.AutoScheduleMultiGraphDispatch
    measure_opt = at.MeasureOptions(
        target=target, timeout=10, number=200, min_repeat_ms=500)
    tid = dispatch.add_graph_task(
        "transformer", multi_graph, measure_opt, scheduler_option="auto_tensorize")
    dispatch.auto_schedule(tid)
    sch_tensors = dispatch.get_schedules(tid)
    cost = at.evaluate_graph(multi_graph, sch_tensors, target, 0, 10, False)
    print("Whole graph cost is %f ms" % cost)


@register_test
def test2():
    # https://huggingface.co/bert-base-uncased/blob/main/config.json
    bert_base_config = {
        "architectures": [
            "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522
    }

    N = 1  # Batch Size
    T = bert_base_config['max_position_embeddings']
    d_model = bert_base_config['hidden_size']
    d_ff = bert_base_config['intermediate_size']
    num_blocks = bert_base_config['num_hidden_layers']
    num_heads = bert_base_config['num_attention_heads']
    dtype = "float16"
    out_dtype = "float16"
    target = "cuda"

    model = tensor_graph.testing.models.Transformer(num_blocks, num_heads, d_ff, d_model, dtype=dtype, out_dtype=out_dtype)
    model.eval()

    x = tensor_graph.core.GraphTensor([N, T, d_model], dtype=dtype, name="data")

    # get forward graph and tir graph
    fwd_graph = tensor_graph.core.make_fwd_graph(model, [x])
    tir_graph = tensor_graph.core.make_tir_graph(fwd_graph, inference=True)
    multi_graph = tg.make_tir_multi_graph(tir_graph)

    dispatch = tensor_graph.core.AutoScheduleMultiGraphDispatch
    measure_opt = at.MeasureOptions(
        target=target, timeout=10, number=200, min_repeat_ms=500)
    tid = dispatch.add_graph_task(
        "transformer", multi_graph, measure_opt, scheduler_option="auto_tensorize")
    dispatch.auto_schedule(tid)
    sch_tensors = dispatch.get_schedules(tid)
    cost = at.evaluate_graph(multi_graph, sch_tensors, target, 0, 10, False)
    print("Whole graph cost is %f ms" % cost)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="test case", type=int, default=1)
    parser.add_argument("--all", help="test all", action="store_true")

    args = parser.parse_args()
    if args.all:
        for k, v in TEST_CASES.items():
            print("############################################")
            print("test", k)
            v()
            print("Pass!")
    else:
        assert args.case in TEST_CASES, "Can't find case %s." % (
            str(args.case))
        case = TEST_CASES[args.case]
        case()
