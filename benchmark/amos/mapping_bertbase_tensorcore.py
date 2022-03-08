# we implement a simple graph frontend based on TVM called tensor_graph
import argparse
from tvm import tensor_graph, tg, auto_tensorize as at


def main(batch, dtype, trials):
    trials = 200 if trials < 0 else trials
    print("Batch", batch)
    print("Dtype", dtype)
    print("Trials", trials)
    ########################################
    # set the configurations of the network
    ########################################
    N = batch
    in_dtype = dtype
    # currently, please use the same data type for input and output
    out_dtype = in_dtype
    target = "cuda"
    # https://huggingface.co/bert-base-uncased/blob/main/config.json
    bert_base_config = {
        "architectures": ["BertForMaskedLM"],
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
        "vocab_size": 30522,
    }
    T = bert_base_config["max_position_embeddings"]
    d_model = bert_base_config["hidden_size"]
    d_ff = bert_base_config["intermediate_size"]
    num_blocks = bert_base_config["num_hidden_layers"]
    num_heads = bert_base_config["num_attention_heads"]
    # obtain the network implementation
    model = tensor_graph.testing.models.Transformer(
        num_blocks, num_heads, d_ff, d_model, dtype=in_dtype, out_dtype=out_dtype
    )
    model.eval()

    # set the input shape
    x = tensor_graph.core.GraphTensor([N, T, d_model], dtype=in_dtype, name="data")

    # get forward graph and tir graph using tensor_graph
    # the mapping and scheduling all happens on tir graph
    fwd_graph = tensor_graph.core.make_fwd_graph(model, [x])
    tir_graph = tensor_graph.core.make_tir_graph(fwd_graph, inference=True)
    multi_graph = tg.make_tir_multi_graph(tir_graph)

    # we use a dispatch logic to regulate the whole exploration process
    dispatch = tensor_graph.core.AutoScheduleMultiGraphDispatch
    # the measure option
    measure_opt = at.MeasureOptions(target=target, timeout=100, number=200, min_repeat_ms=500)
    # add our network task into the dispather
    # we only set 20 trials per round for each subgraph for simplicity
    # to obtain good performance, it is recommended to set trials to at least 200
    tid = dispatch.add_graph_task(
        "bertbase", multi_graph, measure_opt, scheduler_option="auto_tensorize", trials=trials
    )
    ########################################
    # start tuning and exploration
    ########################################
    rounds = 10
    # we can tune multiple rounds to obtain better performance
    # e.g. set rounds to 20
    for i in range(rounds):
        # begin autoschedule: mapping exploration + scheduling
        dispatch.auto_schedule(tid)
        # check the intermediate results
        sch_tensors = dispatch.get_schedules(tid)
        if dispatch.ready(tid):
            # check if the whole net is runnable
            # if so, evaluate the end-to-end performance
            cost = at.evaluate_graph(multi_graph, sch_tensors, target, 0, 100, True)
            print("Whole graph cost is %f ms" % cost, flush=True)
        else:
            print("not ready yet")


example_text = """
 example:
    python mapping_bertbase_tensorcore.py --dtype float16 --trials 20
    python mapping_bertbase_tensorcore.py --dtype float32 --trials 20
    python mapping_bertbase_tensorcore.py --dtype float64 --trials 200
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "float64"],
        default="float16",
    )
    parser.add_argument("--trials", type=int, default=-1)

    args = parser.parse_args()
    main(args.batch, args.dtype, args.trials)
