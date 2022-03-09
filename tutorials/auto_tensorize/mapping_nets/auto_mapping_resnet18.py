"""
AutoTensorize: find mappings for ResNet-18 on GPU
===========================================
**Author**: `Size Zheng <https://github.com/KnowingNothing>`


The way to find mappings for the whole network is
essentially to complete the mapping layer by layer.
Some layers themselves cannot be mapped to Tensor Core.
We use a fallback scheduler (TG) for these layers.
It is also feasible to use Ansor as the fallback scheduler.
But currently, the integration of Ansor is not complete.

We use ResNet-18 as example in this tutorial.
"""
# we implement a simple graph frontend based on TVM called tensor_graph
from tvm import tensor_graph, tg, auto_tensorize as at


def main():
    ########################################
    # set the configurations of the network
    ########################################
    batch_size = 1
    in_dtype = "float16"
    # currently, please use the same data type for input and output
    out_dtype = in_dtype
    target = "cuda"
    # obtain the network implementation of ResNet-18
    model = tensor_graph.testing.models.resnet18(
        num_classes=1000, dtype=in_dtype, out_dtype=out_dtype
    )
    model.eval()

    # set the input shape
    img_shape = [batch_size, 3, 224, 224]
    # use tensor_graph frontend to construct a symbolic input tensor
    img_tensor = tensor_graph.core.GraphTensor(img_shape, in_dtype, name="data")

    # get forward graph and tir graph using tensor_graph
    # the mapping and scheduling all happens on tir graph
    fwd_graph = tensor_graph.core.make_fwd_graph(model, [img_tensor])
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
        "resnet18", multi_graph, measure_opt, scheduler_option="auto_tensorize", trials=20
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


if __name__ == "__main__":
    main()
