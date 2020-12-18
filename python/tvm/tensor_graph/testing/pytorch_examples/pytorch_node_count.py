import torch


def count_nodes(var, verbose=True):
    node_count = 0
    node_names = list()
    seen = set()

    def add_nodes(var):
        nonlocal node_count
        if var in seen: return
        
        if not torch.is_tensor(var) and not hasattr(var, 'variable'):
            node_count += 1
            node_names.append(str(type(var).__name__))

        seen.add(var)

        if hasattr(var, 'next_functions'):
            for u in var.next_functions:
                if u[0] is not None:
                    add_nodes(u[0])

        if hasattr(var, 'saved_tensors'):
            for t in var.saved_tensors:
                add_nodes(t)

    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    if verbose:
        print(f'Found {node_count} nodes.')
    
    return node_count, node_names


if __name__ == '__main__':
    # import torchvision.models as models
    # model = models.resnet18()
    # img = torch.empty(1, 3, 224, 224)
    # output = model(img)
    # node_count, node_names = count_nodes(output) # 72

    import tensor_graph.testing.pytorch_examples.capsule_torch as models1
    model1 = models1.CapsuleNetwork()
    img1 = torch.empty(2, 1, 28, 28)
    output1 = model1(img1)
    node_count, node_names = count_nodes(output1) # 91

    import tensor_graph.testing.pytorch_examples.MILSTM as models2
    model2 = models2.MILSTM()
    img2 = torch.empty(1, 28*28)
    output2 = model2(img2)
    node_count, node_names = count_nodes(output2) # 67


    import tensor_graph.testing.pytorch_examples.SCRNN as models3
    model3 = models3.SCRNN()
    img3 = torch.empty(1, 28*28)
    output3 = model3(img3)
    node_count, node_names = count_nodes(output3) # 12
    
    import tensor_graph.testing.pytorch_examples.lltm as models4
    model4 = models4.RnnLLTM(28*28, 128, 10)
    img4 = torch.empty(1, 28*28)
    output4 = model4(img4)
    node_count, node_names = count_nodes(output4) # 12

    import tensor_graph.testing.pytorch_examples.subLSTM as models5
    model5 = models5.RnnsubLSTM()
    img5 = torch.empty(1, 28*28)
    output5 = model5(img5)
    node_count, node_names = count_nodes(output5) # 14

    import tensor_graph.testing.pytorch_examples.mobilenet_v1_torch as models6
    model6 = models6.MobileNetv1()
    img6 = torch.empty(1, 3, 224, 224)
    output6 = model6(img6)
    node_count, node_names = count_nodes(output6) # 85

    import tensor_graph.testing.pytorch_examples.mobilenet_v2_torch as models7
    model7 = models7.MobileNetV2()
    img7 = torch.empty(1, 3, 224, 224)
    output7 = model7(img7)
    node_count, node_names = count_nodes(output7) # 153

    import tensor_graph.testing.pytorch_examples.capsule_conv as models8
    model8 = models8.CapsuleConv2d(1, 256, 9, (1, 1), (1, 8))
    img8 = torch.empty(1, 1, 28, 28, 1, 1)
    output8 = model8(img8)
    node_count, node_names = count_nodes(output8) # 26


    import tensor_graph.testing.pytorch_examples.shufflenet_v1_torch as models9
    model9 = models9.ShuffleNet()
    img9 = torch.empty(1, 3, 224, 224)
    output9 = model9(img9)
    node_count, node_names = count_nodes(output9) # 218

    import tensor_graph.testing.pytorch_examples.yolo_v1 as models10
    model10 = models10.yolo_v1()
    img10 = torch.empty(1, 3, 448, 448)
    output10 = model10(img10)
    node_count, node_names = count_nodes(output10) # 58