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
    import torchvision.models as models
    model = models.resnet18()
    img = torch.empty(1, 3, 224, 224)
    output = model(img)
    node_count, node_names = count_nodes(output)
    print(f'Found {node_count} nodes')
