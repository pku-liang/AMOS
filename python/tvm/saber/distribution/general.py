import numpy as np


class ShapeGroup(object):
    def __init__(self, group_id, shapes):
        """
        group_id: int
        shapes: ndarry(float32) MxN, M is group size, N is shape dim
        """
        self.group_id = group_id
        self.shapes = shapes


def normalize_params(params):
    array = np.array(params)
    max_item = np.max(array, axis=0)
    min_item = np.min(array, axis=0)
    return (array - min_item) / (max_item - min_item + 1e-10)


def identity_feature(params):
    return params.to_flatten_tuple()


def split_ndarry(ndarry, conds):
    return [ndarry[cond] for cond in conds]


def group_shapes(
    cluster,
    shape_class,
    full_param_input_lst,
    count_lst,
    normalize_func=normalize_params,
    feature_func=identity_feature):
    """
    cluster: tool such as KMeans
    """    
    X = np.array([list(map(float, feature_func(x))) for x in full_param_input_lst])
    shapes = X
    X = normalize_func(X)
    
    cluster.fit(X)
    predicts = cluster.predict(X).squeeze()
    subarrys = split_ndarry(shapes, [predicts == i for i in range(len(cluster.get_centers()))])
    shape_groups = [
        ShapeGroup(i, list(map(lambda x: shape_class.from_flatten_tuple(x.astype("int32").tolist()), y)))
        for i, y in enumerate(subarrys)
    ]
    return shape_groups