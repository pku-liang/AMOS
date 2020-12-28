import json

from tvm.tensor_graph.nn.functional import cross_entropy
from tvm.tensor_graph.testing.datasets import load_mnist_dataset
from tvm.tensor_graph.testing.learners.image_classification_learner import ImageClassificationLearner
from tvm.tensor_graph.testing.models import LeNet


def pprint_dict(d):
    return json.dumps(d, indent=2)


batch_size = 4
# lr = 1e-6
lr = lambda epoch: [1e-2, 1e-3, 1e-4][epoch]  # learning rate scheduler
num_epochs = 3
num_classes = 10
target = 'llvm'
dtype = 'float64'

model = LeNet()
train_loader = load_mnist_dataset(batch_size)[0]
criterion = cross_entropy
learner = ImageClassificationLearner(model, train_loader, num_classes, criterion, lr, debug_mode=True, target=target, dtype=dtype)

state_dict = {
    key: (nparray.min(), nparray.max())
    for key, nparray in learner.state_dict.items()
}
print('state_dict:', pprint_dict(state_dict))

images, targets = next(iter(train_loader))
# noinspection PyProtectedMember
learner._train_one_step(images, targets, record=False)
debug_dict = {
    key: (nparray.min(), nparray.max()) if not isinstance(nparray, float) else nparray
    for key, nparray in learner.debug_dict.items()
}
grads_dict = {
    key: (nparray.min(), nparray.max()) if not isinstance(nparray, float) else nparray
    for key, nparray in learner.grads_dict.items()
}

print('debug_dict:', pprint_dict(debug_dict))
print('grads_dict:', pprint_dict(grads_dict))

for epoch_idx in range(num_epochs):
    learner.train_one_epoch(epoch_idx)

print(learner.get_gradient('conv1_weight').shape)
