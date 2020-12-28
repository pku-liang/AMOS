from collections import OrderedDict
from typing import Union, Callable

import numpy as np
import torch
import tvm

from tvm.tensor_graph.nn.functional import cross_entropy
from tvm.tensor_graph.testing.learners.utils import to_tuple


class ImageClassificationLearner:

    # noinspection PyProtectedMember
    def __init__(self, model, train_loader, num_classes, criterion,
                 lr: Union[float, Callable[[int], float]],  # lr(epoch,) -> lr
                 debug_mode=False, print_freq=1000, target='llvm', dtype='float64'):
        self.model = model
        self.train_loader = train_loader
        self.num_classes = num_classes
        self.criterion = criterion
        self.lr = lr if isinstance(lr, float) else lr(0)
        self._lr_func = lr if not isinstance(lr, float) else lambda epoch: lr

        self.debug_mode = debug_mode
        self.print_freq = print_freq
        self.target = target
        self.dtype = dtype
        self.ctx = tvm.context(target)

        self._build_func()
        self._allocate_buffers_for_endpoints()
        self._initialize_weights()

    def _build_func(self):
        images_pth, labels_pth = next(iter(self.train_loader))
        self.images = tvm.te.placeholder(list(images_pth.shape), dtype=self.dtype, name='images')
        self.labels = tvm.te.placeholder([labels_pth.shape[0], self.num_classes], dtype=self.dtype, name='labels')
        if self.debug_mode:
            self.logit, self.debug_tensors = self.model(self.images, debug_mode=self.debug_mode)
        else:
            self.logit = self.model(self.images, debug_mode=self.debug_mode)
        self.loss = cross_entropy(self.logit, self.labels)
        self.gradients = tvm.tg.gradient(self.loss, self.model.weights)
        extra_args = list(self.debug_tensors.values()) if self.debug_mode else list()
        self.sched = tvm.te.create_schedule([self.loss.op] + [tensor.op for tensor in extra_args] + [grad.op for grad in self.gradients])
        args = [self.images, self.labels, *self.model.weights, self.logit, self.loss, *extra_args, *self.gradients]
        # print(tvm.lower(self.sched, args, simple_mode=True))
        self.func = tvm.build(self.sched, args, target=self.target)

    def _allocate_buffers_for_endpoints(self):
        def create_buffer(tensor):
            np_buffer = np.zeros(to_tuple(tensor.shape)).astype(self.dtype)
            tvm_buffer = tvm.nd.array(np_buffer, self.ctx)
            return tvm_buffer

        self.logit_tvm = create_buffer(self.logit)
        self.loss_tvm = create_buffer(self.loss)
        if self.debug_mode:
            self.debug_tensors_tvm = {
                key: create_buffer(tensor)
                for key, tensor in self.debug_tensors.items()
            }
        else:
            self.debug_tensors_tvm = {}

    def _initialize_weights(self):

        # TODO: support BatchNorm2d
        # NOTE: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py#L49-L60
        def init_weight(var):
            w_pth = torch.empty(*to_tuple(var.shape), dtype=torch.float64)
            if len(w_pth.shape) == 4:  # Conv2d
                # NOTE: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
                torch.nn.init.kaiming_normal_(w_pth, mode='fan_out', nonlinearity='relu')
            elif len(w_pth.shape) == 2:  # Linear
                torch.nn.init.normal_(w_pth, mean=0, std=0.01)
            elif len(w_pth.shape) == 1:  # bias
                torch.nn.init.constant_(w_pth, 0)
            else:
                raise NotImplementedError(f'Unrecognized weight shape: {var.shape}')
            return w_pth.numpy()

        self.weights_np = [init_weight(var) for var in self.model.weights]
        # self.weights_np = [np.random.uniform(-1, 1, to_tuple(var.shape)).astype(self.dtype) for var in self.model.weights]
        self.weights_tvm = [tvm.nd.array(var, self.ctx) for var in self.weights_np]

    def _preprocess_batch(self, images, targets):
        images_np = images.numpy().astype(self.dtype)
        images_tvm = tvm.nd.array(images_np, self.ctx)
        labels_np = np.zeros([images.shape[0], 10]).astype(self.dtype)
        labels_pth = torch.tensor(labels_np)
        labels_pth.scatter_(1, targets.unsqueeze(0).T, 1.0)
        labels_tvm = tvm.nd.array(labels_pth.numpy(), self.ctx)
        return images_tvm, labels_tvm

    def _reset_gradients(self):
        grads_np = [np.zeros(to_tuple(var.shape)).astype(self.dtype) for var in self.gradients]
        self.grads_tvm = [tvm.nd.array(var, self.ctx) for var in grads_np]

    def _execute_func(self, images_tvm, targets_tvm):
        self.func(
            images_tvm, targets_tvm, *self.weights_tvm, self.logit_tvm, self.loss_tvm,
            *self.debug_tensors_tvm.values(), *self.grads_tvm
        )
        debug_tensors_np = {key: tvm_array.asnumpy() for key, tvm_array in self.debug_tensors_tvm.items()}
        if not self.debug_mode:
            return self.logit_tvm.asnumpy(), self.loss_tvm.asnumpy().item(0)
        else:
            return self.logit_tvm.asnumpy(), self.loss_tvm.asnumpy().item(0), debug_tensors_np

    def _update_weights(self):
        for k, grad in enumerate(self.grads_tvm):
            assert(self.weights_np[k].shape == grad.asnumpy().shape)
            self.weights_np[k] -= self.lr * grad.asnumpy()
        self.weights_tvm = [tvm.nd.array(var, self.ctx) for var in self.weights_np]

    def _train_one_step(self, images, targets, record=True):
        batch_tvm = self._preprocess_batch(images, targets)
        self._reset_gradients()
        logit_np, loss_val, *debug_tensors_np = self._execute_func(*batch_tvm)
        if self.debug_mode:
            self.debug_tensors_np = debug_tensors_np[0]
            self.debug_tensors_np.update({ 'logit': logit_np, 'loss': loss_val, })
        else:
            self.debug_tensors_np = dict()

        preds = torch.from_numpy(np.argmax(logit_np, axis=1))
        if record:
            self.running_acc += (preds == targets).sum().item()
            self.running_loss += loss_val * images.size()[0]
        self._update_weights()

    def train_one_epoch(self, epoch_idx):
        num_covered = 0
        self.running_acc = 0.0
        self.running_loss = 0.0
        self.lr = self._lr_func(epoch_idx)
        for i, (images, targets) in enumerate(self.train_loader):
            num_covered += images.size()[0]
            self._train_one_step(images, targets)
            if i % self.print_freq == 0:
                loss_avg = self.running_loss / num_covered
                acc_avg = self.running_acc / num_covered
                print(f"epoch = {epoch_idx+1}, iteration = {i+1}: lr = {self.lr}, loss = {loss_avg}, acc = {acc_avg}")
        assert num_covered == len(self.train_loader.dataset)
        acc_avg = self.running_acc / num_covered
        print(f"epoch = {epoch_idx+1}: accuracy = {acc_avg}")

    def get_gradient(self, weight_key):
        for weight, grad in zip(self.model.weights, self.grads_tvm):
            if weight.name == weight_key:
                return grad.asnumpy()

    @property
    def state_dict(self):
        return OrderedDict({
            weight.name: weight_np
            for weight, weight_np in zip(self.model.weights, self.weights_np)
        })

    @property
    def grads_dict(self):
        return OrderedDict({
            weight.name: grad.asnumpy()
            for weight, grad in zip(self.model.weights, self.grads_tvm)
        })

    @property
    def debug_dict(self):
        return self.debug_tensors_np
