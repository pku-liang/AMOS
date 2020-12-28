import tvm
from tvm import topi
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from tvm.tensor_graph.testing.learners.utils import to_tuple, assert_print

print_per_iteration = 100
debug_mode = False
debug_print_cnt = 2
CLIP_VALUE = 1e20
batch_size = 100
learning_rate = 1
num_epoches = 3
state_size = 128
input_size = 28*28
dtype = "float64"
target_platform = "llvm"


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def numpy_init(weight_list,*args):
  '''
  the first argument is randomly initialized. All others are zero initialized.
  '''
  weight_np = [np.random.uniform(-1, 1, to_tuple(var.shape)).astype(dtype) for var in weight_list]
  init = [weight_np]
  if len(args) > 0:
    for item in args:
      init.append([np.zeros(to_tuple(var.shape), dtype=dtype) for var in item])
  return init

def cross_entropy(inputs, targets):
  '''
  https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
  loss(x,class) = −x[class]+log(\Sigma:j exp(x[j]))
  -x[class]: since targets is one-hot, we use "inner-dot" to compute x_class
  log(\Sigma:j exp(x[j])) may overflow when computing exp(x[j])
  >>>>Trick: maxval + log(\Sigma: j exp(x[j] - maxval))
  Finally, compute the average over batches
  '''
  assert_print(inputs.shape[0].value == targets.shape[0].value)
  assert_print(inputs.shape[1].value == targets.shape[1].value)
  N, C = inputs.shape
  c = tvm.te.reduce_axis([0, C], "c")
  k1 = tvm.te.reduce_axis([0, C], name="k1")
  # First compute the maximum for each batch
  max_val = tvm.te.compute([N], lambda n: tvm.te.max(inputs[n, k1], axis=[k1]), name="max_val")
  # Use the log_softmax trick to avoid overflow
  sum_val = tvm.te.compute([N], lambda i: tvm.te.sum(tvm.tir.exp(inputs[i, c]-max_val[i]), axis=[c]), "sum_val")
  rrn = tvm.te.reduce_axis([0, N], "rrn")
  rrc = tvm.te.reduce_axis([0, C], "rrc")
  x_class = tvm.te.compute([N], lambda i: tvm.te.sum(inputs[i, rrc]*targets[i, rrc], axis=[rrc]), name="x_class")
  return tvm.te.compute([1],
    lambda i: tvm.te.sum((tvm.tir.log(sum_val[i+rrn])+max_val[i+rrn] - x_class[i+rrn]) / N, axis=[rrn]), name="cross_entropy")

def internel_sublstm(input, weight_ih, weight_hh, bias_ih, bias_hh, old_h, old_c):
  '''
  input: [batch_size, input_size]
  weight_ih: [4*state_size, input_size]
  weight_hh: [4*state_size, state_size]
  bias_ih:[4*state_size]
  bias_hh:[4*state_size]
  old_h  & old_c: [batch_size, state_size]
  ------
  cell_gate: [batch_size, state_size]
  '''
  gates = topi.add(topi.nn.dense(input, weight_ih, bias_ih), topi.nn.dense(old_h, weight_hh, bias_hh))
  in_gate, forget_gate, cell_gate, out_gate = topi.split(gates, 4, axis=1)

  in_gate = topi.sigmoid(in_gate)
  forget_gate = topi.sigmoid(forget_gate)
  cell_gate = topi.sigmoid(cell_gate)
  out_gate = topi.sigmoid(out_gate)

  new_c = forget_gate * old_c + cell_gate - in_gate
  new_h = topi.sigmoid(new_c) - out_gate
  return [new_h, new_c, cell_gate]

def sublstm(input, targets, weight_ih, weight_hh, bias_ih, bias_hh, weight_ih2, weight_hh2, bias_ih2, bias_hh2, weight_for_classify, bias_for_classify, old_h, old_c, old_h2, old_c2):
  '''
  input: [batch_size, input_size]
  targets: [batch_size, 10] one-hot
  weight_ih(2): [4*state_size, input_size]
  weight_hh(2): [4*state_size, state_size]
  bias_ih(2):[4*state_size]
  bias_hh(2):[4*state_size]
  old_h(2)  & old_c(2): [batch_size, state_size]
  '''
  new_h, new_c, cell_gate = internel_sublstm(input, weight_ih, weight_hh, bias_ih, bias_hh, old_h, old_c)
  new_h2, new_c2, cell_gate2 = internel_sublstm(cell_gate, weight_ih2, weight_hh2, bias_ih2, bias_hh2, old_h2, old_c2)
  result = topi.nn.dense(new_h2, weight_for_classify, bias_for_classify)
  loss = cross_entropy(result, targets)
  return loss, result, new_h, new_c, new_h2, new_c2

def main():
  global debug_print_cnt
  img = tvm.te.placeholder([batch_size, input_size], dtype=dtype, name="img")
  label = tvm.te.placeholder([batch_size, 10], dtype=dtype, name="label")

  weight_ih = tvm.te.placeholder([4*state_size, input_size], dtype=dtype, name="weight_ih")
  weight_hh = tvm.te.placeholder([4*state_size, state_size], dtype=dtype, name="weight_hh")
  bias_ih = tvm.te.placeholder([4*state_size], dtype=dtype, name="bias_ih")
  bias_hh = tvm.te.placeholder([4*state_size], dtype=dtype, name="bias_hh")

  weight_ih2 = tvm.te.placeholder([4*state_size, state_size], dtype=dtype, name="weight_ih2")
  weight_hh2 = tvm.te.placeholder([4*state_size, state_size], dtype=dtype, name="weight_hh2")
  bias_ih2 = tvm.te.placeholder([4*state_size], dtype=dtype, name="bias_ih2")
  bias_hh2 = tvm.te.placeholder([4*state_size], dtype=dtype, name="bias_hh2")

  old_h = tvm.te.placeholder([batch_size, state_size], dtype=dtype, name="old_h")
  old_c = tvm.te.placeholder([batch_size, state_size], dtype=dtype, name="old_c")

  old_h2 = tvm.te.placeholder([batch_size, state_size], dtype=dtype, name="old_h2")
  old_c2 = tvm.te.placeholder([batch_size, state_size], dtype=dtype, name="old_c2")

  weight_for_classify = tvm.te.placeholder([10, state_size], dtype=dtype, name="weight_for_classify")
  bias_for_classify = tvm.te.placeholder([10], dtype=dtype, name="bias_for_classify")

  # Helper list
  weight_to_update = [weight_ih, weight_hh, bias_ih, bias_hh, weight_ih2, weight_hh2, bias_ih2, bias_hh2, weight_for_classify, bias_for_classify]
  old_hc = [old_h, old_c, old_h2, old_c2]
  
  # Function
  loss, result, new_h, new_c, new_h2, new_c2 = sublstm(img, label, *weight_to_update, *old_hc)

  #Helper list
  loss_and_result = [loss, result]
  new_hc = [new_h, new_c, new_h2, new_c2]
  grad_list = tvm.tg.gradient(loss, weight_to_update)

  s = tvm.te.create_schedule([var.op for var in loss_and_result] + [var.op for var in new_hc] + [grad.op for grad in grad_list])
  print(tvm.lower(s, [img, label, *weight_to_update, *old_hc, *loss_and_result, *new_hc, *grad_list], simple_mode=True))
  func = tvm.build(s, [img, label, *weight_to_update, *old_hc, *loss_and_result, *new_hc, *grad_list], target= target_platform)

  weight_np, old_hc_np, loss_result_np, new_hc_np, grad_np = numpy_init(weight_to_update, old_hc, loss_and_result, new_hc, grad_list)
  ctx = tvm.context(target_platform)

  for ep in range(num_epoches):
    train_num_covered = 0
    running_acc = 0.0
    running_loss = 0.0
    for i, data in enumerate(train_loader):
      img_tvm = tvm.nd.array(data[0].squeeze(1).view(batch_size, 28*28).numpy().astype(dtype), ctx)
      label_torch = torch.tensor(np.zeros([batch_size, 10]).astype(dtype))
      label_torch.scatter_(1, data[1].unsqueeze(0).T, 1.0)
      #print("label_torch", label_torch)
      label_tvm = tvm.nd.array(label_torch.numpy(), ctx)
      weight_tvm = [tvm.nd.array(var) for var in weight_np]
      old_hc_tvm = [tvm.nd.array(var) for var in old_hc_np]
      loss_result_tvm = [tvm.nd.array(var) for var in loss_result_np]
      new_hc_tvm = [tvm.nd.array(var) for var in new_hc_np]
      grad_tvm = [tvm.nd.array(var) for var in grad_np]
      if debug_mode:
        print("before func, loss_result_tvm", loss_result_tvm)
        print("before func, img_tvm", img_tvm)
        print("before func, label_tvm", label_tvm)
        print("before func, weight_tvm", weight_tvm)
        print("before func, old_hc_tvm", old_hc_np)
        print("before func, new_hc_tvm", new_hc_tvm)
        print("before func, grad_tvm", grad_tvm)
      func(img_tvm, label_tvm, *weight_tvm, *old_hc_tvm, *loss_result_tvm, *new_hc_tvm, *grad_tvm)
      if debug_mode:
        print("after func, loss_result_tvm", loss_result_tvm)
        print("after func, img_tvm", img_tvm)
        print("after func, label_tvm", label_tvm)
        print("after func, weight_tvm", weight_tvm)
        print("after func, old_hc_tvm", old_hc_np)
        print("after func, new_hc_tvm", new_hc_tvm)
        print("after func, grad_tvm", grad_tvm)
        debug_print_cnt = debug_print_cnt - 1
        if debug_print_cnt == 0:
          exit(0)

      train_num_covered += batch_size
      # loss_result_tvm is a list: 
      # >>>>>> loss:[1], result:[batch_size, 10]
      _, predict = torch.max(torch.from_numpy(loss_result_tvm[1].asnumpy()), 1)
      num_correct = (predict == data[1]).sum()
      running_acc += num_correct.item()
      running_loss += loss_result_tvm[0].asnumpy().item(0)

      if i % print_per_iteration == 0:
        print("epoch=", ep+1, "iteration=", i+1, "loss=", running_loss/train_num_covered, "acc=", running_acc/train_num_covered)
      
      for k, gradient in enumerate(grad_tvm):
        assert(weight_np[k].shape == gradient.asnumpy().shape)
        gradient_clipped = np.clip(np.nan_to_num(gradient.asnumpy(), nan=CLIP_VALUE), -CLIP_VALUE, CLIP_VALUE)
        weight_np[k] -= learning_rate * gradient_clipped
      
      #we update hidden_states and cell_states for next iteration
      for k, item in enumerate(new_hc_tvm):
        assert(old_hc_np[k].shape == item.asnumpy().shape)
        item_clipped = np.clip(np.nan_to_num(item.asnumpy(), nan=CLIP_VALUE), -CLIP_VALUE, CLIP_VALUE)
        old_hc_np[k] = item_clipped
      
      #zero the new hidden_states and new cell_states
      for k, item in enumerate(new_hc_np):
        new_hc_np[k] = np.zeros(new_hc_np[k].shape, dtype=dtype)
          
    assert(train_num_covered == len(train_dataset))
    running_acc /= len(train_dataset)
    print("epoch=", ep+1, "accuracy=", running_acc)


if __name__ == "__main__":
  main()
