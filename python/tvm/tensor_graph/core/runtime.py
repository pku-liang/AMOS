import tvm
import time
import _thread
import queue

from tvm import tg


"""
These are deprecated functions, will be removed in future 
"""
# def run_function(func, real_bufs, q):
#   func_beg = time.time()
#   func(*real_bufs)
#   func_end = time.time()
#   q.put((func_end - func_beg) * 1e3)
#   print("from process=", (func_end - func_beg) * 1e3)


# def run_all(tir_graph, input_tensors, label_tensors=None, lr_tensor=None, target="llvm", dev=0):
#   for inp, tensor in zip(tir_graph.inputs, input_tensors):
#     tir_graph.set_inputs({inp: tensor})
#   if label_tensors is not None:
#     for inp, tensor in zip(tir_graph.labels, label_tensors):
#       tir_graph.set_labels({inp, tensor})
#   if lr_tensor is not None:
#     tir_graph.set_lr(lr_tensor)
  
#   tir_graph.allocate_buffer(target, dev)

#   beg = time.time()
#   for mark in tir_graph.call_order:
#     sch = tir_graph.schedules[mark]
#     func = tir_graph.functions[mark]
#     bufs = tir_graph.bufs[mark]
#     # print(tvm.lower(sch, bufs, simple_mode=True))
#     subgraph = tir_graph.subgraphs[mark]
#     real_bufs = [tir_graph.tvm_array_dict[tir_graph.subgraphs[mark].index[x]] for x in bufs]
#     # p = multi.Process(target=func, args=tuple(real_bufs))
#     q = queue.Queue(1)
#     func_beg = time.time()
#     try:
#       # p.start()
#       # p.join()
#       # func(*real_bufs)
#       _thread.start_new_thread(run_function, (func, real_bufs, q))
#       succ = True
#     except Exception as e:
#       succ = False
#     func_end = time.time()

#     # time_cost = (func_end - func_beg) * 1e3
#     time_cost = q.get(block=True, timeout=10)

#     if succ and time_cost >= 0.0:
#       gflop = 0
#       for op, stat in subgraph.op_stat_dict.items():
#         gflop += stat.gflop
#       gflops = gflop / time_cost * 1e3
#       percentage = gflops / (15.7 * 1e3)
#     else:
#       gflops = 0.0
#       percentage = 0.0
    
#     print("time cost =", time_cost, "ms. percentage =", percentage, "gflops =", gflops)

#     for namespace in subgraph.connected_sets.keys():
#       global_primitive_scheduler.feedback(namespace, percentage)

#   end = time.time()
#   print("total time cost =", (end - beg) * 1e3, "ms")


"""
New functions and classes
"""

class SingleGraphSession(object):
  def __init__(
    self,
    tir_graph,
    target="llvm",
    dev_id=0,
    test_only=False,
    reference_file="",
    expected_tuning_iterations=1000,
    report_profile=False,
    report_iteration=True,
    report_iteration_period=100,
    autoschedule_trial_ratio=0.5,
    autoschedule_topk=20,
    autoschedule_new_trial=4,
    autoschedule_policy="profile",
    autoschedule_parallel=1,
    autoschedule_timeout=10.0,
    autoschedule_log_file="autoschedule_log.txt",
    profile_parallel=4,
    profile_timeout=10.0,
    build_parallel=1,
    build_timeout=1.0,
    build_log_file="build_log.txt",
    evaluate_log_file="evaluate_log.txt",
    execution_explore_probability=0.5,
    execution_parallel=1,
    execution_timeout=100.0,
    synchronize_subgraph=True,
    execution_log_file="execution_log.txt"):

    self.sess_option = tg.create_session_option(
      report_profile,
      report_iteration,
      report_iteration_period,
      autoschedule_trial_ratio,
      autoschedule_topk,
      autoschedule_new_trial,
      autoschedule_policy,
      autoschedule_parallel,
      autoschedule_timeout,
      autoschedule_log_file,
      profile_parallel,
      profile_timeout,
      build_parallel,
      build_timeout,
      build_log_file,
      evaluate_log_file,
      execution_explore_probability,
      execution_parallel,
      execution_timeout,
      synchronize_subgraph,
      execution_log_file
    )

    self.sess_id = tg.create_session(target, dev_id, self.sess_option)
    self.tir_graph = tir_graph
    self.task_id = self._set_task(self.tir_graph)
    self.test_only = test_only
    self.reference = reference_file

    if self.test_only:
      self._test_schedule_reference()
    else:
      self._begin_tuning_for_task(expected_tuning_iterations)

  def get_context(self):
    return tg.get_context_from_session(self.sess_id)

  def set_weights(self, tir_graph, weight_bindings):
    tg.initialize_weights(self.sess_id, tir_graph, weight_bindings)

  def _set_task(self, tir_graph):
    return tg.add_task(self.sess_id, tir_graph)

  def _begin_tuning_for_task(self, expected_iterations):
    tg.begin_tuning(self.sess_id, self.task_id, expected_iterations, reference=self.reference)

  def _end_tuning_for_task(self):
    tg.end_tuning(self.sess_id, self.task_id)

  def _test_schedule_reference(self):
    tg.test_schedule_reference(self.sess_id, self.task_id, reference=self.reference)

  def run(self, data_bindings, save_to=""):
    tg.run_task(self.sess_id, self.task_id, data_bindings, save_to=save_to)

  def __del__(self):
    if not self.test_only:
      self._end_tuning_for_task()
    tg.delete_session(self.sess_id)
