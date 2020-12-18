from .scheduler import global_primitive_scheduler, reschedule_subgraph


def build_all(fwd_graph, tir_graph, build_trial=10, target="llvm"):
  total_trials = 0
  for mark, subgraph in tir_graph.subgraphs.items():
    succ = False
    for trial in range(build_trial):
      total_trials += 1
      succ = tir_graph.build_for(target, mark=mark)
      if succ:
        break
      else:
        for namespace in subgraph.connected_sets.keys():
          global_primitive_scheduler.feedback(namespace, 0.0)
        reschedule_subgraph(fwd_graph, tir_graph, mark)
    if not succ:
      raise RuntimeError("Can't build subgraph", mark)
  
  print("total_trials=", total_trials)
