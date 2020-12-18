import tvm

from tensor_graph.core.scheduler import PrimitiveScheduler


def test1():
  import random
  scheduler = PrimitiveScheduler()
  for i in range(1000):
    name = "test_" + str(i%3)
    outer, inner, use_factor = scheduler.schedule_allreduce(name, 3, 224, i%3+1)
    if outer == 32 and use_factor == 0:
      perf = 0.8
    elif inner == 32 and use_factor == 1:
      perf = 0.7
    else:
      perf = 0.1
    scheduler.feedback(name, perf)


def test2():
  import random
  scheduler = PrimitiveScheduler()
  for i in range(1000):
    name = "test_" + str(i%3)
    f_list1, f_list2, f_list3 = scheduler.schedule_decomposition(name, (4, 3, 6), 1, 3, 0)
    scheduler.feedback(name, random.random())


def test3():
  import random
  scheduler = PrimitiveScheduler()
  for i in range(1000):
    name = "test_" + str(i%3)
    f_list1, f_list2, f_list3 = scheduler.schedule_reductive(name, (4, 3, 6), 1, 2)
    scheduler.feedback(name, random.random())


if __name__ == "__main__":
  test1()
  test2()
  test3()