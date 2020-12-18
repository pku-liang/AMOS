import tvm
from tvm import tg

utilbox = tg.UtilBox

def test1():
  for ext in [1, 3, 56, 223, 224, 1024]:
    for nparts in range(1, 5):
      for policy in ["normal", "power2"]:
        factors = utilbox.any_part_split(ext, nparts, policy)
        print(ext, nparts, policy, len(factors))

  print("Success!")


def test2():
  for total in range(1, 10):
    permutations = utilbox.permutation(total)
    print(total, len(permutations))
  
  print("Success!")


def test3():
  for i in range(1, 7):
    for j in range(1, 7):
      choices = utilbox.choose_from(i, j)
      print(i, j, len(choices))
    
  print("Success!")


if __name__ == "__main__":
  test1()
  test2()
  test3()
