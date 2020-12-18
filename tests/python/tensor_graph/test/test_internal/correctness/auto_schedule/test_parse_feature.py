import json


def test(filename):
  with open(filename, "r") as fin:
    for line in fin:
      print(line)
      obj = json.loads(line)
      print(obj.keys())
      print(len(obj["features"]))
      print(len(obj["features"][0]))
      print(obj["features"][0][15])

if __name__ == "__main__":
  test("autoschedule_log_profile.txt")