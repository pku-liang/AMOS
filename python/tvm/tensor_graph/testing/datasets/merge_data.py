import argparse


def merge(in_files, out_file):
  with open(out_file, "a") as fout:
    for in_file in in_files:
      with open(in_file, "r") as fin:
        for line in fin:
          fout.write(line)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=str, default="dataset.txt")
  parser.add_argument("--files", type=str, nargs="*")

  args = parser.parse_args()
  merge(args.files, args.out)