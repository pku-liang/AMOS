def rearrange_color():
  with open("output.txt", "w") as fout:
    with open("color_svg.txt", "r") as fin:
      for line in fin:
        lst = line.split()
        for x in lst:
          if len(x) > 0:
            fout.write(x + "\n")


def id_color():
  count = 0
  with open("id_color.txt", "w") as fout:
    with open("color_svg.txt", "r") as fin:
      for line in fin:
        count += 1
        fout.write(str(count) + ": \"" + line[:-1] + "\",\n")


id_color()
