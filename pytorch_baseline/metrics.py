import csv, os, json

def write(row, out_csv):

  new = not os.path.exists(out_csv)

  with open(out_csv,"a",newline="") as f:
    w = csv.DictWriter(f,fieldnames=list(row.keys()))
    if new:
      w.writeheader()
    w.writerow(row)

  print(json.dumps(row,indent=2)[:2000])
  print("Wrote:", out_csv)
