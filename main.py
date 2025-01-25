import csv

# Open the original file and process the data
input_csv = 'data/train.csv'
output_csv = 'data/test13.csv'

with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)  # Read the header from the original file
    molecules = []
    for row in reader:
        molecules.append(row)

    for row in molecules[10:100]:
        co = row[1:2248]
        ecfc = row[200:2248]
        tot = 0
        for element in ecfc:
            if element!=0:
                tot+=1
        co.append(tot)
        label = row[-1]
        writer.writerow(co + [label])

print(f"Processed data has been saved to {output_csv}")


