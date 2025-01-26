import csv

iteration = 4
input_csv = 'data/train.csv'
output1_csv = 'data/training_'+str(iteration)+'.csv'
output2_csv = 'data/test_'+str(iteration)+'.csv'

with open(input_csv, 'r') as infile, open(output1_csv, 'w', newline='') as outfile1, open(output2_csv, 'w', newline='') as outfile2:
    reader = csv.reader(infile)
    writer1 = csv.writer(outfile1)
    writer2 = csv.writer(outfile2)
    header = next(reader)  # Read the header from the original file
    molecules = []
    for row in reader:
        molecules.append(row)

    for k in range(len(molecules)):
        row = molecules[k]


        features = row[1:200]
        ecfc = row[200:2248]
        fcfc = row[2248:4296]
        tot = 0
        for i in range(2048):
            tot+=float(ecfc[i])*float(fcfc[i])
        features.append(tot)
        label = row[-1]

        if k>2000:
            writer1.writerow(features + [label])
        if k<100 and k>10:

            writer2.writerow(features + [label])

print(f"Processed data has been saved to {output1_csv} and {output2_csv}")


