import csv

iteration = 222
input_csv = 'train.csv'
output1_csv = 'traininggg.csv'
output2_csv = 'testinggg.csv'

with open(input_csv, 'r') as infile, open(output1_csv, 'w', newline='') as outfile1, open(output2_csv, 'w', newline='') as outfile2:
    reader = csv.reader(infile)
    writer1 = csv.writer(outfile1)
    writer2 = csv.writer(outfile2)
    header = next(reader)
    k = 0
    for row in reader:
        features = row[1:200]
        ecfc = row[200:2248]
        fcfc = row[2248:4296]

        co = [float(ecfc[i])+float(fcfc[i]) for i in range(2048)]
        features+=co
        label = row[-1]
        if k<500:
            writer2.writerow(features + [label])
        if k>500:
            writer1.writerow(features + [label])
        k+=1

print(f"Processed data has been saved")


