import csv, os
import numpy as np


new_path = os.path.join(os.getcwd())
os.chdir(new_path)
rows = []

def load():
	file_name = raw_input('Enter the file name with .csv extension: ')
	with open(file_name, 'r') as csvfile:
		csvreader = csv.reader(csvfile)

		for row in csvreader:
			rows.append(row)
		
	data = [[tuple(row[:len(row)-1]),row[len(row)-1]] for row in rows]		
	for (index, point) in enumerate(data):
		if data[index][1] == "tested_positive":
			data[index][1] = 1
		else:
			data[index][1] = -1

	return data


# data = load()
# X = np.array([x for (x,y) in data])
# X = X.astype(dtype=float)	
# Y = set([y for (x,y) in data])
# print(X[0])
# print(X[1])
# print(np.dot(X[0],X[1]))
# print(Y)	