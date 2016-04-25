import csv, sys, argparse

class myDict(dict):
	def __missing__(self, key):
		return None

def removewords(file1, file2):
	list1 = []
	list2 = myDict()
	with open(file2, 'r') as shortlist:
		reader = csv.reader(shortlist)
		for row in reader:
			for i, v in enumerate(row):
				print i, v
				if i == 0:
					list2[v] = 1

	print list2

	with open(file1, 'r') as longlist:
		reader = csv.reader(longlist)
		for row in reader:
			for i, v in enumerate(row):
				if i == 0:
					if list2[v] is None:
						list1.append(row)

	print list1
	with open(file1, 'w+') as longlist:
		writer = csv.writer(longlist)
		for i, value in enumerate(list1):
			print i, value
			writer.writerow(value)



removewords('dictionary2.csv', 'stopwords_long.csv')