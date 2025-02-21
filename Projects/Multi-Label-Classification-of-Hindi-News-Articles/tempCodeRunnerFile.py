reader = csv.reader(file)
for row in reader:
    for word in row:
        if len(word.strip()) != 0:
            stopwords.append(word.strip())