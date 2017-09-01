# This for loading dictionaries of NRC,BingLui,senti140 Lexicons
def load_generic_dictionary(file_name):
    lexicon_dict={}
    f0=open(file_name,'r')
    line = f0.readline()
    while line:
        row= line.split("\t")
        line=f0.readline()
        lexicon_dict.update({row[0]: row[1]})
    f0.close()
    return lexicon_dict


# This is for loading dictionaries for Positive,Negative Lexicons
def load_word_dictionary(filename):
    f = open(filename, 'r')
    line = f.readline()
    lines = line.split(",");
    f.close()
    return lines


# This is for loading AFINN-96 and AFINN-111 dictionaries
def load_afinn_dictionary(filename):
    f = open(filename, 'r')
    afinn = {}
    line = f.readline()
    nbr = 0
    while line:
        try:
            nbr += 1
            l = line[:-1].split('\t')
            afinn[l[0]] = float(l[1]) / 4
            line = f.readline()
        except ValueError:
            break
    f.close()
    return afinn


# This is for loading sentiwordnet dictionaries
def load_senti_word_net_dictionary(filename):
    sentiWordnetDict={}
    tempDictionary={}
    f0=open(filename,'r')
    line = f0.readline()
    line_number=0
    while (line):
        line_number+=1
        if not ((line.strip()).startswith("#")):
            data = line.split("\t")
            wordTypeMarker = data[0]
            if (len(data) == 6):
                synsetScore = float(data[2])- float(data[3])
                synTermsSplit = data[4].split(" ")
                for synTermSplit in synTermsSplit:
                    synTermAndRank = synTermSplit.split("#")
                    synTerm = synTermAndRank[0] + "#" + wordTypeMarker
                    synTermRank = int(synTermAndRank[1]);
                    if not (tempDictionary.has_key(synTerm)):
                        tempDictionary[str(synTerm)]={}
                    tempDictionary[str(synTerm)][str(synTermRank)]=synsetScore
        line = f0.readline()
    for k1, v1 in tempDictionary.iteritems():
        score = 0.0
        sum = 0.0
        for k2, v2 in v1.iteritems():
            score += v2/ float(k2)
            sum += 1.0 /float(k2)
        score /= sum
        sentiWordnetDict[k1]=score
    f0.close()
    return sentiWordnetDict
