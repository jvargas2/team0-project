import pandas as pd
import matplotlib.pyplot as plt
n_maxcol = 3000
cols = dict()
for i in range(26):
    cols[str(i)] = []
    pass
def convert(cell):
    """
    this function is for test
    """
    for i in range(26):
        cols[str(i)].append(str(cell).count(chr(ord('A')+i)))
        pass
    pass

df = pd.read_csv('test.txt',header=None)
df[0] = df[0].apply(convert)
result = pd.DataFrame()
for i in range(26):
    result[chr(ord('A')+i)] = cols[str(i)]
    pass
result['TYPE'] = df[1]
result.to_csv('result.csv')
print(result)
