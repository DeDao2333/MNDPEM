
from numpy.core._multiarray_umath import array


NODE_SHAPE = {
    'o': [0, 2, 3, 4, 8, 10, 11, 12, 14, 15, 16, 18, 20, 21, 23, 24, 28, 29, 30, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 55, 58, 59, 61],
    's': [1, 5, 6, 7, 9, 13, 17, 19, 22, 25, 26, 27, 31, 32, 41, 48, 54, 56, 57, 60]
}

EDGE_SHAPE = {

}

POS = {0: array([-0.12335322, -0.19201138]), 1: array([0.17485701, 0.35671631]),
           2: array([0.07919733, -0.38996203]),
           3: array([-0.25453724, -0.14257345]), 4: array([-0.70788629, -0.39261314]),
           5: array([0.34355124, 0.87321401]),
           6: array([0.29277757, 0.77016971]), 7: array([0.13246678, 0.20705315]), 8: array([-0.15788977, -0.09206864]),
           9: array([0.3918546, 0.76624491]), 10: array([-0.0367484, -0.28514571]),
           11: array([-0.73136899, -0.29413094]),
           12: array([-0.25099427, -0.69910759]), 13: array([0.37089148, 0.71279034]),
           14: array([-0.15219012, -0.35684928]),
           15: array([-0.34627505, -0.16290967]), 16: array([-0.05778825, -0.3721889]),
           17: array([0.32353267, 0.64316626]),
           18: array([-0.32107843, -0.28403897]), 19: array([0.21628986, 0.27836009]),
           20: array([-0.04535174, -0.19664098]),
           21: array([-0.32645092, -0.37411763]), 22: array([0.49275339, 0.82439663]),
           23: array([-0.36080489, -0.06072352]),
           24: array([-0.38289132, -0.32861059]), 25: array([0.44273158, 0.52166417]),
           26: array([0.3888508, 0.41382776]),
           27: array([0.29742865, 0.42469334]), 28: array([0.03747214, 0.05960392]),
           29: array([-0.29685266, -0.42238225]),
           30: array([0.14466974, 0.05244146]), 31: array([0.20270162, 0.85016152]),
           32: array([0.59269948, 0.81615464]),
           33: array([-0.15050031, -0.44591285]), 34: array([0.00492376, -0.51989721]),
           35: array([-0.47929773, -0.6186459]),
           36: array([-0.08272773, 0.05810827]), 37: array([-0.12488617, -0.308844]),
           38: array([0.0201549, -0.44306772]),
           39: array([0.03103741, 0.41470912]), 40: array([-0.09308744, -0.14669876]),
           41: array([0.27261886, 0.60239032]),
           42: array([0.03752446, -0.19113951]), 43: array([-0.1159973, -0.56173193]),
           44: array([0.12974491, -0.39140766]),
           45: array([-0.3117144, -0.22640841]), 46: array([0.00202912, -0.8055585]),
           47: array([0.04190661, -0.09748699]),
           48: array([0.23416479, 0.94125718]), 49: array([0.1099198, -0.76299817]),
           50: array([-0.19647236, -0.28978145]),
           51: array([-0.47278444, -0.28816666]), 52: array([-0.11851139, -0.39826541]),
           53: array([-0.08187792, -0.7502496]),
           54: array([0.23290362, 0.5117531]), 55: array([-0.57883631, -0.17574226]), 56: array([0.3226477, 1.]),
           57: array([0.22828492, 0.69765793]), 58: array([0.25727796, -0.57987671]),
           59: array([-0.27251882, -0.03056126]),
           60: array([0.80375535, 0.85400456]), 61: array([-0.02194624, -0.57202305])}

CLMC_PREDICTED_EDGES = {
       'true': [(0, 40), (1, 54), (6, 56), (13, 57), (17, 27), (19, 30), (23, 51), (36, 37), (38, 43), (42, 50), (43, 53)],
       'false': [(5, 6), (11, 33), (13, 27), (20, 43), (22, 56), (43, 45)]
}

CLMC_PREDICTED_LABEL = {
       'true': [],
       'false': [39, 30]
}

MNDP_PREDICTED_EDGES = {
       'true': [],
       'false': []
}

MNDP_PREDICTED_LABEL = {
       'true': [],
       'false': [28, 30, 39]
}

OUR_PREDICTED_EDGES = {
       'false': [(1, 56), (2, 18), (2, 59), (8, 36), (10, 23), (14, 35), (17, 26), (25, 54), (26, 57), (28, 36), (28, 39), (29, 59)],
       'true': [(1, 54), (19, 30), (38, 43)]
}

OUR_PREDICTED_LABEL = {
       'true': [],
       'false': [30]
}

MISSING_NETWORK = {
    'exits': [],
    'removed': [(40, 0), (54, 1), (9, 6), (56, 6), (57, 6), (19, 7), (40, 7), (54, 7), (6, 9), (57, 9), (47, 10), (41, 13), (57, 13), (33, 14), (38, 14), (18, 15), (45, 15), (38, 16), (27, 17), (15, 18), (7, 19), (30, 19), (47, 20), (50, 20), (37, 21), (51, 23), (17, 27), (30, 28), (19, 30), (28, 30), (47, 30), (14, 33), (50, 33), (37, 36), (21, 37), (36, 37), (14, 38), (16, 38), (43, 38), (0, 40), (7, 40), (13, 41), (50, 42), (38, 43), (53, 43), (15, 45), (49, 46), (10, 47), (20, 47), (30, 47), (46, 49), (20, 50), (33, 50), (42, 50), (23, 51), (43, 53), (1, 54), (7, 54), (6, 56), (6, 57), (9, 57), (13, 57)],
}