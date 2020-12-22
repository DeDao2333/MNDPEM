import matplotlib.pyplot as plt


def display_rate_football():
    # football
    MNDP_M = [0.9254, 0.9008, 0.8883, 0.8517, 0.8516, 0.8297, 0.8080, 0.7857]
    MNDP_EM = [0.9263, 0.9242, 0.9074, 0.9036, 0.9036, 0.9021, 0.8679, 0.8258]
    CLMC = [0.9232, 0.9239, 0.9232, 0.9232, 0.9043, 0.8657, 0.8391, 0.8211]
    GEMSEC = [0.8923, 0.8835, 0.8866, 0.8678, 0.8652, 0.8438, 0.8032, 0.7933]
    # BigClam = [0.3698, 0.3554, 0.3331, 0.3167, 0.2845, 0.2669]
    DANMF = [0.9038, 0.8915, 0.9026, 0.8868, 0.9006, 0.8746, 0.8492, 0.8289]
    Louvain = [0.8903, 0.8850, 0.8795, 0.8582, 0.8513, 0.8415, 0.8288, 0.8213]

    rate = ['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%']

    plt.figure(figsize=(6, 4.3))
    plt.plot(rate, GEMSEC, c='#6B6B6B', marker='o', ms=9, label='GEMSEC')
    # plt.plot(rate, BigClam, marker='o', mec='r', mfc='w')
    plt.plot(rate, DANMF, c='#228B22', marker='o', ms=9, label='DANMF')
    plt.plot(rate, Louvain, c='#DD6D22', marker='o', ms=9, label='Louvain')
    plt.plot(rate, CLMC, c='#A25EA2', marker='o', ms=9, label='CLMC')
    plt.plot(rate, MNDP_M, c='#A0522D', marker='<', ms=9, label='MNDP-M')
    plt.plot(rate, MNDP_EM, c='#2B91D5', marker='D', ms=8, label='MNDPEM')
    # fig.suptitle('Categorical Plotting')
    plt.xlabel('Different ratio of missing edges', fontsize=14)
    plt.ylabel('NMI', fontsize=14)
    plt.title('Football', fontsize=14)
    plt.legend(loc="lower left", fontsize=9)
    plt.savefig("football.png")
    plt.show()


def display_rate_karate():
    rate = ['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%']
    # karate
    MNDP_M = [1.0000, 0.8372, 0.8365, 0.8365, 0.7308, 0.6499, 0.5739, 0.5159]
    MNDP_EM = [1.0000, 1.0000, 1.0000, 0.8372, 0.8372, 0.7308, 0.7329, 0.6459]
    CLMC = [1.0000, 1.0000, 0.8365, 0.8333, 0.8333, 0.5743, 0.4175, 0.3476]
    GEMSEC = [1.0000, 1.0000, 0.8372, 0.8365, 0.7329, 0.6766, 0.6499, 0.5801]
    BigClam = [1.0000, 0.8372, 0.8255, 0.8255, 0.7201, 0.6494, 0.5883, 0.5618]
    DANMF = [1.0000, 0.8365, 0.8365, 0.8372, 0.7329, 0.6169, 0.4765, 0.4177]
    Louvain = [0.7071, 0.7071, 0.7071, 0.6873, 0.6175, 0.5804, 0.5214, 0.4923]

    plt.figure(figsize=(6, 4.3))
    plt.plot(rate, GEMSEC, c='#6B6B6B', marker='o', ms=9, label='GEMSEC')
    plt.plot(rate, BigClam, c='#6A5ACD', marker='o', ms=9, label='BigClam')
    plt.plot(rate, DANMF, c='#228B22', marker='o', ms=9, label='DANMF')
    plt.plot(rate, Louvain, c='#DD6D22', marker='o', ms=9, label='Louvain')
    plt.plot(rate, CLMC, c='#A25EA2', marker='o', ms=9, label='CLMC')
    plt.plot(rate, MNDP_M, c='#A0522D', marker='<', ms=9, label='MNDP-M')
    plt.plot(rate, MNDP_EM, c='#2B91D5', marker='D', ms=8, label='MNDPEM')
    # fig.suptitle('Categorical Plotting')
    plt.xlabel('Different ratio of missing edges', fontsize=14)
    plt.ylabel('NMI', fontsize=14)
    plt.title('Zachary’s karate club', fontsize=14)
    plt.legend(loc="lower left", fontsize=9)
    plt.savefig("karate.png")
    plt.show()


def display_rate_dolphins():
    rate = ['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%']

    # Dolphins
    MNDP_M = [0.8888, 0.8873, 0.8141, 0.7532, 0.7011, 0.6544, 0.6270, 0.4855]
    MNDP_EM = [0.8991, 0.8936, 0.8865, 0.8870, 0.8870, 0.8870, 0.8083, 0.6147]
    CLMC = [0.8809, 0.8809, 0.8809, 0.8809, 0.8809, 0.8782, 0.7769, 0.5562]
    GEMSEC = [0.8888, 0.8141, 0.8141, 0.7769, 0.7036, 0.6333, 0.6270, 0.5752]
    BigClam = [0.8888, 0.8888, 0.7656, 0.7783, 0.6333, 0.5673, 0.5495, 0.4478]
    DANMF = [0.8141, 0.7543, 0.6040, 0.5660, 0.5449, 0.4809, 0.4530, 0.4064]
    Louvain = [0.8141, 0.7532, 0.6041, 0.5665, 0.5400, 0.5199, 0.4877, 0.4685]

    plt.figure(figsize=(6, 4.3))
    plt.plot(rate, GEMSEC, c='#6B6B6B', marker='o', ms=9, label='GEMSEC')
    plt.plot(rate, BigClam, c='#6A5ACD', marker='o', ms=9, label='BigClam')
    plt.plot(rate, DANMF, c='#228B22', marker='o', ms=9, label='DANMF')
    plt.plot(rate, Louvain, c='#DD6D22', marker='o', ms=9, label='Louvain')
    plt.plot(rate, CLMC, c='#A25EA2', marker='o', ms=9, label='CLMC')
    plt.plot(rate, MNDP_M, c='#A0522D', marker='<', ms=9, label='MNDP-M')
    plt.plot(rate, MNDP_EM, c='#2B91D5', marker='D', ms=8, label='MNDPEM')
    # fig.suptitle('Categorical Plotting')
    plt.xlabel('Different ratio of missing edges', fontsize=14)
    plt.ylabel('NMI', fontsize=14)
    plt.title('Dolphin social network', fontsize=14)
    plt.legend(loc="lower left", ncol=2, fontsize=9)
    plt.savefig("Dolphins.png")
    plt.show()


if __name__ == '__main__':
    display_rate_karate()
    display_rate_football()
    display_rate_dolphins()
