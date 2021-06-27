def load_file(test_set_percentage):
    f1 = open("../data/yelp_lp/yelp_ub.train_0.8_lr.dat", "w")
    f2 = open("../data/yelp_lp/yelp_ub.test_0.8_new.dat", "w")
    i = 0
    with open('../data/yelp_triple.dat') as infile:
        for line in infile.readlines():
            u, b, label = [int(item) for item in line.strip().split()]
            # print("u: ", u, "b: ", b, "label: ", label)
            if i % round(100/test_set_percentage) == 0:
                f2.write('{} {} {}\n'.format(u, b, label))
            else:
                f1.write('{} {} {}\n'.format(u, b, label))
            i += 1


if __name__ == '__main__':
    load_file(20)
