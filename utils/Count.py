def counter_filter(raw_dataset, count):
    import collections
    counter = collections.Counter([tk for tk in raw_dataset])
    counter = dict(filter(lambda x: x[1] >= count, counter.items()))
    return counter

if __name__ == '__main__':
    data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    print(counter_filter(data, 2))
