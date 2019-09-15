class Relation(object):
    def __init__(self, idx, tag, head, tail, embedding):
        self.idx = idx
        self.tag = tag
        self.head = head
        self.tail = tail
        self.embedding = embedding
