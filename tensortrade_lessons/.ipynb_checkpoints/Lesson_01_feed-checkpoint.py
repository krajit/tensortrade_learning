from tensortrade.feed import Stream, DataFeed

def counter():
    i = 0
    while True:
        yield i
        i += 1



# Create a stream with a source function
s = Stream.source(counter(), dtype='int').rename("Counter")
s2 = Stream.source(range(20,100), dtype='int').rename("Counter2")


# A list of stream becomes a datafeed. We typically operate on the feed
feed = DataFeed([s, s2])

# feed needs to be compiled, I guess
feed.compile()


# Retrieve the next 10 values from the stream
for i in range(10):
    print(feed.next())
