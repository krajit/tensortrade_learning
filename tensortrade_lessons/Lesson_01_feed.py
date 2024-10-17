from tensortrade.feed import Stream, DataFeed

def counter():
    i = 0
    while True:
        yield i
        i += 1

# Create a stream with a source function
s = Stream.source(counter(), dtype='int').rename("Counter")
feed = DataFeed([s])
feed.compile()


# Retrieve the next 10 values from the stream
for i in range(10):
    print(feed.next())
