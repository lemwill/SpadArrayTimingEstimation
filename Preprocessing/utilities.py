import sys

def progressbar(it, count = 0, stride = 0, prefix = "", size = 50):
    """Progress bar iterator, adds a nice progress bar in the console.
       file iterators cannot know how many items will be, so count
       must be set. Otherwise an exception will be thrown
       modified from example at
       http://code.activestate.com/recipes/576986-progress-bar-for-console-programs-as-iterator/
       """
    if(count == 0):
        try:
            count = len(it)
        except:
            print "Cannot find length, no progress bar"
            for i, item in enumerate(it):
                yield item

    if(stride == 0):
        try:
            if(count > 100):
                stride = count / 100
            else:
                stride = 1
        except:
            print "Problem with stride??? exiting"
            exit(1)

    def _show(_i):
        x = int(size*_i/count)
        sys.stdout.write("%s [%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), _i, count))
        sys.stdout.flush()


    _show(0)
    for i, item in enumerate(it):
        yield item
        if(i % stride == 0 or (i-1) == count):
            _show(i+1)

    sys.stdout.write("%s [%s%s] %i/%i\r" % (prefix, "#"*size, "."*(0), count, count))
    #sys.stdout.write("\n")
    sys.stdout.flush()