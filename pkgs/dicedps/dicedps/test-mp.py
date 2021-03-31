from pathos.multiprocessing import ProcessingPool
import time

data = (
    ['a', '2'], ['b', '4'], ['c', '6'], ['d', '8'],
    ['e', '1'], ['f', '3'], ['g', '5'], ['h', '7']
)

def mp_worker(x):
    inputs, the_time = x
    print(" Processs %s\tWaiting %s seconds" % (inputs, the_time))
    time.sleep(int(the_time))
    print(" Process %s\tDONE" % inputs)

def mp_handler():
    p = ProcessingPool(2)
    p.map(mp_worker, data)

if __name__ == '__main__':
    mp_handler()
