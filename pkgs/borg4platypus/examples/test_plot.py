import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.switch_backend('Tcl')
    plt.plot(range(10))
    plt.show(block=True)
    print('Done')