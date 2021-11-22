# import matplotlib.pyplot as plt
import time
# lossi = []
# for i in range(1000):
#   lossi.append(i)
#   plt.plot(lossi)
#   # plt.plot(history.history['val_accuracy'])
#   plt.title('model loss')
#   plt.ylabel('loss')
#   plt.xlabel('epoch')
#   plt.legend(['test'], loc='upper left')
#   plt.show()
#   time.sleep(0.3)

import matplotlib.pyplot as plt
import numpy as np

plt.ion()
fig, ax = plt.subplots(1,2)
# fig = plt.figure()
for i in range(50):
    y = np.random.random([10,1])
    ax[0].clear()
    ax[1].clear()
    ax[0].plot(y)
    ax[1].plot(-y)
    plt.show()
    plt.pause(0.001)
    # plt.clf()
    # fig.clear()
    time.sleep(0.02)