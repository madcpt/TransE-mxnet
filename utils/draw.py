def draw(p1): 
    import matplotlib.pyplot as plt
    plt.figure('Draw')
    plt.plot(p1)  
    plt.savefig("easyplot01.png")
    plt.draw()  
    plt.pause(1)  
    plt.close()
