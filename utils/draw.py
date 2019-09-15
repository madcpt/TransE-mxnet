def draw(p1): 
    import matplotlib.pyplot as plt
    plt.figure('Draw')
    plt.plot(p1)  
    plt.draw()  
    plt.pause(1)  
    plt.savefig("easyplot01.jpg")
    plt.close()
