def draw(p1, display=False, pic_name = "easyplot01.png"): 
    import matplotlib.pyplot as plt
    plt.figure('Draw')
    plt.plot(p1)
    plt.savefig(pic_name)
    if display:
        plt.draw()  
        plt.pause(5)  
    plt.close()
