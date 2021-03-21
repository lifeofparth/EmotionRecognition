import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def menu():
    print("Choose from the following:\n\
            1. angry\n\
            2. disgust\n\
            3. fear\n\
            4. happy\n\
            5. neutral\n\
            6. sad\n\
            7. surprise")
    choice = int(input("Enter a number"))
    return choice

def show_image(emotion):
    train_dir = '../../train'
    files = []
    for dirname, dirs, filenames in os.walk(train_dir, topdown=True):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))
    img = mpimg.imread(files[0])
    imgplot = plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    emotion = menu()
    show_image(emotion)
