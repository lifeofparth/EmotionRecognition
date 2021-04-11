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
    if (int(emotion) == 1):
        train_dir = '../train/angry'
    elif (int(emotion) == 2):
        train_dir = '../train/disgust'
    elif (int(emotion) == 3):
        train_dir = '../train/fear'
    elif (int(emotion) == 4):
        train_dir = '../train/happy'
    elif (int(emotion) == 5):
        train_dir = '../train/neutral'
    elif (int(emotion) == 6):
        train_dir = '../train/sad'
    elif (int(emotion) == 7):
        train_dir = '../train/surprise'

    files = []
    for filename in os.listdir(train_dir):
        files.append(os.path.join(train_dir, filename))
    print(files[0])
    img = mpimg.imread(files[0])
    imgplot = plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    emotion = menu()
    show_image(emotion)
