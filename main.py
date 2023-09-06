import sys, pygame
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.image as img
from tkinter import filedialog
from PIL import Image
import numpy as np
import pickle
import os

# digits = load_digits()
# plt.imshow(digits.images[0])
# plt.gray()
# plt.show()


def draw_point(surface, colour, thickness):
    x, y = pygame.mouse.get_pos()
    x = int(x / thickness[0]) * thickness[0]
    y = int(y / thickness[1]) * thickness[1] 
    rect = pygame.Rect((x, y), thickness)
    # pygame.draw.rect(surface, colour, rect)
    point = pygame.Surface(thickness)
    point.set_alpha(64)
    point.fill(colour)
    surface.blit(point, (x, y))
    pygame.display.update()


def run_paint():

    pygame.init()

    BLACK = 0, 0, 0
    WHITE = 255, 255, 255

    size = (320, 320)
    draw_thickness = [int(size[0] / 8), int(size[1] / 8)] # always 8x8
    screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
    screen.fill(BLACK)

    quit_paint = False
    while quit_paint == False:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                quit_paint = True

            if event.type == pygame.MOUSEBUTTONDOWN:
                # draw while the button is held down
                if event.button == 1:
                    while not (pygame.event.peek(pygame.MOUSEBUTTONUP) \
                      or pygame.event.peek(pygame.KEYUP)):
                        draw_point(screen, WHITE, draw_thickness)
                        pygame.time.wait(10)

            if event.type == pygame.KEYUP:
                # clear screen
                if event.key == pygame.K_c:
                    screen.fill(BLACK)
                    pygame.display.update()
                # save screen
                if event.key == pygame.K_s:
                    filename = filedialog.asksaveasfilename(
                        initialdir=r"C:\Users\cache\Python\MachineLearning\DigitClf\digits",
                        title="Save File",
                        filetypes=(("png files", ".png"), ("all files", ".*")),
                        defaultextension=".png"
                    )
                    pygame.image.save(screen, filename)
    pygame.quit()


def main():
    DIRECTORY = r"C:\Users\cache\Python\MachineLearning\DigitClf\digits"
    run_paint()
    
    IMG_W = 8
    IMG_H = 8
    
    for root, dirs, files in os.walk(DIRECTORY):
        X_test = np.zeros((len(files), IMG_W * IMG_H))
        for i, name in enumerate(files):
            print(name, end=", ")
            img = Image.open(os.path.join(DIRECTORY, name)) 
            img.thumbnail((IMG_W, IMG_H), Image.ANTIALIAS)
            X_test[i, :] = np.asarray(img)[:, :, 0].reshape(1, -1)
            # print(X_test.shape)

    f = open("clf.pickle", "rb")
    clf = pickle.load(f)
    f.close()
    print()
    print(clf.predict(X_test))


if __name__ == "__main__":
    main()
