import cv2
# make sure you run brew install pytesseract
# import pytesseract
import easyocr
from PIL import Image
import pytesseract
import numpy as np
import cv2


def generate_range(number, range_number):
    range_list = [number]
    for i in range(1, range_number):
        range_list.append(number + i)
        range_list.append(number - i)

    range_list.sort()

    return range_list


def parse_image(image):
    reader = easyocr.Reader(["en"])
    text = reader.readtext(image=image, detail=0)
    print(text)


# # Load the number plate section image
# number_plate_section_image = cv2.imread(filename)
#
# # Read the image using Tesseract
# recognized_text = pytesseract.image_to_string(number_plate_section_image)
#
# # Print the recognized text
# print(recognized_text)


def main():
    filename = "1"

    # load the image
    image = cv2.imread("platenums/" + filename + ".jpg")
    # image = cv2.imread("outputs/3.jpg")

    print("easy ocr: with complete image")
    parse_image(image)

    # Create the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Sharpen the image
    print("easy ocr: with sharpened image")
    sharpened_image = cv2.filter2D(image, -1, kernel)
    parse_image(sharpened_image)

    # normalize the image
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    normalized_image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
    print("easy ocr: with normalized image")
    parse_image(normalized_image)

    # convert the image to grayscale
    grayscale_image = cv2.cvtColor(normalized_image, cv2.COLOR_RGB2GRAY)
    print("easy ocr: with grayscaled image")
    parse_image(grayscale_image)

    # apply a gaussian blur to the graysccaled image
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    print("easy ocr: with blurred image")
    parse_image(blurred_image)

    # threshold the image
    thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    print("easy ocr: with thresholded blurred image")
    parse_image(thresholded_image)

    # inverse the image
    inverse_image = 255 - image
    print("easy ocr: with inverse image")
    parse_image(inverse_image)

    return

    # find contours in the image
    contours = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    number_contour_coordinates = []

    for i, contour in enumerate(contours):
        # Calculate the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        if i > 0:
            px, py, pw, ph = cv2.boundingRect(contours[i - 1])
            if w in generate_range(pw, 3) and h in generate_range(ph, 3) and w > 10 and h > 30:
                number_contour_coordinates.append({"x": x, "y": y, "w": w, "h": h})
                number_contour_coordinates.append({"x": px, "y": py, "w": pw, "h": ph})

    for i, number_contour in enumerate(number_contour_coordinates):
        output_filename = "outputs/{}.jpg".format(i)
        cv2.imwrite(output_filename,
                    image[number_contour["y"]:number_contour["y"] + number_contour["h"],
                    number_contour["x"]: number_contour["x"] + number_contour["w"]])

        print("Text from image {} is {}".format(output_filename, parse_image(output_filename)))

        # Draw a rectangle around the black object
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save the image
        # cv2.imwrite("output-{}.jpg".format(i), image)

        # Filter out contours that are too small or too large
        # if w < 100 or h < 100:
        #     continue
        #
        # # Filter out contours that are not located in the center of the image
        # if x < image.shape[1] * 0.25 or x + w > image.shape[1] * 0.75 or y < image.shape[0] * 0.25 or y + h > \
        #         image.shape[0] * 0.75:
        #     continue

        # Assign the black object contour
        # black_object_contour = contour
        # break

    # filter the contours to identify the number plate
    # number_plate_contours = []
    #
    # for contour in contours:
    #     # calculate the bounding box of the container
    #     x, y, w, h = cv2.boundingRect(contour)
    #
    #     # filter out contours that are too small or too large
    #     # if w<100 or w > 500 or h < 50 or h > 200:
    #     #     continue
    #     # Filter out contours that are not located in the bottom half of the image
    #     # if y + h < image.shape[0] * 0.5:
    #     #     continue
    #
    #     # assign the number plate contour
    #     # number_plate_contours.append(contour)
    #     number_plate_contours.append([x, y, w, h])

    # extract the number plate sections from the image
    # number_plate_sections = []
    # if len(number_plate_contours) > 0:
    #     for nm_contour in number_plate_contours:
    #         # crop the image to the bounding box of the number plate
    #         number_plate_image = image[y:nm_contour[1] + nm_contour[3], x:nm_contour[0] + nm_contour[2]]
    #
    #         # split the number plate image into equal sections
    #         number_plate_sections = cv2.split(number_plate_image)
    #
    #         # Save the number plate sections
    #         for i, section in enumerate(number_plate_sections):
    #             cv2.imwrite("outputs/{}-{}.jpg".format(filename, i), section)


if __name__ == "__main__":
    main()
