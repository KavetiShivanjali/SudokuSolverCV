## End to End Sudoku solver project

Objective: Utilizes Convolutional Neural Networks (CNN) to extract the sudoku grid from an input image, predicts the digits using image recognition, employs backtracking to solve the sudoku puzzle, and finally displays the solved sudoku using OpenCV.

End product: A Streamlit application that accepts a sudoku image as input, processes it to display both the original unsolved sudoku (input image) and the solved sudoku image.

## Stream lit drop box UI for dropping the input sudoku images.

![Screenshot 2024-06-02 113626](https://github.com/KavetiShivanjali/SudokuSolverCV/assets/30626886/aa663de4-c7e7-4bf1-9aa4-d9d556e6612e)

## Displaying original Unsolved and Solved sudoku images.

![sudoku_app-3](https://github.com/KavetiShivanjali/SudokuSolverCV/assets/30626886/11d0b256-b62f-4dd1-bd42-47d4e54fdd9a)

## Methodology:

Step - 1 : Preprocessing of the input image [converting to binary image, warping, grid detection, horizontal lines detection, vertical lines detection, cell extraction, digit prediction]

Step - 2 : Storing the predicted digits into a matrix form and feeding to the Back tracking algorithm to solve the sudoku.

Step - 3 : Printing the solved sudoku on to the input image fed.

## Model configuration for digit recognition:

Data set source : Chars74k dataset

Data set size : 633 images belonging to 1 to 9 digits classes

Model : CNN

Model parameters : 17214154 ~17M

Model size : ~200 MB

activation function : [relu , softmax]

input size : 150 x 150

Optimizer : adam

epochs : 50

accuracy : ~99.5


