# Not-AiThinker
Facial recognition program with GUI using the facial recognition library!

How to use:

Through the program's GUI you can add images to the training directory, simply place images inside of a folder named after the person in the images, in the directory.

Using the "encode" function on the GUI will create a pickle file in the output directory containing information used to show the location of faces when presented in the GUI.

Validation can be used to make sure that the facial recognition is working, place images similar to your training data here to test your data.

The unknown folder is the dedicated place for images to test your newly encoded facial recognition model on!


Once the files are added to the directories and the encode button has created your pickle file, you may use the "recognize" button to test your facial recognition on images in the unkown folder,
and the "validate" button to test your facial recognition on images in the validation folder.
