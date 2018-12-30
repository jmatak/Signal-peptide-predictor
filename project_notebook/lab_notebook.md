# Signal peptide classifier
Lab notebook for course Applied Bioinformatics at KTH. 
Project Assignment - Predicting signal peptides
Authors: **Josip Matak, Josip Mrđen**
Year: 2018/2019, P2


# 2018, December 12th
----------------------
Added dataset into the project, it is in Fasta like format with folders representing the sequences which contain or don't contain signal peptides. Labels - n, h, c if peptide is present, other letters for no peptide Signal peptide mostly at the beginning of the protein sequence. 

# 2018, December 14th
----------------------
Added parsing of the files with BioPython to classes in Python (for loading the dataset). Furthermore, Added staticstics for the n, h, c part and splitting into training and testing set. Statistics include average length and standard deviation of each part. Serves as a module.

# 2018, December 18th
-------------------------
Created prototype of the model, saving and loading. Prototype is a neural network based, Sliding window implemented and data is augmented. Neural network has architecture of INPUT X 128 X 64 X 16 X OUTPUT. 

# 2018, December 19th 
--------------------------
Added human and wild boar proteomes

# 2018, December 25th
---------------------------
Started first tests on finished model. Implementation of logging and monitoring system. 

# 2018, December 26th
---------------------------
First training results with all the data provided are done, with sliding window. Splitting 0.8 of training and validation, 82% success. Results can be plotted. Added testing on already saved models. After training, program starts to evaluate on given data.


# 2018, December 27th
---------------------------
Arguments of command line added for robust training of data. Tests made for testing separately on transmembrane and non-transmembrane data. Report updated with final results. 
