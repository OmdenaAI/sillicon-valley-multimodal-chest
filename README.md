
# Silicon Valley Chapter: Multimodal Model for Pneumonia and Tuberculosis Detection from Chest X-Rays and EHR/Clinical Data

<img src="./src/visualizations/DALL·E 2024-04-17 11.30.33.png">

## Project Overview
Duration: 6-8 weeks

This project aims to develop a multimodal model that utilizes both chest X-rays and electronic health records (EHR) or clinical data to enhance the detection of pneumonia and tuberculosis. By integrating visual and textual information, the model seeks to improve diagnostic accuracy and support healthcare professionals in making more informed decisions.

## Objectives
- **Develop a multimodal AI model**: Leverage advanced machine learning techniques to integrate and analyze data from different modalities.
- **Improve detection accuracy**: Enhance the capability to accurately detect pneumonia and tuberculosis from combined datasets.
- **Support clinical decisions**: Provide a tool that aids clinicians in diagnosing these respiratory diseases more effectively.

## Datasets Used
- **Chest X-ray Images**: The project uses the [PadChest dataset](http://bimcv.cipf.es/bimcv-projects/padchest/), a public, labeled, large-scale chest X-ray dataset with over 160,000 high-resolution images. Each image is accompanied by patient details such as age and sex, and annotated with 174 radiographic findings, 19 differential diagnoses, and 104 anatomic locations.
- **EHR/Clinical Data**: The project explored the [MIMIC-III dataset](https://mimic.physionet.org/), which consists of de-identified health data from patients admitted to intensive care units. It includes clinical notes, lab results, medications, and other relevant health data.

## Technologies and Methods Used
- **NLP Techniques**: Explored Naive Bayes and Linear Support Vector Classifier for processing and analyzing textual data from EHR.
- **Computer Vision Model**: 
  - **Custom Loss Function**: Implemented a custom loss function using class frequency to handle data imbalance.
  - **Pretrained Models**: Utilized DenseNet pretrained model for advanced feature extraction from chest X-ray images.
- **Machine Learning Frameworks**: [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/)
- **Data Handling**: Python, [Pandas](https://pandas.pydata.org/)
- **Imaging Libraries**: [OpenCV](https://opencv.org/), [PIL](https://python-pillow.org/)

## Contributors
- Project led and managed by the Omdena Silicon Valley Chapter Lead Nishrin Kachwala.
- Contributions from Omdena members specializing in AI, machine learning, and data science.

## Contribution Guidelines for Collaborators
- Have a Look at the [project structure](#project-structure) and [folder overview](#folder-overview) below to understand where to store/upload your contribution
- If you're creating a task, Go to the task folder and create a new folder with the below naming convention and add a README.md with task details and goals to help other contributors understand
    - Task Folder Naming Convention : _task-n-taskname.(n is the task number)_  ex: task-1-data-analysis, task-2-model-deployment etc.
    - Create a README.md with a table containing information table about all contributions for the task.
- If you're contributing for a task, please make sure to store in relavant location and update the README.md information table with your contribution details.
- Make sure your File names(jupyter notebooks, python files, data sheet file names etc) has proper naming to help others in easily identifing them.
- Please restrict yourself from creating unnessesary folders other than in 'tasks' folder (as above mentioned naming convention) to avoid confusion. 

## Project Structure

    ├── LICENSE
    ├── README.md          <- The top-level README for developers/collaborators using this project.
    ├── original           <- Original Source Code of the challenge hosted by omdena. Can be used as a reference code for the current project goal.
    │ 
    │
    ├── reports            <- Folder containing the final reports/results of this project
    │   └── README.md      <- Details about final reports and analysis
    │ 
    │   
    ├── src                <- Source code folder for this project
        │
        ├── data           <- Datasets used and collected for this project
        │   
        ├── docs           <- Folder for Task documentations, Meeting Presentations and task Workflow Documents and Diagrams.
        │
        ├── references     <- Data dictionaries, manuals, and all other explanatory references used 
        │
        ├── tasks          <- Master folder for all individual task folders
        │
        ├── visualizations <- Code and Visualization dashboards generated for the project
        │
        └── results        <- Folder to store Final analysis and modelling results and code.
--------

## Folder Overview

- Original          - Folder Containing old/completed Omdena challenge code.
- Reports           - Folder to store all Final Reports of this project
- Data              - Folder to Store all the data collected and used for this project 
- Docs              - Folder for Task documentations, Meeting Presentations and task Workflow Documents and Diagrams.
- References        - Folder to store any referneced code/research papers and other useful documents used for this project
- Tasks             - Master folder for all tasks
  - All Task Folder names should follow specific naming convention
  - All Task folder names should be in chronologial order (from 1 to n)
  - All Task folders should have a README.md file with task Details and task goals along with an info table containing all code/notebook files with their links and information
  - Update the [task-table](./src/tasks/README.md#task-table) whenever a task is created and explain the purpose and goals of the task to others.
- Visualization     - Folder to store dashboards, analysis and visualization reports
- Results           - Folder to store final analysis modelling results for the project.


