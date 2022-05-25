
### Installation

#### Follow the following steps to run all the assignemt parts correctly on your machine.

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```

2. Create python virtual env and activate it
   ```sh
    python3 -m venv /path/to/new/virtual/environment
    source /path/to/new/virtual/environment/bin/activate
   ```

3. Install requirements
   ```sh
   pip install requirements.txt

   ```
4. Run the data pre-processing script
    ```sh
    python data_preprocessing.py

    ```
5. Train the random forrest classifier
    ```sh
    python random_forest.py train

    ```
6. Evaluate the random forrest classifier
    ```sh
    python random_forest.py eval

    ```
7. Train the CNN model
    ```sh
    python cnn.py train

    ```
6. Evaluate the CNN model
    ```sh
    python cnn.py eval

    ```
    
    
  Extra  Added  CNN Model: VGG16
  
8. Train the CNN2 model
    ```sh
    python cnn2.py train

    ```
9. Evaluate the CNN2 model
    ```sh
    python cnn2.py eval

    ```



