
---

# Potato Disease Classification - API and Frontend

This project contains both the API backend and frontend for potato disease classification.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Lazy-Coder-03/potato-disease.git
   ```

2. Navigate to the API directory and install the Python dependencies:

   ```bash
   cd potato-disease/api
   pip install -r requirements.txt
   ```

3. Navigate to the frontend directory and create a `.env` file with the API URL:

   - **Using Command Line:**
     ```bash
     cd ../frontend
     echo REACT_APP_API_URL=http://localhost:8080/predict > .env
     ```
   
   - **Using GUI (Windows):**
     1. Open File Explorer.
     2. Go to the `potato-disease` folder.
     3. Navigate to the `frontend` subfolder.
     4. Right-click in the `frontend` folder and select `New` > `Text Document`.
     5. Rename the text file to `.env` (remove the `.txt` extension if visible).
     6. Edit the `.env` file and add the line:
        ```
        REACT_APP_API_URL=http://localhost:8080/predict
        ```
     7. Save the file.

4. Install the Node.js dependencies for the frontend:

   ```bash
   npm install
   ```

## Running the Backend

1. Start the backend server:

   ```bash
   cd ../api
   python main.py
   ```

   This will start the API server at `http://localhost:8080`.

## Running the Frontend

1. Start the frontend development server:

   ```bash
   cd ../frontend
   npm start
   ```

   This will start the frontend server at `http://localhost:3000`.

## Simultaneous Execution

To run the frontend and backend simultaneously:

1. Open a command prompt or terminal for the backend:

   ```bash
   cd potato-disease/api
   python main.py
   ```

2. Open another command prompt or terminal for the frontend:

   ```bash
   cd potato-disease/frontend
   npm start
   ```

Now you can access the frontend at `http://localhost:3000` and interact with the API backend at `http://localhost:8080`.

---
